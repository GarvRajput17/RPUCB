"""
MIND routing -- Li et al., "Multi-Interest Network with Dynamic Routing"
(Alibaba, CIKM'19). New file.

Two components, both of which DCM later replaces or drops, which is what
makes `mind` vs `dcm` a routing-rule comparison:

  MINDRouting          Behaviour-to-Interest (B2I) dynamic routing, Sec 3.3
  LabelAwareAttention  training-time interest selection, Sec 3.4

Where B2I differs from vanilla capsule routing, and from DCM:

  shared bilinear map    One matrix S for every (behaviour, interest) pair
                         rather than a separate W_ij per pair, because the
                         behaviour set is variable-length and per-pair
                         matrices would not generalise across users.

  random logit init      Routing logits start from a normal draw rather
                         than zeros. With a shared S, zero-initialised
                         logits make every interest capsule identical for
                         all iterations, so the symmetry never breaks.

  soft routing           Weights are a softmax over interests. DCM's SAR
                         replaces this with hard single assignment, and
                         DCM's VA-FPI replaces the random initialisation
                         entirely -- those two substitutions are precisely
                         what the `mind -> dcm` cross-backbone row is
                         about.

  adaptive K_u           K_u = max(1, min(K_max, floor(log2 |I_u|))). With
                         the history capped at 50, log2(50) ~ 5.64, so
                         most users get 5 interests rather than the 7 DCM
                         uses -- the capacity asymmetry §5's writeup
                         caveat already records. Implemented as a fixed
                         K_max tensor plus a per-user validity mask, so
                         batches stay rectangular.

Deviation, deliberate and for the same reason DCM's docstring gives for
VA-FPI: the routing logits are drawn once at construction into a fixed
buffer rather than resampled on every forward pass. Positives and
negatives are scored in separate forward passes within one loss term, and
resampling would hand the same user two incompatible interest sets inside
a single comparison. The draw still breaks capsule symmetry, which is the
property MIND needs from it; it simply does not vary per call.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing_common import NEG_SCORE, slot_valid_from_counts, squash


def adaptive_interest_count(n_valid, k_max):
    """
    K_u = max(1, min(k_max, floor(log2 |I_u|))) -> [B] long.

    |I_u| is the size of the user's (capped) history bag, matching the
    gameplan's arithmetic: log2 of a 50-item history is about 5.6, giving
    5 interests for a typical user.
    """
    n = n_valid.float().clamp(min=1.0)
    k = torch.floor(torch.log2(n)).long()
    return k.clamp(min=1, max=k_max)


class MINDRouting(nn.Module):
    """
    Args:
        embed_dim: width of the behaviour (item) embeddings.
        k_max: interest capsule ceiling. 7, matching DCM's fixed K so the
            two backbones share a capacity ceiling.
        routing_iters: 3.
        adaptive_k: True for the faithful baseline. False pins every user
            at k_max interests -- and with k_max=1 gives the collapsed
            `mind_rpucb` variant, where routing reduces to weighted
            pooling plus squash.
        max_hist_len: history cap, needed to size the logit buffer.
        logit_std: standard deviation of the initial routing logits.
        logit_seed: seed for the one-time logit draw.

    Forward:
        item_embeds: [B, L, d]
        valid_mask:  [B, L] bool

    Returns:
        interests:  [B, k_max, d]
        slot_valid: [B, k_max] bool, True for the first K_u slots
    """

    def __init__(
        self,
        embed_dim,
        k_max=7,
        routing_iters=3,
        adaptive_k=True,
        max_hist_len=50,
        logit_std=1.0,
        logit_seed=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_max = k_max
        self.routing_iters = routing_iters
        self.adaptive_k = adaptive_k

        # Shared bilinear mapping S (B2I).
        self.S = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.S)

        # Symmetry-breaking logits, drawn once. Small enough to keep in the
        # state_dict, and doing so pins the routing initialisation to the
        # checkpoint rather than to whatever seed happens to be live.
        generator = torch.Generator().manual_seed(logit_seed)
        logits = torch.randn(max_hist_len, k_max, generator=generator) * logit_std
        self.register_buffer("b_init", logits, persistent=True)

    def forward(self, item_embeds, valid_mask):
        B, L, _ = item_embeds.shape

        n_valid = valid_mask.sum(dim=1).clamp(min=1)
        if self.adaptive_k:
            k_u = adaptive_interest_count(n_valid, self.k_max)
        else:
            k_u = torch.full_like(n_valid, self.k_max)
        slot_valid = slot_valid_from_counts(k_u, self.k_max)          # [B, K]

        # Behaviours mapped through the shared matrix once; the routing
        # loop reuses this.
        Se = torch.einsum("bld,de->ble", item_embeds, self.S)         # [B, L, d]

        # Pairs that do not exist: padded history slots, or interest slots
        # above this user's K_u.
        pair_valid = valid_mask.unsqueeze(2) & slot_valid.unsqueeze(1)  # [B, L, K]
        neg = torch.full(
            (1,), NEG_SCORE, device=item_embeds.device, dtype=item_embeds.dtype
        )

        b = self.b_init[:L].unsqueeze(0).expand(B, -1, -1).to(item_embeds.dtype)

        interests = None
        for _ in range(self.routing_iters):
            logits = torch.where(pair_valid, b, neg.expand_as(b))
            # Softmax over interests: each behaviour distributes its weight
            # across capsules. This is the step SAR replaces with a hard
            # argmax assignment.
            w = F.softmax(logits, dim=2) * pair_valid.to(b.dtype)

            z = torch.einsum("blk,bld->bkd", w, Se)                   # [B, K, d]
            interests = squash(z, dim=-1)

            # Agreement update.
            b = b + torch.einsum("bkd,bld->blk", interests, Se)

        interests = interests * slot_valid.unsqueeze(-1).to(interests.dtype)
        return interests, slot_valid


class LabelAwareAttention(nn.Module):
    """
    MIND Sec 3.4: at training time the target item selects which interests
    to read, producing one user vector for the loss.

        v_u = V_u * softmax(p * V_u^T e_target)

    `power` is MIND's tunable attention hardness. p=1 is plain attention;
    large p approaches a hard argmax over interests, which is what DCM's
    condition-association rule does explicitly. Keeping it soft here is
    the faithful choice and is what makes `mind -> dcm` a comparison
    between two selection rules rather than one rule under two names.

    MIND writes the exponent as pow(x, p). Applying a real power to
    similarities that can be negative is not well defined, so this uses
    the standard reimplementation -- a temperature on the logits -- which
    has the same limiting behaviour in p.

    At evaluation there is no label, so the model falls back to max over
    interests (§5's locked scoring rule, shared with DCM).
    """

    def __init__(self, power=2.0):
        super().__init__()
        self.power = float(power)

    def forward(self, interests, target_embed, slot_valid=None):
        """
        interests:    [B, K, d]
        target_embed: [B, d]
        Returns:      [B, d]
        """
        logits = torch.einsum("bkd,bd->bk", interests, target_embed) * self.power
        if slot_valid is not None:
            logits = logits.masked_fill(~slot_valid, NEG_SCORE)
        attn = F.softmax(logits, dim=1)
        return torch.einsum("bk,bkd->bd", attn, interests)