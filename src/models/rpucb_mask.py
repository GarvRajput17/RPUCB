"""
The RP-UCB mask, implemented once.

    m = clamp( sigmoid(w) + beta * sigmoid(gamma) * max(0, ln(n_bar / N)), 0, 1 )

Before this refactor the formula existed in three places and two of them
agreed. `deepcf_rpucb.py` used `sigmoid(w + gamma * explore)` -- gamma
inside the sigmoid, no beta, gamma initialised to zeros so the exploration
term started at exactly zero -- while `rpucb_attn_full.py` and
`pinterest_dcm_rpucb.py` used the canonical form above with
`gamma_init=2.0`. That made model 2 of the matrix test a different
mechanism from models 5, 6, 8, 9 and 10 without anything in the code
saying so. One module removes the possibility.

Handles all three mask granularities the matrix needs:

    n_slots = 1   shared d-dim mask (DeepCF both sides, MIND user side,
                  every item side, dcm_rpucb_d)
    n_slots = K   per-slot masks indexed (entity, k) (dcm_rpucb_multi)
    dim = K*d     one wide mask over a concatenated embedding
                  (dcm_rpucb_kd) -- expressed as n_slots=1 with a wider dim

Initialisation note. `mask_init` is the value `sigmoid(w)` takes at
construction, and it is a substantive modelling choice rather than a
detail. At the default of 0.5, a user whose interaction count is at or
above the dataset mean gets no exploration bonus, so their mask is exactly
0.5 -- and because this codebase masks the user side *and* the item side,
their score is built from representations attenuated to 0.25x relative to
the unmasked baseline. The masked model therefore starts behind its own
control, worst for the users with the most evidence, and has to spend
early epochs recovering. Setting mask_init close to 1.0 instead makes the
masked model start as a functional identity of its baseline and learn to
close dimensions from there. Both are available; 0.5 is the default
because it reproduces the pre-refactor behaviour.

The exploration term saturates the upper clamp for sparse users -- with
n_bar around 30 and N=1, the bonus alone exceeds 1 -- and `torch.clamp`
passes zero gradient outside its range. So the learnable parts of the mask
receive no signal at all from the sparsest users, which is worth knowing
before interpreting a result about sparsity. `saturation()` reports how
much of the batch is in that regime.
"""

import math

import torch
import torch.nn as nn


class RPUCBMask(nn.Module):
    """
    Args:
        num_entities: users or items, whichever side this mask gates.
        dim: width of the representation being masked. For the
            concatenated-head variant this is K*d, not d.
        counts: LongTensor [num_entities], N per entity. Interaction counts
            come from the *train* split only -- deriving n_bar from a split
            that includes val or test would leak.
        n_slots: 1 for a shared mask, K for per-slot masks.
        gamma_init: pre-sigmoid init of the global confidence vector. 2.0
            gives sigmoid(gamma) ~ 0.88, i.e. exploration active from step
            one. Do not set to 0.0 -- that reproduces the old DeepCF bug
            where the bonus starts inert.
        beta: exploration scale. Tuned on validation, per family (§3).
        mask_init: value of sigmoid(w) at construction. See module docstring.
    """

    # Read by BaseCF.init_weights, which would otherwise re-randomise `w`
    # to normal(0, 0.01) and silently override mask_init.
    _skip_default_init = True

    def __init__(
        self,
        num_entities,
        dim,
        counts=None,
        n_slots=1,
        gamma_init=2.0,
        beta=1.0,
        mask_init=0.5,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.dim = dim
        self.n_slots = n_slots
        self.beta = float(beta)
        self.mask_init = float(mask_init)

        self.w = nn.Embedding(num_entities * n_slots, dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(gamma_init)))

        if counts is None:
            counts = torch.ones(num_entities, dtype=torch.long)
        counts = counts.long()
        # persistent=False: reconstructible from the dataset, and keeping it
        # out of the state_dict matters when 150 checkpoints are being
        # written to shared storage.
        self.register_buffer("counts", counts, persistent=False)
        self.n_bar = max(1.0, float(counts.float().mean().item()))

        self.reset_parameters()

    def reset_parameters(self):
        p = min(max(self.mask_init, 1e-4), 1.0 - 1e-4)
        w0 = math.log(p / (1.0 - p))
        nn.init.normal_(self.w.weight, mean=w0, std=0.01)

    def exploration(self, ids):
        """max(0, ln(n_bar / N)) per entity -> [B]."""
        n = self.counts[ids].float().clamp(min=1.0)
        n_bar = torch.as_tensor(self.n_bar, device=n.device, dtype=n.dtype)
        return torch.clamp(torch.log(n_bar / n), min=0.0)

    def forward(self, ids):
        """
        ids: [B] entity ids.
        Returns: [B, n_slots, dim]. Callers with n_slots == 1 can either
        broadcast against [B, K, d] directly or use `flat()`.
        """
        B = ids.size(0)

        if self.n_slots == 1:
            w = self.w(ids).unsqueeze(1)                                # [B,1,dim]
        else:
            offsets = torch.arange(self.n_slots, device=ids.device)
            flat_ids = ids.unsqueeze(1) * self.n_slots + offsets.unsqueeze(0)
            w = self.w(flat_ids)                                        # [B,K,dim]

        explore = self.exploration(ids).view(B, 1, 1)                   # [B,1,1]
        sigma_gamma = torch.sigmoid(self.gamma).view(1, 1, -1)          # [1,1,dim]

        return (torch.sigmoid(w) + self.beta * explore * sigma_gamma).clamp(0.0, 1.0)

    def flat(self, ids):
        """[B, dim] for the n_slots == 1 case."""
        assert self.n_slots == 1, "flat() requires a shared (n_slots=1) mask"
        return self.forward(ids).squeeze(1)

    @torch.no_grad()
    def saturation(self, ids, tol=1e-6):
        """
        Fraction of mask entries pinned at the upper clamp, where gradient
        to `w` and `gamma` is exactly zero. Log this during the dry run:
        if it is high, a null result on the sparse datasets says more about
        the clamp than about RP-UCB.
        """
        m = self.forward(ids)
        return (m >= 1.0 - tol).float().mean().item()

    def l1_penalty(self, ids):
        """
        Mean |mask| over 'dense' entities only (N >= n_bar) -- the L1
        sparsity term the pre-refactor code applied per model, generalised
        to any (n_slots, dim) shape.

        Entities below n_bar are exempted deliberately: driving their mask
        toward zero would fight the exploration bonus that is supposed to
        keep them open, which is the opposite of what this regulariser is
        for.

        Returns a plain 0.0 (not a zero tensor) when nothing in the batch
        qualifies, so callers can do `loss + lambda_l1 * m.l1_penalty(ids)`
        unconditionally without an `if dense.any()` guard at every call
        site -- see train.py's `_mask_l1_penalty`, which calls this on
        whichever of `user_mask` / `item_mask` a model has, generically
        across all ten models rather than branching on model name.
        """
        m = self.forward(ids)
        dense = self.counts[ids].float() >= self.n_bar
        if not dense.any():
            return 0.0
        return m[dense].abs().mean()