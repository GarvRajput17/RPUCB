"""
Loss functions, one per family (§3), plus the BPR robustness fallback.

    family                model keys                                  loss
    bce                   deepcf, deepcf_rpucb, deepcf_rpucb_attn      BCE w/ negative sampling
    sampled_softmax       mind, mind_rpucb_multi, mind_rpucb           sampled softmax
    sampled_softmax_logq  dcm, dcm_rpucb_*                             sampled softmax + logQ

Native-per-family, uniform-within-family, per §3: three independent
pairwise wrapper tests, each internally matched on backbone AND loss,
rather than one loss imposed on architectures whose own papers use
something else (DeepCF's own paper uses BCE, not BPR). `bpr_loss` is kept
and reachable via a config flag (`loss_override: bpr`) as the robustness
check §3 asks for. BPR composes with any family's score derivation
because it is a loss-function choice, not an architecture choice --
`train.py`'s `compute_loss` derives (pos_score, neg_scores) the same way
regardless of which loss ultimately consumes them.

`uniform_log_q` is the fidelity-not-forgotten implementation of
Pinterest's logQ correction (§9). Under exactly-uniform sampling over the
full catalog, log Q(i) = log(1/num_items) for every item -- a single
global scalar, identical in every row and column of the logit matrix.
Subtracting a true global constant from every softmax logit leaves the
softmax, and therefore the loss, unchanged: the correction is inert for
this sampler by identity, not by approximation.

Inert in exact arithmetic, that is -- not bit-for-bit in floating
point. The constant is log(1/3706) ~ -8.22 on ml-1m, so every logit
shifts by that much before the logsumexp and the rounding differs: about
1e-7 relative in float32, one ulp in float64. Nothing depends on it, the
difference being orders of magnitude below any training signal, but a
codebase whose point is that reported numbers can be traced to what ran
should not claim an exactness it does not have. It is still
wired through `sampled_softmax_loss`'s `pos_log_q`/`neg_log_q` arguments
for the DCM family, so the correction lives in the code path Pinterest
specifies and is ready to become load-bearing the moment negative
sampling stops being uniform (e.g. popularity-based sampling), rather
than depending on someone remembering to add it back in later.
"""

import torch
import torch.nn.functional as F

LOSS_FAMILY = {
    "deepcf": "bce",
    "deepcf_rpucb": "bce",
    "deepcf_rpucb_attn": "bce",

    "mind": "sampled_softmax",
    "mind_rpucb_multi": "sampled_softmax",
    "mind_rpucb": "sampled_softmax",

    "dcm": "sampled_softmax_logq",
    "dcm_rpucb_multi": "sampled_softmax_logq",
    "dcm_rpucb_kd": "sampled_softmax_logq",
    "dcm_rpucb_d": "sampled_softmax_logq",
    "dcm_rpucb_shared": "sampled_softmax_logq",
}


def bpr_loss(pos_scores, neg_scores):
    """
    pos_scores: [B]
    neg_scores: [B, num_negatives] (or [B], treated as one negative)
    """
    if neg_scores.dim() == 1:
        neg_scores = neg_scores.unsqueeze(1)
    pos_scores = pos_scores.unsqueeze(1).expand_as(neg_scores)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()


def bce_loss(pos_scores, neg_scores):
    """
    Pointwise BCE with negative sampling (DeepCF / NCF-style): the
    positive is one label-1 example and each sampled negative an
    independent label-0 example -- not a pairwise comparison the way BPR
    is.

    pos_scores: [B] raw logits
    neg_scores: [B, num_negatives] raw logits
    """
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    return pos_loss + neg_loss


def mask_l1_loss(mask_values):
    """Plain mean |mask|, kept for anything that wants an unconditional
    L1 term without the dense-entity gating RPUCBMask.l1_penalty applies."""
    return mask_values.abs().mean()


def uniform_log_q(num_items, shape, device, dtype=torch.float32):
    """log(1/num_items), broadcast to `shape`. See module docstring."""
    value = -torch.log(torch.tensor(float(num_items), device=device, dtype=dtype))
    return value.expand(shape)


def sampled_softmax_loss(pos_scores, neg_scores, pos_log_q=None, neg_log_q=None):
    """
    Cross-entropy over the candidate set {positive, negatives}, target
    class 0. Used both for the plain MIND family (log_q args left None,
    per §3's table -- MIND gets no logQ term) and the DCM family (log_q
    supplied, and exactly inert under uniform sampling -- see above).

    pos_scores: [B]
    neg_scores: [B, num_negatives]
    """
    if pos_log_q is not None:
        pos_scores = pos_scores - pos_log_q
    if neg_log_q is not None:
        neg_scores = neg_scores - neg_log_q

    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)