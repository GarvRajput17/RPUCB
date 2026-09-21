"""
Routing primitives shared by MIND and DCM.

`squash` originates in Sabour et al.'s capsule networks, is used unchanged
by MIND (Li et al., Eq. 3), and is inherited by Pinterest's DCM (Eq. 3).
It lived in `dcm.py` before this refactor, which meant `mind.py` would
either import from its own successor or keep a second copy free to drift.
Neither is acceptable when `mind` vs `dcm` is supposed to isolate the
*routing rule* -- a divergence in the shared non-linearity would silently
contaminate that comparison.

NEG_SCORE is the finite stand-in this codebase uses instead of -inf when
masking out invalid slots before a softmax or argmax. Using a real -inf
produces NaN whenever an entire row is masked (a user whose history is
shorter than K, or whose interest count K_u < K_max), and `torch.where`
propagates NaN gradients even through the branch it does not select. A
large finite negative avoids the whole class of problem: cosine
similarities live in [-1, 1], so -1e4 is unreachable by any real score.
"""

import torch

NEG_SCORE = -1e4


def squash(v, dim=-1, eps=1e-8):
    """
        squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)

    Bounds the output into the unit ball while letting the vector's norm
    encode how strongly the corresponding interest is supported.
    """
    norm_sq = (v ** 2).sum(dim=dim, keepdim=True)
    scale = norm_sq / (1.0 + norm_sq)
    return scale * v / torch.sqrt(norm_sq + eps)


def slot_valid_from_counts(counts, n_slots):
    """
    Boolean [B, n_slots] marking which of `n_slots` interest slots are real
    for each user, given a per-user count of usable slots.

    Both routing modules need this and both would otherwise open-code the
    same broadcast comparison.
    """
    idx = torch.arange(n_slots, device=counts.device).unsqueeze(0)
    return idx < counts.unsqueeze(1)