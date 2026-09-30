"""
Model package exports.

The pre-refactor version still imported five modules deleted in
Checkpoint 1 -- `deepcf_static_mask_attn`, `rpucb_attn`, `pinterest_base`,
`pinterest_base_rpucb` -- so `import src.models` raised ImportError and
nothing in the repo ran. This exports the ten matrix models and nothing
else.

`registry.py` will supersede the explicit key mapping below; MATRIX_MODELS
is here so the models can be exercised before that lands.
"""

from .attention import SelfAttentionInteraction
from .base import BaseCF, build_mlp
from .dcm import DCM
from .deepcf import DeepCF
from .history import build_history_buffers, embed_history_bag
from .deepcf_rpucb import DeepCFRPUCB
from .deepcf_rpucb_attn import DeepCFRPUCBAttn
from .mind import LabelAwareAttention, MINDRouting
from .mind_model import MIND, MINDRPUCB, MINDRPUCBMulti
from .pinterest_dcm import PinterestDCM
from .pinterest_dcm_rpucb import PinterestDCMRPUCB
from .pinterest_tower import LiteDHEN, LiteMaskNet, MLPSummarization, PinterestTower
from .routing_common import NEG_SCORE, squash
from .rpucb_mask import RPUCBMask

# key -> (class, extra constructor kwargs). Everything not listed here
# comes from the resolved config.
MATRIX_MODELS = {
    "deepcf":            (DeepCF, {}),
    "deepcf_rpucb":      (DeepCFRPUCB, {}),
    "deepcf_rpucb_attn": (DeepCFRPUCBAttn, {}),

    "mind":              (MIND, {"adaptive_k": True}),
    "mind_rpucb_multi":  (MINDRPUCBMulti, {"adaptive_k": True}),
    "mind_rpucb":        (MINDRPUCB, {}),

    "dcm":               (PinterestDCM, {"K": 7}),
    "dcm_rpucb_multi":   (PinterestDCMRPUCB,
                          {"K": 7, "head_mode": "per_slot",
                           "mask_granularity": "per_slot"}),
    "dcm_rpucb_kd":      (PinterestDCMRPUCB,
                          {"K": 7, "head_mode": "concat"}),
    "dcm_rpucb_d":       (PinterestDCMRPUCB,
                          {"K": 1, "head_mode": "per_slot",
                           "mask_granularity": "shared"}),

    # Symmetric shared-mask DCM run (§2): available for a like-for-like
    # comparison against mind_rpucb_multi, deliberately not part of the
    # main matrix.
    "dcm_rpucb_shared":  (PinterestDCMRPUCB,
                          {"K": 7, "head_mode": "per_slot",
                           "mask_granularity": "shared"}),
}

__all__ = [
    "BaseCF", "build_mlp", "SelfAttentionInteraction",
    "PinterestTower", "LiteDHEN", "LiteMaskNet", "MLPSummarization",
    "RPUCBMask", "squash", "NEG_SCORE",
    "build_history_buffers", "embed_history_bag",
    "DCM", "MINDRouting", "LabelAwareAttention",
    "DeepCF", "DeepCFRPUCB", "DeepCFRPUCBAttn",
    "MIND", "MINDRPUCBMulti", "MINDRPUCB",
    "PinterestDCM", "PinterestDCMRPUCB",
    "MATRIX_MODELS",
]