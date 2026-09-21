"""
Shared model base.

Two changes from the pre-refactor version.

1. A uniform contract. Previously `DeepCFRPUCB.forward` returned
   `(scores, masks)` while `PinterestDCM.forward` returned bare scores,
   and `evaluate.py` reconciled them with an `isinstance(scores, tuple)`
   check. With ten models and a registry that instantiates them by key,
   the caller cannot keep knowing which model it built. The contract now:

       forward()         -> scores [B]                (always a tensor)
       score()           -> scores [B]
       score_with_mask() -> (scores [B], masks | None)
       score_multi()     -> (scores [B, K], reps | None)

   Single-embedding models inherit `score_multi` as a K=1 wrapper, which
   is also the mechanism behind the K=1 collapse rows (`mind_rpucb`,
   `dcm_rpucb_d`): the interest machinery goes inert and the same code
   path serves both.

   Slots that are not real for a given user -- a history shorter than K,
   or an adaptive K_u below K_max -- come back from `score_multi` already
   set to `NEG_SCORE`, so a `max` over K or an `argmax` for condition
   association ignores them without every caller re-deriving validity.

2. A separate contract for full-catalog scoring:

       encode_items(item_cols, item_ids)  -> item-side representation
       encode_users(user_row, user_ids)   -> user-side representation
       score_encoded(user_enc, item_enc)  -> scores [B, N]

   The pairwise path cannot be reused for full-catalog evaluation. It
   takes raw interaction profiles -- `num_items + num_users` floats per
   (user, item) pair, about 37,000 for AToy -- so scoring one user batch
   against a whole catalog means materialising tens of gigabytes of
   duplicated raw features. Encoding each side once and combining at
   d=64 cuts the per-pair intermediate by 76-290x depending on dataset,
   and lets `evaluate.py` compute item encodings once per evaluation
   rather than once per user batch.

   The representations are deliberately opaque: DeepCF needs two vectors
   per side (the CFNet-rl and CFNet-ml branches), the DCM family needs K
   conditions plus a validity mask, MIND needs K interests plus a
   validity mask. Each family documents its own shapes beside its own
   encoder.

   Not implemented here on purpose. A generic fallback would have to tile
   raw pairs, which is exactly the pattern this contract exists to
   eliminate -- it would turn a missing override into a silent
   out-of-memory crash at evaluation time rather than a clear error at
   the first call.

3. `init_weights` no longer walks into attention and transformer blocks.
   The old version re-initialised every `nn.Linear` it could find to
   normal(0, 0.01), which inside `LiteDHEN` hit
   `TransformerEncoderLayer.linear1/linear2` and
   `MultiheadAttention.out_proj` -- but not `in_proj_weight`, which is a
   raw Parameter and kept its Xavier default. The result was attention
   with a Xavier input projection and a std-0.01 output projection, and a
   transformer whose intended initialisation had been discarded. Modules
   listed in `_PROTECTED_TYPES`, or flagged with `_skip_default_init`,
   are now left alone along with everything beneath them.
"""

import torch
import torch.nn as nn

from .routing_common import NEG_SCORE

# Rows of the [N, num_users] item matrix pushed through the item encoder at
# once. Bounds a transient dense in num_users; what it accumulates is only
# [N, d].
DEFAULT_ITEM_ENCODE_CHUNK = 8192

# Ceiling on (user, item) pairs held in flight inside `score_encoded`. Those
# intermediates are d-wide rather than num_items-wide, so this is a few
# hundred MB rather than the tens of GB the raw-feature path needed. The
# multi-interest families divide it by K, since they hold B*K*C.
DEFAULT_PAIR_CHUNK = 262_144

# Modules whose own initialisation is deliberate and should survive.
_PROTECTED_TYPES = (
    nn.MultiheadAttention,
    nn.TransformerEncoder,
    nn.TransformerEncoderLayer,
    nn.LayerNorm,
)


def build_mlp(layer_sizes, dropout=0.0, activation=nn.ReLU):
    """
    Sequential MLP. Hidden layers get activation (+ dropout if > 0); the
    final layer is linear only.
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            if activation is not None:
                layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def chunk_bounds(total, step):
    """Yield (start, end) pairs covering range(total) in steps of `step`."""
    step = max(1, step)
    for start in range(0, total, step):
        yield start, min(start + step, total)


class BaseCF(nn.Module):

    # ---- pairwise scoring --------------------------------------------
    def forward(self, user_row, item_col, user_ids=None, item_ids=None):
        return self.score(user_row, item_col, user_ids, item_ids)

    def score(self, user_row, item_col, user_ids=None, item_ids=None):
        raise NotImplementedError

    def score_with_mask(self, user_row, item_col, user_ids=None, item_ids=None):
        """Unmasked models return None for the mask half."""
        return self.score(user_row, item_col, user_ids, item_ids), None

    def score_multi(self, user_row, item_col, user_ids=None, item_ids=None):
        """
        Per-slot scores for K-embedding models: ([B, K], reps | None).
        Default implementation treats the model as K=1.
        """
        scores = self.score(user_row, item_col, user_ids, item_ids)
        return scores.unsqueeze(1), None

    # ---- full-catalog scoring ----------------------------------------
    def encode_items(self, item_cols, item_ids=None, chunk_size=DEFAULT_ITEM_ENCODE_CHUNK):
        """
        Item-side representation for [N, num_users] `item_cols`.

        Called once per evaluation with the entire catalog. Implementations
        chunk internally over rows: the input is dense in num_users but the
        output is compact (N x d), so the transient is bounded by
        chunk_size while the retained result stays small.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement encode_items; "
            f"full-catalog evaluation requires it. See models/base.py."
        )

    def encode_users(self, user_row, user_ids=None):
        """User-side representation for one batch of users."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement encode_users; "
            f"full-catalog evaluation requires it. See models/base.py."
        )

    def score_encoded(self, user_enc, item_enc, pair_chunk=DEFAULT_PAIR_CHUNK):
        """[B, N] scores from the two encoded sides."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement score_encoded; "
            f"full-catalog evaluation requires it. See models/base.py."
        )

    @torch.no_grad()
    def score_all_items(
        self, user_row, user_ids, item_cols, item_ids,
        item_chunk_size=DEFAULT_ITEM_ENCODE_CHUNK,
        pair_chunk=DEFAULT_PAIR_CHUNK,
    ):
        """
        Convenience wrapper. `evaluate.py` deliberately does not use this --
        it caches the item encoding across user batches instead, which is
        the whole point. Useful for one-off scoring and for tests.
        """
        item_enc = self.encode_items(item_cols, item_ids, chunk_size=item_chunk_size)
        user_enc = self.encode_users(user_row, user_ids)
        return self.score_encoded(user_enc, item_enc, pair_chunk=pair_chunk)

    # ------------------------------------------------------------------
    @staticmethod
    def mask_invalid_slots(scores, slot_valid):
        """
        Set scores for non-existent interest slots to NEG_SCORE.

        Used by every multi-interest model so that `max` over K and the
        training-time argmax both skip padding slots automatically. A
        finite sentinel rather than -inf, for the NaN reasons documented
        in routing_common.
        """
        if slot_valid is None:
            return scores
        return scores.masked_fill(~slot_valid, NEG_SCORE)

    def init_weights(self):
        skip_prefixes = set()
        for name, module in self.named_modules():
            if not name:
                continue
            if isinstance(module, _PROTECTED_TYPES) or getattr(
                module, "_skip_default_init", False
            ):
                skip_prefixes.add(name)

        def protected(name):
            return any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in skip_prefixes
            )

        for name, module in self.named_modules():
            if name and protected(name):
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)