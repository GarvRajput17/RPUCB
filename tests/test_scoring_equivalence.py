"""
The most important test in the suite.

Every model implements scoring twice. `score()` takes one (user, item)
pair at a time from raw interaction profiles, and is what training uses.
`encode_users` / `encode_items` / `score_encoded` encode each side once
and combine at d=64, and is what every reported number comes from. The
second path exists because the first cannot be tiled across a catalog
without materialising tens of gigabytes -- see models/base.py.

Seven of the eleven encoders were derived by hand during the refactor and
never cross-checked against the pairwise path they were derived from. If
any one of them is wrong, nothing crashes: training optimises one
function and evaluation reports another, and the result is a plausible
number for a model that was never trained to produce it. No amount of
looking at loss curves finds that.

So: for each of the eleven keys, score every (user, item) pair both ways
and require agreement. The tolerance is loose enough for the chunked
matmuls (the two paths accumulate in a different order) and far tighter
than any difference a real bug would produce.
"""

import pytest
import torch

from conftest import all_model_keys
from src.models.registry import build_model


def _both_paths(model, dataset, user_ids, item_ids):
    """(fast [U, I], slow [U, I]) for the cross product of the two id lists."""
    users = torch.tensor(user_ids, dtype=torch.long)
    items = torch.tensor(item_ids, dtype=torch.long)

    with torch.no_grad():
        fast = model.score_all_items(
            dataset.interaction_rows[users], users,
            dataset.interaction_cols[items], items,
        )

        # The pairwise path, as train.py calls it: one raw profile per side
        # per pair.
        pair_u = users.repeat_interleave(len(items))
        pair_i = items.repeat(len(users))
        slow = model.score(
            dataset.interaction_rows[pair_u], dataset.interaction_cols[pair_i],
            pair_u, pair_i,
        ).view(len(users), len(items))

    return fast, slow


@pytest.mark.parametrize("model_key", all_model_keys())
def test_fast_and_slow_paths_agree(model_key, dataset, tiny_config):
    model = build_model(model_key, dataset, tiny_config()).eval()

    # Users spanning the interaction-count range, including the train-less
    # one, and items spanning warm and cold.
    fast, slow = _both_paths(model, dataset, [0, 4, 9, 10, 11], list(range(25)))

    torch.testing.assert_close(
        fast, slow, atol=2e-5, rtol=1e-4,
        msg=lambda s: (
            f"{model_key}: full-catalog scoring disagrees with the pairwise "
            f"path.\nTraining optimises the pairwise path; every reported "
            f"metric comes from the other one.\n{s}"
        ),
    )


@pytest.mark.parametrize("model_key", all_model_keys())
def test_score_encoded_shape_and_finiteness(model_key, dataset, tiny_config):
    model = build_model(model_key, dataset, tiny_config()).eval()
    users = torch.tensor([0, 3, 11])
    items = torch.arange(dataset.num_items)

    with torch.no_grad():
        scores = model.score_all_items(
            dataset.interaction_rows[users], users,
            dataset.interaction_cols[items], items,
        )

    assert scores.shape == (3, dataset.num_items)
    assert torch.isfinite(scores).all(), f"{model_key} produced non-finite scores"


@pytest.mark.parametrize("model_key", all_model_keys())
def test_pair_chunking_does_not_change_scores(model_key, dataset, tiny_config):
    """
    `pair_chunk` is a memory knob read from base.yaml, so the dry run can
    lower it on AToy. It must not be able to change a result -- a knob
    that quietly alters numbers is worse than an out-of-memory error.
    """
    model = build_model(model_key, dataset, tiny_config()).eval()
    users = torch.arange(5)
    items = torch.arange(dataset.num_items)

    with torch.no_grad():
        item_enc = model.encode_items(dataset.interaction_cols[items], items)
        user_enc = model.encode_users(dataset.interaction_rows[users], users)
        whole = model.score_encoded(user_enc, item_enc, pair_chunk=10**6)
        chunked = model.score_encoded(user_enc, item_enc, pair_chunk=7)

    # Not bit-exact, and should not be asserted as such: a different batch
    # shape makes the matmul kernel accumulate in a different order. The
    # tolerance is the same one the pairwise-equivalence test uses, which
    # is orders of magnitude below any difference a real bug produces.
    torch.testing.assert_close(whole, chunked, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("model_key", all_model_keys())
def test_item_encode_chunking_does_not_change_scores(model_key, dataset, tiny_config):
    """Same argument for the item-side chunk size."""
    model = build_model(model_key, dataset, tiny_config()).eval()
    items = torch.arange(dataset.num_items)

    with torch.no_grad():
        whole = model.encode_items(dataset.interaction_cols[items], items, chunk_size=10**6)
        chunked = model.encode_items(dataset.interaction_cols[items], items, chunk_size=3)

    def flatten(enc):
        if torch.is_tensor(enc):
            return [enc]
        if isinstance(enc, dict):
            return [v for v in enc.values() if torch.is_tensor(v)]
        return [v for v in enc if torch.is_tensor(v)]

    parts_whole, parts_chunked = flatten(whole), flatten(chunked)
    assert len(parts_whole) == len(parts_chunked)
    for a, b in zip(parts_whole, parts_chunked):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("model_key", all_model_keys())
def test_score_multi_max_equals_score(model_key, dataset, tiny_config):
    """
    `score()` on a multi-interest model is the max over its K slots, and
    invalid slots come back at NEG_SCORE so the max ignores them. Single-
    embedding models inherit the K=1 wrapper, which is the same mechanism
    the K=1 collapse rows (`mind_rpucb`, `dcm_rpucb_d`) rely on -- so this
    has to hold for all eleven, not just the multi-interest ones.
    """
    model = build_model(model_key, dataset, tiny_config()).eval()
    users = torch.tensor([0, 2, 5, 10, 11])
    items = torch.tensor([1, 7, 13, 22, 24])

    with torch.no_grad():
        direct = model.score(
            dataset.interaction_rows[users], dataset.interaction_cols[items], users, items
        )
        per_slot, _ = model.score_multi(
            dataset.interaction_rows[users], dataset.interaction_cols[items], users, items
        )

    assert per_slot.dim() == 2 and per_slot.size(0) == users.numel()
    torch.testing.assert_close(per_slot.max(dim=1).values, direct, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("model_key", all_model_keys())
def test_scoring_is_deterministic_within_an_instance(model_key, dataset, tiny_config):
    """
    Two identical calls must give identical scores. Anything sampled at
    forward time -- MIND's routing logits are the candidate -- would make
    the equivalence test above flaky rather than false, which is a much
    worse failure to have.
    """
    model = build_model(model_key, dataset, tiny_config()).eval()
    users = torch.arange(4)
    items = torch.arange(dataset.num_items)

    with torch.no_grad():
        first = model.score_all_items(
            dataset.interaction_rows[users], users, dataset.interaction_cols[items], items)
        second = model.score_all_items(
            dataset.interaction_rows[users], users, dataset.interaction_cols[items], items)

    torch.testing.assert_close(first, second, atol=0, rtol=0)
