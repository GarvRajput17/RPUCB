"""
Dataset splitting.

Early stopping needs a validation signal, and there was none: the
pre-refactor pipeline loaded train and test only, so any stopping rule
would have read the test set. The carve added here is where a leak
would now hide, and a leak of this kind does not crash or even look
wrong -- it just makes every reported number optimistic.

Three properties carry the weight:

  the carved item leaves train, including the dense interaction
  matrices the towers actually read;

  the carve is a function of the seed, so a run is reproducible and two
  seeds genuinely differ;

  val-time ranking never sees the test item as a candidate, and
  test-time ranking never sees the val item.
"""

import pytest
import torch

from conftest import COLD_ITEMS, NUM_ITEMS, NUM_USERS, SINGLE_ITEM_USER, TRAINLESS_USER
from src.data.dataset import _carve_val_split


def test_carved_item_is_removed_from_train(dataset):
    for u, val_item in dataset.val_item.items():
        assert val_item not in dataset.user_train_items[u], (
            f"user {u}'s val item {val_item} is still in their train set"
        )


def test_carved_item_is_absent_from_the_dense_matrices(dataset):
    """
    The towers read `interaction_rows` and `interaction_cols`, not
    `train_pairs`. A carve that updated one and not the other would leak
    the val item into the model's input while looking correct
    everywhere a test is likely to check.
    """
    for u, val_item in dataset.val_item.items():
        assert dataset.interaction_rows[u, val_item].item() == 0.0
        assert dataset.interaction_cols[val_item, u].item() == 0.0


def test_test_items_never_appear_in_train(dataset):
    for u, test_item in dataset.test_item.items():
        assert test_item not in dataset.user_train_items.get(u, set())
        assert dataset.interaction_rows[u, test_item].item() == 0.0


def test_same_seed_carves_the_same_split(fresh_dataset):
    a, b = fresh_dataset(seed=42), fresh_dataset(seed=42)
    assert a.val_item == b.val_item


def test_different_seeds_carve_different_splits(fresh_dataset):
    a, b = fresh_dataset(seed=42), fresh_dataset(seed=43)
    assert a.val_item != b.val_item, "the carve does not depend on the seed"


def test_user_with_too_few_items_is_skipped_not_emptied():
    """
    A user with one train item cannot donate it without training on an
    empty set. They keep it and are absent from val -- the count is
    returned so that is visible rather than silently absorbed.
    """
    items = {0: {5}, 1: {1, 2, 3}}
    val_item, skipped = _carve_val_split(items, seed=0, min_train_remaining=1)
    assert skipped == 1
    assert 0 not in val_item and items[0] == {5}
    assert 1 in val_item and len(items[1]) == 2


def test_min_train_remaining_is_respected():
    items = {0: {1, 2}, 1: {1, 2, 3, 4}}
    val_item, skipped = _carve_val_split(items, seed=0, min_train_remaining=2)
    assert 0 not in val_item and skipped == 1
    assert 1 in val_item and len(items[1]) == 3


def test_carve_is_deterministic_for_a_fixed_seed():
    def carve(seed):
        items = {u: set(range(10 * u, 10 * u + 8)) for u in range(5)}
        return _carve_val_split(items, seed=seed)[0]
    assert carve(1) == carve(1)
    assert carve(1) != carve(2)


def test_excluded_items_keep_the_splits_apart(dataset):
    for u in range(NUM_USERS):
        val_excl = dataset.excluded_items(u, "val")
        test_excl = dataset.excluded_items(u, "test")

        assert dataset.user_train_items.get(u, set()) <= val_excl
        if u in dataset.test_item:
            assert dataset.test_item[u] in val_excl, (
                "the test item is a candidate during val ranking"
            )
            assert dataset.test_item[u] not in test_excl, (
                "the test item was excluded from its own ranking"
            )
        if u in dataset.val_item:
            assert dataset.val_item[u] in test_excl
            assert dataset.val_item[u] not in val_excl


def test_excluded_items_rejects_an_unknown_split(dataset):
    with pytest.raises(ValueError, match="split must be"):
        dataset.excluded_items(0, "train")


def test_interaction_counts_come_from_the_reduced_train_set(dataset):
    """
    n_bar in the RP-UCB mask is derived from these. Counting val or test
    interactions would leak the held-out data into the exploration term.
    """
    for u in range(NUM_USERS):
        assert dataset.user_interaction_counts[u].item() == len(
            dataset.user_train_items.get(u, ())
        )
    assert dataset.item_interaction_counts.sum().item() == len(dataset.train_pairs)


def test_rows_and_cols_are_normalised_views_of_the_same_matrix(dataset):
    """
    `interaction_rows` is row-normalised and `interaction_cols`
    column-normalised, but their support must be identical -- they are
    two readings of one train set, and a mismatch would mean the user
    and item towers disagree about what happened.
    """
    rows_support = (dataset.interaction_rows > 0)
    cols_support = (dataset.interaction_cols > 0).t()
    assert torch.equal(rows_support, cols_support)


def test_cold_structure_is_reported(dataset):
    report = dataset.cold_report
    assert report["trainless_users"] >= 1, "the synthetic set has a train-less user"
    assert report["cold_catalog_items"] >= len(COLD_ITEMS)
    assert report["test_users"] == NUM_USERS
    assert report["val_users"] == len(dataset.val_item)


def test_trainless_user_has_an_all_zero_feature_row(dataset):
    """The case that makes every MIND interest vector exactly zero."""
    assert dataset.interaction_rows[TRAINLESS_USER].sum().item() == 0.0
    assert dataset.user_interaction_counts[TRAINLESS_USER].item() == 0


def test_single_item_user_kept_their_item(dataset):
    assert SINGLE_ITEM_USER not in dataset.val_item
    assert len(dataset.user_train_items[SINGLE_ITEM_USER]) == 1


def test_catalog_dimensions_cover_test_only_items(dataset):
    """
    num_items is taken over train and test together. Sizing it from
    train alone would put every cold item out of range of the catalog
    the model scores.
    """
    assert dataset.num_items == NUM_ITEMS
    assert dataset.num_users == NUM_USERS
    for item in COLD_ITEMS:
        assert item < dataset.num_items
        assert dataset.item_interaction_counts[item].item() == 0
