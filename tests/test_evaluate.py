"""
Evaluation.

Every reported number in the project comes out of this module, and two
of its jobs are easy to get subtly wrong.

Exclusions: ranking a user's held-out item against items they already
trained on is the easiest way to manufacture good numbers, and at test
time the val item has to go too or it sits in the candidate set as a
spurious competitor.

The warm subset: cold positives are unrankable by construction rather
than by model quality, and their share rises with sparsity -- the same
axis as the central hypothesis -- so the headline number and the number
that isolates what models can actually rank both have to be on the
record.
"""

import pytest
import torch

from src.evaluate import _apply_exclusions, _select_users, _warm_mask, evaluate_popularity, evaluate_split
from src.models.registry import build_model

DEVICE = torch.device("cpu")


def test_exclusions_remove_train_items_and_the_other_split(dataset):
    users = [u for u in range(dataset.num_users) if u in dataset.test_item][:4]
    batch = [(u, dataset.test_item[u]) for u in users]
    scores = torch.zeros(len(batch), dataset.num_items)

    _apply_exclusions(scores, batch, dataset, "test", DEVICE)

    for row, (u, positive) in enumerate(batch):
        for item in dataset.user_train_items.get(u, ()):
            assert scores[row, item] == float("-inf"), "a train item stayed rankable"
        if u in dataset.val_item:
            assert scores[row, dataset.val_item[u]] == float("-inf")
        assert torch.isfinite(scores[row, positive]), "the positive was excluded"


def test_val_ranking_hides_the_test_item(dataset):
    users = [u for u in dataset.val_item][:4]
    batch = [(u, dataset.val_item[u]) for u in users]
    scores = torch.zeros(len(batch), dataset.num_items)

    _apply_exclusions(scores, batch, dataset, "val", DEVICE)

    for row, (u, positive) in enumerate(batch):
        assert scores[row, dataset.test_item[u]] == float("-inf")
        assert torch.isfinite(scores[row, positive])


def test_warm_mask_marks_cold_positives_and_trainless_users(dataset):
    from conftest import COLD_ITEMS, TRAINLESS_USER

    users = torch.tensor([0, TRAINLESS_USER])
    positives = torch.tensor([dataset.test_item[0], COLD_ITEMS[-1]])
    warm = _warm_mask(dataset, users, positives, DEVICE)

    assert warm.dtype == torch.bool and warm.numel() == 2
    assert not bool(warm[1]), "a train-less user with a cold positive counted as warm"


def test_user_subsample_is_seeded_and_bounded(dataset):
    a = _select_users(dataset, "test", max_users=5, seed=1)
    b = _select_users(dataset, "test", max_users=5, seed=1)
    c = _select_users(dataset, "test", max_users=5, seed=2)

    assert len(a) == 5 and a == b
    assert a != c, "the val subsample does not depend on the seed"
    assert _select_users(dataset, "test", max_users=None, seed=1) == dataset.get_test_data()


def test_subsample_larger_than_the_split_returns_everything(dataset):
    """
    lastfm and AMusic have fewer val users than the 1,000-user default,
    so this is the normal case there rather than an edge case.
    """
    everyone = _select_users(dataset, "val", max_users=10_000, seed=0)
    assert everyone == dataset.get_val_data()


@pytest.mark.parametrize("model_key", ["deepcf", "mind_rpucb", "dcm_rpucb_multi"])
def test_evaluate_split_returns_the_five_reported_metrics(model_key, dataset, tiny_config):
    model = build_model(model_key, dataset, tiny_config()).eval()
    metrics = evaluate_split(model, dataset, "test", DEVICE, user_batch_size=3)

    for key in ("HR@10", "HR@100", "NDCG@10", "Coverage@10", "ILD@10"):
        assert key in metrics, f"{model_key} evaluation is missing {key}"
        assert metrics[key] == metrics[key], f"{key} is NaN"
    assert metrics["num_users_evaluated"] == len(dataset.get_test_data())
    assert metrics["tie_policy"] == "mid"


def test_user_batch_size_does_not_change_the_result(dataset, tiny_config):
    """Another memory knob that must not be able to move a number."""
    model = build_model("deepcf", dataset, tiny_config()).eval()
    one = evaluate_split(model, dataset, "test", DEVICE, user_batch_size=1)
    many = evaluate_split(model, dataset, "test", DEVICE, user_batch_size=64)

    for key in ("HR@10", "HR@100", "NDCG@10", "Coverage@10"):
        assert one[key] == pytest.approx(many[key], abs=1e-6), key


def test_tie_policy_reaches_the_metrics(dataset, tiny_config):
    """
    The optimistic-vs-mid gap is a reported sensitivity check, so the
    flag has to actually travel from the config to the accumulator.
    """
    model = build_model("mind", dataset, tiny_config()).eval()
    mid = evaluate_split(model, dataset, "test", DEVICE, tie_policy="mid")
    optimistic = evaluate_split(model, dataset, "test", DEVICE, tie_policy="optimistic")

    assert optimistic["HR@10"] >= mid["HR@10"]
    assert optimistic["tie_policy"] == "optimistic"


def test_popularity_reference_is_model_free_and_reproducible(dataset):
    """
    The floor every result has to survive: a model that cannot beat
    "recommend the most popular items" is not personalising. First
    light had DeepCF surfacing 780 distinct items on AMusic, which is
    what popularity looks like.
    """
    a = evaluate_popularity(dataset, "test", DEVICE, user_batch_size=4)
    b = evaluate_popularity(dataset, "test", DEVICE, user_batch_size=7)

    assert a["HR@10"] == pytest.approx(b["HR@10"])
    assert 0.0 <= a["Coverage@10"] <= 1.0


def test_evaluate_rejects_an_unknown_split(dataset, tiny_config):
    model = build_model("deepcf", dataset, tiny_config()).eval()
    with pytest.raises(ValueError, match="split must be"):
        evaluate_split(model, dataset, "train", DEVICE)


def test_resident_matrices_give_the_same_answer(dataset, tiny_config):
    """
    train.py passes the interaction matrices in so val evaluation does
    not re-transfer them every epoch. That optimisation must be
    invisible in the result.
    """
    model = build_model("deepcf_rpucb", dataset, tiny_config()).eval()
    plain = evaluate_split(model, dataset, "val", DEVICE)
    resident = evaluate_split(
        model, dataset, "val", DEVICE,
        interaction_rows_gpu=dataset.interaction_rows.to(DEVICE),
        interaction_cols_gpu=dataset.interaction_cols.to(DEVICE),
    )
    assert plain["HR@10"] == pytest.approx(resident["HR@10"])
    assert plain["NDCG@10"] == pytest.approx(resident["NDCG@10"])
