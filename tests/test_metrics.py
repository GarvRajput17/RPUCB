"""
Full-catalog metrics, with tie handling as the main event.

First light made the stakes concrete. An item with no train interactions
has an all-zero interaction column, so every model maps it to one shared
embedding and every such item takes the same score. A user with no train
history is worse: for the MIND family every interest vector comes out
exactly zero, so the whole catalog scores 0.0. Under the optimistic
convention both are rank 1 -- 43 free hits on AMusic, 65% of MIND's
measured HR@10.

And cold fractions rise with sparsity (0% on lastfm to 30% on AToy),
which is the same axis as the central hypothesis. An artifact that grows
with sparsity is indistinguishable from the effect being tested, so the
`mid` default is load-bearing rather than a convention.
"""

import math

import pytest
import torch

from src.metrics import (
    FullCatalogMetrics,
    count_ties,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    ranks_of_positives,
)


def test_rank_with_no_ties():
    scores = torch.tensor([[0.9, 0.5, 0.1, 0.3]])
    positive = torch.tensor([1])                     # second best
    assert ranks_of_positives(scores, positive).item() == pytest.approx(2.0)


@pytest.mark.parametrize("policy,expected", [
    ("optimistic", 1.0),     # 0 strictly greater
    ("mid", 2.5),            # 1 + 0 + 3/2
    ("pessimistic", 4.0),    # 1 + 0 + 3
])
def test_tie_policies_on_a_four_way_tie(policy, expected):
    """The AMusic cold-item case in miniature: four items, all equal."""
    scores = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    positive = torch.tensor([0])
    got = ranks_of_positives(scores, positive, tie_policy=policy).item()
    assert got == pytest.approx(expected)


def test_trainless_user_is_not_a_free_hit_under_mid():
    """
    MIND on a train-less user scores every item exactly 0.0. Under
    optimistic that is a guaranteed HR@10 hit for a model that computed
    nothing; under mid it is a miss, as it must be.
    """
    num_items = 1000
    scores = torch.zeros(1, num_items)
    positive = torch.tensor([7])

    optimistic = ranks_of_positives(scores, positive, tie_policy="optimistic")
    mid = ranks_of_positives(scores, positive, tie_policy="mid")

    assert hit_rate_at_k(optimistic, 10).item() == 1.0
    assert hit_rate_at_k(mid, 10).item() == 0.0
    assert mid.item() == pytest.approx(1.0 + (num_items - 1) / 2)


def test_ties_are_detected_by_tolerance_not_equality():
    """
    Identical inputs can differ by an ulp when they land in different
    chunks of a batched matmul. Exact `==` would miss those and fall
    back to optimistic behaviour for exactly the cold items this is
    meant to neutralise.
    """
    # 3e-7 is about five float32 ulps at 0.5, so it survives the cast --
    # 1e-8 would not, which is itself the reason a tolerance is needed.
    scores = torch.tensor([[0.5, 0.5 + 3e-7, 0.5 - 3e-7, 0.9]])
    positive = torch.tensor([0])
    assert scores[0, 1] != scores[0, 0], "the perturbation was lost to rounding"

    assert count_ties(scores, positive, tie_atol=1e-6).item() == pytest.approx(2.0)
    assert count_ties(scores, positive, tie_atol=1e-12).item() == pytest.approx(0.0)


def test_excluded_items_must_already_be_neg_inf():
    """
    `ranks_of_positives` does not know what to exclude. -inf never
    counts as greater or tied, so the contract holds -- worth pinning,
    because ranking against a user's own train items is the easiest way
    to manufacture good numbers.
    """
    scores = torch.tensor([[0.2, float("-inf"), float("-inf"), 0.9]])
    positive = torch.tensor([0])
    assert ranks_of_positives(scores, positive).item() == pytest.approx(2.0)


def test_hit_rate_and_ndcg():
    ranks = torch.tensor([1.0, 10.0, 11.0])
    assert hit_rate_at_k(ranks, 10).tolist() == [1.0, 1.0, 0.0]

    ndcg = ndcg_at_k(ranks, 10)
    assert ndcg[0].item() == pytest.approx(1.0)
    assert ndcg[1].item() == pytest.approx(1.0 / math.log2(11))
    assert ndcg[2].item() == 0.0


def test_ndcg_handles_a_half_integer_rank():
    """`mid` produces 2.5, which a log2(rank+1) formula must accept."""
    value = ndcg_at_k(torch.tensor([2.5]), 10).item()
    assert value == pytest.approx(1.0 / math.log2(3.5))


def test_intra_list_diversity_bounds():
    identical = torch.ones(4, 3)
    ild_same = intra_list_diversity(torch.tensor([[0, 1, 2]]), identical)
    assert ild_same.item() == pytest.approx(0.0, abs=1e-6)

    orthogonal = torch.eye(4)
    ild_orth = intra_list_diversity(torch.tensor([[0, 1, 2]]), orthogonal)
    assert ild_orth.item() == pytest.approx(1.0, abs=1e-6)


def test_accumulator_reports_all_and_warm_separately():
    num_items = 20
    acc = FullCatalogMetrics(num_items, item_vectors=torch.eye(num_items))

    # user 0: clean rank 1. user 1: a cold positive tied with everything.
    scores = torch.stack([
        torch.cat([torch.tensor([5.0]), torch.zeros(num_items - 1)]),
        torch.zeros(num_items),
    ])
    positives = torch.tensor([0, 3])
    warm = torch.tensor([True, False])

    acc.update(scores, positives, warm_mask=warm)
    out = acc.compute()

    assert out["num_users_evaluated"] == 2 and out["num_users_warm"] == 1
    assert out["HR@10"] == pytest.approx(0.5)
    assert out["HR@10_warm"] == pytest.approx(1.0)
    assert out["tie_policy"] == "mid"
    assert out["frac_positives_with_ties"] == pytest.approx(0.5)
    assert 0.0 <= out["Coverage@10"] <= 1.0


def test_coverage_counts_distinct_surfaced_items():
    acc = FullCatalogMetrics(100)
    # Same top-10 for both users: 10 distinct items out of 100.
    scores = torch.zeros(2, 100)
    scores[:, :10] = torch.arange(10, 0, -1).float()
    acc.update(scores, torch.tensor([0, 1]))
    assert acc.compute()["Coverage@10"] == pytest.approx(0.10)


def test_compute_before_update_raises():
    with pytest.raises(RuntimeError, match="before any update"):
        FullCatalogMetrics(10).compute()


def test_unknown_tie_policy_raises():
    with pytest.raises(ValueError, match="tie_policy"):
        ranks_of_positives(torch.zeros(1, 4), torch.tensor([0]), tie_policy="average")


def test_optimistic_minus_mid_measures_the_artifact():
    """
    The sensitivity check the protocol calls for: the gap between the
    two policies on the same scores *is* the size of the cold-item
    artifact, so it has to be reproducible from one run's data.
    """
    num_items = 200
    scores = torch.zeros(4, num_items)
    scores[0, :] = torch.randn(num_items)            # a real ranking
    positives = torch.tensor([0, 1, 2, 3])

    hits = {}
    for policy in ("mid", "optimistic"):
        acc = FullCatalogMetrics(num_items, tie_policy=policy)
        acc.update(scores, positives)
        hits[policy] = acc.compute()["HR@10"]

    assert hits["optimistic"] >= hits["mid"]
    assert hits["optimistic"] - hits["mid"] == pytest.approx(0.75, abs=0.26), (
        "three of four users are all-zero rows: optimistic should gift them"
    )
