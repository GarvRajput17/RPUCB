"""
Losses.

§3 assigns each family its own native loss rather than imposing one on
architectures whose own papers use something else. That makes
LOSS_FAMILY a protocol document, so the test that matters most is that
it covers every registry key and partitions them the way the comparison
structure assumes.

The other claim worth pinning is the logQ one. Under exactly-uniform
sampling log Q(i) = log(1/num_items) for every item, so subtracting it
from every logit leaves the softmax bit-for-bit unchanged. The docstring
says this is exact rather than approximate, and "exact" is testable.
"""

import pytest
import torch

from src.losses import (
    LOSS_FAMILY,
    bce_loss,
    bpr_loss,
    mask_l1_loss,
    sampled_softmax_loss,
    uniform_log_q,
)
from src.models import MATRIX_MODELS


def test_loss_family_covers_every_registry_key():
    assert set(LOSS_FAMILY) == set(MATRIX_MODELS), (
        f"LOSS_FAMILY and MATRIX_MODELS disagree: "
        f"{set(LOSS_FAMILY) ^ set(MATRIX_MODELS)}"
    )


def test_each_family_is_internally_uniform():
    """
    §3: uniform within a family, native across families. A wrapper test
    is only a wrapper test if the backbone *and* the loss are matched on
    both sides of it.
    """
    by_prefix = {}
    for model, loss in LOSS_FAMILY.items():
        by_prefix.setdefault(model.split("_", 1)[0], set()).add(loss)

    assert by_prefix == {
        "deepcf": {"bce"},
        "mind": {"sampled_softmax"},
        "dcm": {"sampled_softmax_logq"},
    }


def test_logq_is_inert_under_uniform_sampling():
    """
    Inert by identity, not by approximation -- but in float32 only to
    rounding. The constant is log(1/3706) ~ -8.22, so every logit shifts
    by +8.22 before the logsumexp. The tolerance below is that rounding
    and nothing else; a logQ term that had actually become load-bearing
    (a non-uniform sampler, say) would move the loss by orders of
    magnitude more.
    """
    torch.manual_seed(0)
    num_items = 3706
    pos = torch.randn(32)
    neg = torch.randn(32, 8)

    plain = sampled_softmax_loss(pos, neg)
    corrected = sampled_softmax_loss(
        pos, neg,
        pos_log_q=uniform_log_q(num_items, pos.shape, pos.device),
        neg_log_q=uniform_log_q(num_items, neg.shape, neg.device),
    )
    assert plain.item() == pytest.approx(corrected.item(), rel=1e-6)


def test_logq_residual_shrinks_with_precision():
    """
    The claim isolated from float32. The residual is rounding, so it
    tracks the precision -- roughly 1e-7 relative in float32 and one ulp
    in float64 -- rather than being a property of the correction. If the
    sampler ever stops being uniform, logQ becomes load-bearing and this
    test should start failing loudly rather than drifting.
    """
    torch.manual_seed(0)
    pos = torch.randn(32, dtype=torch.float64)
    neg = torch.randn(32, 8, dtype=torch.float64)
    log_q = torch.log(torch.tensor(1 / 3706, dtype=torch.float64))

    plain = sampled_softmax_loss(pos, neg)
    corrected = sampled_softmax_loss(
        pos, neg,
        pos_log_q=log_q.expand(pos.shape), neg_log_q=log_q.expand(neg.shape),
    )
    assert plain.item() == pytest.approx(corrected.item(), rel=1e-15)


def test_uniform_log_q_value_and_shape():
    q = uniform_log_q(100, (4, 8), torch.device("cpu"))
    assert q.shape == (4, 8)
    assert torch.allclose(q, torch.full((4, 8), -torch.log(torch.tensor(100.0))))


def test_bce_rewards_separating_positives_from_negatives():
    good = bce_loss(torch.tensor([5.0, 5.0]), torch.tensor([[-5.0], [-5.0]]))
    bad = bce_loss(torch.tensor([-5.0, -5.0]), torch.tensor([[5.0], [5.0]]))
    assert good < bad and good.item() < 0.05


def test_bpr_is_zero_loss_at_infinite_margin_and_log2_at_none():
    tied = bpr_loss(torch.zeros(4), torch.zeros(4, 1))
    assert tied.item() == pytest.approx(0.6931, abs=1e-3)      # -log(0.5)
    assert bpr_loss(torch.full((4,), 20.0), torch.full((4, 1), -20.0)).item() < 1e-6


def test_bpr_accepts_one_dimensional_negatives():
    """
    `loss_override: bpr` has to work for every family, and the three
    families shape their negatives differently at the call site.
    """
    flat = bpr_loss(torch.zeros(4), torch.zeros(4))
    shaped = bpr_loss(torch.zeros(4), torch.zeros(4, 1))
    assert flat.item() == pytest.approx(shaped.item())


def test_sampled_softmax_is_cross_entropy_on_the_candidate_set():
    pos = torch.tensor([2.0])
    neg = torch.tensor([[0.0, 0.0, 0.0]])
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    assert sampled_softmax_loss(pos, neg).item() == pytest.approx(expected.item())


def test_all_losses_produce_gradients():
    for loss_fn in (bce_loss, bpr_loss, sampled_softmax_loss):
        pos = torch.randn(8, requires_grad=True)
        neg = torch.randn(8, 4, requires_grad=True)
        loss_fn(pos, neg).backward()
        assert pos.grad is not None and torch.isfinite(pos.grad).all()
        assert neg.grad is not None and torch.isfinite(neg.grad).all()


def test_mask_l1_is_a_plain_mean_absolute_value():
    assert mask_l1_loss(torch.tensor([-1.0, 1.0, 0.0])).item() == pytest.approx(2 / 3)
