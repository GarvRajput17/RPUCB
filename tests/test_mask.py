"""
The RP-UCB mask.

Before the refactor this formula existed in three places and two of them
agreed: `deepcf_rpucb.py` used `sigmoid(w + gamma * explore)` with gamma
initialised to zeros -- exploration inert from step one -- while the
other two used the canonical form. Model 2 of the matrix was therefore
testing a different mechanism from models 5, 6, 8, 9 and 10, with
nothing in the code saying so.

The formula is now in one module, so what needs testing is that the one
implementation is the intended one:

    m = clamp( sigmoid(w) + beta * sigmoid(gamma) * max(0, ln(n_bar/N)), 0, 1 )
"""

import math

import pytest
import torch

from src.models.rpucb_mask import RPUCBMask


def make_mask(counts, **kwargs):
    return RPUCBMask(num_entities=len(counts), dim=4,
                     counts=torch.tensor(counts), **kwargs)


def test_matches_the_closed_form():
    counts = [1, 2, 10, 40]
    mask = make_mask(counts, gamma_init=2.0, beta=0.75, mask_init=0.5)
    ids = torch.arange(4)

    with torch.no_grad():
        got = mask(ids)
        w = mask.w(ids).unsqueeze(1)
        n = torch.tensor(counts, dtype=torch.float32)
        explore = torch.clamp(torch.log(torch.tensor(mask.n_bar) / n), min=0.0)
        expected = (torch.sigmoid(w)
                    + 0.75 * explore.view(-1, 1, 1) * torch.sigmoid(mask.gamma)).clamp(0, 1)

    torch.testing.assert_close(got, expected)


def test_mask_init_is_the_value_of_sigmoid_w():
    """
    `mask_init` is a modelling choice, not a detail: at 0.5 with both
    sides masked, a dense user/item pair starts at 0.25x the unmasked
    control's signal. The parity-start robustness run sets it near 1.0,
    so the constructor has to honour it.
    """
    for target in (0.5, 0.9, 0.99):
        mask = make_mask([100] * 8, mask_init=target)   # dense: no bonus
        with torch.no_grad():
            realised = torch.sigmoid(mask.w.weight).mean().item()
        assert abs(realised - target) < 0.01, f"mask_init={target} -> {realised}"


def test_dense_entity_gets_no_exploration_bonus():
    mask = make_mask([1, 1, 1, 100])
    # n_bar is the mean count, so entity 3 is far above it.
    assert mask.exploration(torch.tensor([3])).item() == 0.0
    assert mask.exploration(torch.tensor([0])).item() > 0.0


def test_gamma_init_zero_would_not_disable_exploration():
    """
    The old DeepCF bug put gamma inside the sigmoid with a zero init, so
    the bonus started at exactly zero. In the canonical form gamma is
    outside, and sigmoid(0) = 0.5, so a zero init merely halves the
    bonus. Worth pinning: it is the difference between "weakened" and
    "absent", and the docstring tells people not to set it to 0.0.
    """
    mask = make_mask([1, 1, 1, 50], gamma_init=0.0, mask_init=0.5)
    with torch.no_grad():
        m = mask(torch.tensor([0]))
    assert m.min().item() > 0.5, "exploration bonus vanished at gamma_init=0"


def test_clamped_into_the_unit_interval():
    mask = make_mask([1] + [10_000] * 50, beta=50.0)     # enormous bonus
    with torch.no_grad():
        m = mask(torch.arange(51))
    assert m.min() >= 0.0 and m.max() <= 1.0


def test_saturation_reports_the_zero_gradient_regime():
    """
    Above the upper clamp `torch.clamp` passes zero gradient, so the
    learnable parts of the mask get no signal at all from the sparsest
    users. A null result on the sparse datasets could be that rather
    than anything about RP-UCB, which is why this is logged per epoch.
    """
    saturated = make_mask([1] + [10_000] * 50, beta=50.0)
    assert saturated.saturation(torch.tensor([0])) == pytest.approx(1.0)

    unsaturated = make_mask([100] * 20, mask_init=0.5)
    assert unsaturated.saturation(torch.arange(20)) == pytest.approx(0.0)


def test_l1_penalty_exempts_sparse_entities():
    """
    Driving a sparse entity's mask toward zero would fight the
    exploration bonus that is meant to keep it open, so the penalty
    applies to dense entities only.
    """
    mask = make_mask([1, 1, 1, 1, 200])
    sparse_only = mask.l1_penalty(torch.tensor([0, 1, 2]))
    assert sparse_only == 0.0 and not torch.is_tensor(sparse_only), (
        "a plain 0.0 is the documented contract: callers add it unconditionally"
    )
    assert torch.is_tensor(mask.l1_penalty(torch.tensor([0, 4])))


def test_per_slot_masks_are_independent():
    mask = RPUCBMask(num_entities=5, dim=4, counts=torch.tensor([3] * 5), n_slots=3)
    with torch.no_grad():
        m = mask(torch.tensor([0, 1]))
    assert m.shape == (2, 3, 4)
    assert not torch.allclose(m[0, 0], m[0, 1]), "per-slot masks are identical"


def test_flat_requires_a_shared_mask():
    shared = RPUCBMask(5, 4, counts=torch.ones(5, dtype=torch.long), n_slots=1)
    assert shared.flat(torch.tensor([0, 1])).shape == (2, 4)

    per_slot = RPUCBMask(5, 4, counts=torch.ones(5, dtype=torch.long), n_slots=3)
    with pytest.raises(AssertionError):
        per_slot.flat(torch.tensor([0]))


def test_counts_stay_out_of_the_state_dict():
    """persistent=False: 150 checkpoints on shared storage."""
    mask = make_mask([1, 2, 3])
    assert "counts" not in mask.state_dict()
    assert {"w.weight", "gamma"} <= set(mask.state_dict())


def test_init_weights_does_not_clobber_mask_init(dataset, tiny_config):
    """
    `BaseCF.init_weights` re-randomises every Embedding to normal(0,
    0.01). The mask's `w` is an Embedding, so without the
    `_skip_default_init` flag a mask_init of 0.9 would silently become
    sigmoid(0) = 0.5 -- and the parity-start robustness run would be
    measuring nothing.
    """
    from src.models.registry import build_model
    model = build_model("deepcf_rpucb", dataset, tiny_config(mask_init=0.9))
    with torch.no_grad():
        realised = torch.sigmoid(model.user_mask.w.weight).mean().item()
    assert abs(realised - 0.9) < 0.02, f"mask_init was overwritten: {realised}"


def test_beta_changes_the_mask_for_sparse_entities_only():
    """What the beta sweep is actually varying."""
    low = make_mask([1, 1, 200, 200], beta=0.1)
    high = make_mask([1, 1, 200, 200], beta=2.0)
    high.load_state_dict(low.state_dict())      # same w, same gamma

    with torch.no_grad():
        m_low, m_high = low(torch.arange(4)), high(torch.arange(4))

    assert (m_high[0] > m_low[0]).any(), "beta had no effect on a sparse entity"
    torch.testing.assert_close(m_low[2:], m_high[2:])
