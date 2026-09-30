"""
Registry construction and history bags.

Both bugs covered here produced working models with plausible metrics,
which is why they need tests rather than a careful reading.

The registry bug: `inspect.signature` reports `**kwargs` as one
VAR_KEYWORD parameter, it does not expand into the names reachable
through it. `MINDRPUCBMulti` and `MINDRPUCB` both take `**kwargs` and
forward to `MIND.__init__`, so a naive `name in signature.parameters`
filter dropped `user_train_items`, `interaction_cols`, `embed_dim`,
`max_hist_len`, `k_max`, `adaptive_k`, and both interaction-count
tensors. Nothing raised. Those two models trained on all-zero history
bags with counts=None, which makes n_bar 1.0, which makes the
exploration term log(1/1) = 0 for every user -- RP-UCB entirely inert in
the two models whose entire purpose is to test it.

The history bug: bags were `list(items)[:max_hist_len]` from a set of
small ints, and CPython iterates those in roughly value order, so
truncation systematically kept the lowest item ids -- which in these
datasets correlate with insertion order and popularity.
"""

import pytest
import torch

from conftest import all_model_keys
from src.models import MATRIX_MODELS
from src.models.history import build_history_buffers, embed_history_bag
from src.models.registry import accepted_kwargs, build_model, config_kwargs, model_summary

MASKED = [k for k in all_model_keys() if "rpucb" in k]
HISTORY_MODELS = [k for k in all_model_keys() if k.startswith(("dcm", "mind"))]


@pytest.mark.parametrize("model_key", MASKED)
def test_masked_models_receive_real_interaction_counts(model_key, dataset, tiny_config):
    """
    The RP-UCB bonus is beta * sigmoid(gamma) * max(0, ln(n_bar / N)). If
    counts never arrive, RPUCBMask defaults them to all-ones, n_bar
    becomes 1.0, and the bonus is ln(1/1) = 0 for every user forever. The
    model trains, converges, and reports numbers for a mechanism that was
    switched off.
    """
    model = build_model(model_key, dataset, tiny_config())
    masks = [m for name, m in model.named_modules() if name.endswith("_mask")]
    assert masks, f"{model_key} has no RPUCBMask despite 'rpucb' in its key"

    for mask in masks:
        assert mask.n_bar > 1.0, (
            f"{model_key}: n_bar={mask.n_bar}, so the exploration term is "
            f"identically zero. Interaction counts did not reach the mask."
        )
        assert not torch.all(mask.counts == 1), f"{model_key}: counts are all ones"

    # And the bonus is actually nonzero for a sparse user.
    user_mask = getattr(model, "user_mask", None)
    if user_mask is not None:
        sparse = torch.tensor([10])          # one train item
        assert user_mask.exploration(sparse).item() > 0.0


@pytest.mark.parametrize("model_key", HISTORY_MODELS)
def test_history_bags_are_populated(model_key, dataset, tiny_config):
    """A bag of all zeros is indistinguishable from a bag of item 0."""
    model = build_model(model_key, dataset, tiny_config())
    assert hasattr(model, "history_mask"), f"{model_key} built no history buffers"

    valid_per_user = model.history_mask.sum(dim=1)
    # Every user with train items must have a non-empty bag; the
    # train-less user must have an empty one.
    for u in range(dataset.num_users):
        has_train = len(dataset.user_train_items.get(u, ())) > 0
        assert bool(valid_per_user[u] > 0) == has_train, (
            f"{model_key}: user {u} has train items={has_train} but "
            f"{int(valid_per_user[u])} history slots"
        )


def test_accepted_kwargs_follows_var_keyword_up_the_mro():
    """The mechanism behind the MIND bug, tested directly."""
    from src.models.mind_model import MIND, MINDRPUCB, MINDRPUCBMulti

    for cls in (MINDRPUCBMulti, MINDRPUCB):
        names = accepted_kwargs(cls)
        for required in ("user_train_items", "interaction_cols", "embed_dim",
                         "max_hist_len", "k_max", "adaptive_k",
                         "user_interaction_counts"):
            assert required in names, (
                f"{cls.__name__} would not receive {required!r}; this is the "
                f"exact shape of the original MIND bug"
            )
    assert "_defer_init" not in accepted_kwargs(MIND), (
        "_defer_init is internal plumbing and must never be settable from a config"
    )


def test_fixed_kwargs_beat_config():
    """
    §2's table is the authority on what makes each model that model. A
    stray `K: 7` in a config must not be able to turn `dcm_rpucb_d` --
    the 1/7-capacity row -- back into a seven-condition model, which
    would silently delete the capacity comparison.
    """
    from src.data.dataset import RecDataset  # noqa: F401  (documents the fixture's type)
    cls, fixed = MATRIX_MODELS["dcm_rpucb_d"]
    assert fixed["K"] == 1 and fixed["mask_granularity"] == "shared"


def test_fixed_kwargs_survive_a_conflicting_config(dataset, tiny_config):
    model = build_model("dcm_rpucb_d", dataset, tiny_config(K=7, mask_granularity="per_slot"))
    assert model.K == 1, "config K overrode the registry's fixed K for dcm_rpucb_d"


def test_unknown_model_key_raises(dataset, tiny_config):
    with pytest.raises(KeyError, match="unknown model"):
        build_model("deepcf_rpcub", dataset, tiny_config())


def test_model_summary_is_json_safe():
    import json
    for key in all_model_keys():
        json.dumps(model_summary(key))       # goes into every results file


def test_config_kwargs_maps_K_to_k_max():
    """MIND names it k_max, DCM names it K; both come from the one config key."""
    kwargs = config_kwargs({"K": 5})
    assert kwargs["K"] == 5 and kwargs["k_max"] == 5


# ---------------------------------------------------------------- history

def test_history_sampling_is_not_lowest_id_biased():
    """
    The original truncation kept the lowest ids. With 200 candidates and
    a cap of 10, a lowest-id truncation returns exactly 0..9 every time.
    """
    items = set(range(200))
    ids, mask = build_history_buffers(1, {0: items}, max_hist_len=10, seed=0)
    chosen = set(ids[0][mask[0]].tolist())

    assert len(chosen) == 10
    assert chosen != set(range(10)), "history bag is still a lowest-id prefix"
    assert max(chosen) > 20, f"bag {sorted(chosen)} is clustered at the low end"


def test_history_bag_is_stable_across_calls():
    """
    The bag must be identical between the positive and the negative
    forward pass inside one loss term, or the two are scoring different
    users. It is built once at construction, so this is really a test
    that nothing reshuffles it.
    """
    items = {0: set(range(50))}
    a, _ = build_history_buffers(1, items, max_hist_len=10, seed=3)
    b, _ = build_history_buffers(1, {0: set(range(50))}, max_hist_len=10, seed=3)
    assert torch.equal(a, b)

    c, _ = build_history_buffers(1, {0: set(range(50))}, max_hist_len=10, seed=4)
    assert not torch.equal(a, c), "the bag does not depend on the seed"


def test_history_shorter_than_cap_is_kept_whole():
    ids, mask = build_history_buffers(2, {0: {3, 9}, 1: set()}, max_hist_len=5, seed=0)
    assert set(ids[0][mask[0]].tolist()) == {3, 9}
    assert mask[0].sum() == 2
    assert mask[1].sum() == 0


def test_embed_history_bag_equals_the_naive_gather():
    """
    `embed_history_bag` dedupes item ids before touching the tower --
    13.8x less memory on ml-1m. It is only allowed to do that because
    `summarize` is row-independent, so the two must agree exactly.
    """
    torch.manual_seed(0)
    num_items, num_users, B, L, d = 40, 12, 5, 7, 3
    cols = torch.randn(num_items, num_users)
    summarize = torch.nn.Sequential(
        torch.nn.Linear(num_users, 16), torch.nn.GELU(), torch.nn.Linear(16, d)
    ).eval()
    hist = torch.randint(0, num_items, (B, L))

    with torch.no_grad():
        naive = summarize(cols[hist.reshape(-1)]).view(B, L, d)
        deduped = embed_history_bag(summarize, cols, hist, d)

    torch.testing.assert_close(naive, deduped, atol=1e-6, rtol=1e-6)
