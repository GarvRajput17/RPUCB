"""
Shared fixtures.

The synthetic dataset goes through `RecDataset` and the real model
registry rather than being hand-built. That is the whole design of this
suite: the MIND bug -- `MINDRPUCBMulti` and `MINDRPUCB` silently
receiving neither interaction counts nor history, so RP-UCB was inert and
the history bag all zeros -- lived in the registry's kwarg filtering, not
in any model. A test that constructs models directly would have passed
throughout.

It is deliberately awkward data. Twelve users with interaction counts
from eleven down to one, a user with no training rows at all, three items
that appear only in test, and a user whose single train item means the
validation carve has to skip them. Those are the cases that broke things
during the refactor, and a clean synthetic matrix would exercise none of
them.
"""

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import RecDataset  # noqa: E402

NUM_USERS = 12
NUM_ITEMS = 25
WARM_ITEMS = list(range(22))        # 0..21 appear in train
COLD_ITEMS = [22, 23, 24]           # test only: every model maps these alike
TRAINLESS_USER = 11                 # appears in test only
SINGLE_ITEM_USER = 10               # too few train items to donate one to val


def _build_rating_files(directory, seed=7):
    rng = random.Random(seed)
    train = []

    for u in range(NUM_USERS):
        if u == TRAINLESS_USER:
            continue
        k = max(1, 11 - u)
        for i in rng.sample(WARM_ITEMS, k):
            train.append((u, i))

    # Every warm item must appear at least once, or it is cold and the
    # "cold items are exactly COLD_ITEMS" assumption below is wrong.
    covered = {i for _, i in train}
    for i in WARM_ITEMS:
        if i not in covered:
            train.append((0, i))

    by_user = {}
    for u, i in train:
        by_user.setdefault(u, set()).add(i)

    test = []
    for u in range(NUM_USERS):
        if u >= 9:
            test.append((u, COLD_ITEMS[u - 9]))          # cold positives
        else:
            choices = [i for i in WARM_ITEMS if i not in by_user.get(u, set())]
            test.append((u, rng.choice(choices)))

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, pairs in (("train.rating", train), ("test.rating", test)):
        (directory / name).write_text(
            "".join(f"{u}\t{i}\t1\n" for u, i in sorted(set(pairs)))
        )
    return directory


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return _build_rating_files(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="session")
def dataset(data_dir):
    return RecDataset(str(data_dir), num_negatives=2, seed=42)


@pytest.fixture
def fresh_dataset(data_dir):
    """A per-test instance, for anything that mutates or reseeds."""
    def make(seed=42, **kwargs):
        return RecDataset(str(data_dir), num_negatives=2, seed=seed, **kwargs)
    return make


@pytest.fixture
def tiny_config():
    """
    A resolved config at toy scale.

    Mirrors the shape of what `load_config` produces -- the same keys
    `registry.config_kwargs` reads -- at dimensions that run on a CPU in
    milliseconds. The divisibility constraints `src/config.py` enforces
    hold here too: embed_dim 8 over n_fields 4 and attn_heads 2, and
    K*embed_dim = 24 over n_fields 4 for the concat-head variant.
    """
    def make(**overrides):
        config = {
            "embed_dim": 8,
            "dropout": 0.0,
            "rl_layers": [16, 8],
            "ml_layers": [16, 8],
            "attn_heads": 2,
            "gamma_init": 2.0,
            "beta": 1.0,
            "mask_init": 0.5,
            "K": 3,
            "adaptive_k": True,
            "head_mode": "per_slot",
            "mask_granularity": "per_slot",
            "max_hist_len": 5,
            "summarization_hidden": 16,
            "n_fields": 4,
            "n_heads_dhen": 2,
            "transformer_layers": 1,
            "routing_iters": 3,
            "label_aware_power": 2.0,
            "seed": 42,
            "lambda_l1": 0.0001,
            "num_negatives": 2,
            "tie_policy": "mid",
            "tie_atol": 1e-6,
        }
        config.update(overrides)
        return config
    return make


@pytest.fixture(scope="session")
def model_keys():
    from src.models import MATRIX_MODELS
    return sorted(MATRIX_MODELS)


def all_model_keys():
    """Importable at collection time, for parametrize."""
    from src.models import MATRIX_MODELS
    return sorted(MATRIX_MODELS)


@pytest.fixture(autouse=True)
def deterministic_torch():
    torch.manual_seed(1234)
    yield
