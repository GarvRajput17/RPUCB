"""
Model construction from a resolved config.

Formalises what `main.py`'s `build_model` was doing provisionally. The
split this module exists to make explicit is between two sources of
constructor arguments that look alike at the call site but are not:

  config-derived   scalars and lists from the merged YAML. Reproducible
                   from the resolved config alone, and written into every
                   results JSON (§6).

  dataset-derived  tensors and mappings built from the training split --
                   interaction counts, the dense item matrix, per-user
                   history. These cannot go in a YAML, must never be
                   pickled into a results file, and change with the run
                   seed because the val split is carved per seed.

Filtering by `inspect.signature` rather than maintaining a per-model kwarg
list: the ten models take overlapping but unequal argument sets, and a
hand-maintained mapping would need editing every time a constructor
changes. The signature is the mapping.

The subtlety that made the first version of this silently wrong:
`inspect.signature` reports `**kwargs` as a single VAR_KEYWORD parameter
named "kwargs", it does not expand into the names that would be accepted
through it. `MINDRPUCBMulti` and `MINDRPUCB` both take `**kwargs` and
forward to `MIND.__init__`, so a naive `if name in signature.parameters`
filter dropped every argument they do not name explicitly --
`user_train_items`, `interaction_cols`, `embed_dim`, `max_hist_len`,
`k_max`, `adaptive_k`, and for MINDRPUCB both interaction-count tensors.
Nothing raised. Those two models would have trained on all-zero history
bags with `counts=None`, which makes n_bar 1.0, which makes the
exploration term log(1/1) = 0 for every user -- RP-UCB entirely inert --
and still produced plausible-looking metrics.

`accepted_kwargs` therefore walks the MRO: a class that accepts
`**kwargs` also accepts whatever its base constructors name, because that
is exactly where the kwargs go.
"""

import inspect

from . import MATRIX_MODELS


def config_kwargs(config):
    """Everything a model might take that comes from the merged YAML."""
    return {
        "embed_dim": config.get("embed_dim", 64),
        "dropout": config.get("dropout", 0.0),
        "rl_layers": config.get("rl_layers"),
        "ml_layers": config.get("ml_layers"),
        "attn_heads": config.get("attn_heads", 2),
        "gamma_init": config.get("gamma_init", 2.0),
        "beta": config.get("beta", 1.0),
        "mask_init": config.get("mask_init", 0.5),
        "K": config.get("K", 7),
        "k_max": config.get("K", 7),
        "adaptive_k": config.get("adaptive_k", True),
        "head_mode": config.get("head_mode", "per_slot"),
        "mask_granularity": config.get("mask_granularity", "per_slot"),
        "max_hist_len": config.get("max_hist_len", 50),
        "summarization_hidden": config.get("summarization_hidden", 256),
        "n_fields": config.get("n_fields", 4),
        "n_heads_dhen": config.get("n_heads_dhen", 2),
        "transformer_layers": config.get("transformer_layers", 2),
        "routing_iters": config.get("routing_iters", 3),
        "label_aware_power": config.get("label_aware_power", 2.0),
        # Seeded per run so the history bag and MIND's symmetry-breaking
        # logits vary with the seed, same as every other stochastic choice.
        "history_seed": config.get("seed", 0),
        "logit_seed": config.get("seed", 0),
    }


def dataset_kwargs(dataset):
    """Tensors and mappings derived from the (seed-specific) train split."""
    return {
        "user_train_items": getattr(dataset, "user_train_items", None),
        "interaction_cols": getattr(dataset, "interaction_cols", None),
        "user_interaction_counts": getattr(dataset, "user_interaction_counts", None),
        "item_interaction_counts": getattr(dataset, "item_interaction_counts", None),
    }


def accepted_kwargs(cls):
    """
    Every keyword name `cls(**kwargs)` will accept, following `**kwargs`
    up the MRO.

    A class whose `__init__` declares `**kwargs` forwards them somewhere,
    and for these models that somewhere is always a base constructor. So
    when VAR_KEYWORD is present, the names its bases accept are also
    accepted here. Without this, subclasses that exist purely to preset a
    few arguments -- `MINDRPUCBMulti`, `MINDRPUCB` -- look to the filter
    like they accept almost nothing.
    """
    names = set()

    for klass in inspect.getmro(cls):
        init = klass.__dict__.get("__init__")
        if init is None:
            continue

        params = inspect.signature(init).parameters
        saw_var_keyword_here = False

        for name, param in params.items():
            if name == "self":
                continue
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                saw_var_keyword_here = True
                continue
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            names.add(name)

        if not saw_var_keyword_here:
            # This class does not forward unknown keywords, so nothing
            # further up the MRO is reachable through it.
            break

    # Internal plumbing that callers must never set: subclasses pass it
    # themselves so init_weights runs exactly once, at the leaf.
    names.discard("_defer_init")
    return names


def build_model(model_name, dataset, config):
    """
    Instantiate `model_name` for `dataset` under `config`.

    Precedence: config-derived and dataset-derived candidates first, then
    MATRIX_MODELS' per-model fixed kwargs last. §2's table is the authority
    on what makes each of the ten models that model -- a stray `K: 7` in a
    config must not be able to turn `dcm_rpucb_d` back into a
    seven-condition model.
    """
    if model_name not in MATRIX_MODELS:
        raise KeyError(
            f"unknown model {model_name!r}; known keys: {sorted(MATRIX_MODELS)}"
        )

    cls, fixed_kwargs = MATRIX_MODELS[model_name]

    candidates = config_kwargs(config)
    candidates.update(dataset_kwargs(dataset))
    candidates.update(fixed_kwargs)

    accepted = accepted_kwargs(cls)
    kwargs = {k: v for k, v in candidates.items() if k in accepted and v is not None}
    kwargs["num_users"] = dataset.num_users
    kwargs["num_items"] = dataset.num_items

    return cls(**kwargs)


def model_summary(model_name):
    """(class name, fixed kwargs) -- for logging and the provenance block."""
    cls, fixed_kwargs = MATRIX_MODELS[model_name]
    return {"class": cls.__name__, "fixed_kwargs": dict(fixed_kwargs)}