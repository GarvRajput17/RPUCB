"""
Beta selection (§3/§5: "only beta tuned, on validation only").

This module holds the *decision*, not the computation. `select_beta.py`
re-evaluates each tuning checkpoint on the full validation set and hands
the resulting scores here; everything below is arithmetic over those
numbers. The split is deliberate: the selection rule is the part that
has to be defensible in the writeup and the part Checkpoint 4 will want
to test, and neither should require a GPU, a checkpoint tree, or torch
to exercise.

Two things the rule has to survive.

  Scale.      HR@10 on ml-1m and on AToy differ by more than an order of
              magnitude, so a plain mean across datasets is really just
              the densest dataset voting. Selection is therefore by mean
              *rank* across (dataset, model) rows, with mean normalised
              score as the tie-break and both printed.

  Incomplete  A row missing one beta cannot be ranked against rows that
  rows.       have all of them -- the missing cell is not a loss, it is
              an absence. Such rows are excluded and named, rather than
              being silently ranked over whatever did finish.
"""

import re
from pathlib import Path

# The three loss families of §3, keyed by the model-key prefix. This must
# partition the models exactly as LOSS_FAMILY in src/losses.py does;
# select_beta.py asserts that at startup, because the two living apart is
# only safe while something checks. The prefix is used rather than
# importing LOSS_FAMILY directly so this module stays torch-free.
FAMILIES = ("deepcf", "mind", "dcm")

FAMILY_DISPLAY = {
    "deepcf": "DeepCF (BCE)",
    "mind": "MIND (sampled softmax)",
    "dcm": "DCM (sampled softmax + logQ)",
}

# Every model carrying an RP-UCB mask, i.e. every model for which beta is
# read at all. A tuned value is written to all members of a family, not
# just the one that was swept: beta is tuned per family, so the family's
# other members inherit it by definition. The unmasked controls
# (`deepcf`, `mind`, `dcm`) are absent because their configs have no beta
# line to write -- adding one would imply the value does something.
MASKED_MODELS = {
    "deepcf": ["deepcf_rpucb", "deepcf_rpucb_attn"],
    "mind": ["mind_rpucb_multi", "mind_rpucb"],
    "dcm": ["dcm_rpucb_multi", "dcm_rpucb_kd", "dcm_rpucb_d", "dcm_rpucb_shared"],
}

# One masked model per family is swept, not all seven. Beta is a single
# per-family value, so sweeping the other members would multiply cost
# without adding a degree of freedom. The "pure mask effect" variant is
# the representative in each case -- it is the model in whose comparison
# beta most directly matters (§2).
DEFAULT_SWEEP_MODELS = ["deepcf_rpucb", "mind_rpucb_multi", "dcm_rpucb_multi"]

DEFAULT_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]

# Reported alongside whichever metric drives the choice, so a selection
# that only holds under one metric is visible as such.
WITNESS_METRICS = ["HR@10", "HR@100", "NDCG@10"]


def beta_tag(beta):
    """
    The run tag for one grid point: 0.25 -> 'beta0.25', 1.0 -> 'beta1'.

    Single definition on purpose. The tag is what names the results and
    checkpoint directories, so the sweep script and the selector have to
    agree on it exactly or the selector silently finds nothing. Rather
    than formatting it independently in bash, experiments/tune_beta.sh
    calls this.
    """
    return f"beta{float(beta):g}"


def parse_tag(tag):
    """'beta0.25' -> 0.25. Raises on anything else."""
    if not tag.startswith("beta"):
        raise ValueError(f"not a beta tag: {tag!r}")
    return float(tag[len("beta"):])


def parse_grid(text):
    """'0,0.25,0.5,1,2' -> [0.0, 0.25, 0.5, 1.0, 2.0], deduplicated, sorted."""
    values = sorted({float(part) for part in text.split(",") if part.strip()})
    if len(values) < 2:
        raise ValueError(f"a grid needs at least two values, got {text!r}")
    return values


def family_of(model):
    """Model key -> family. `dcm_rpucb_kd` -> `dcm`."""
    prefix = model.split("_", 1)[0]
    if prefix not in FAMILIES:
        raise ValueError(f"cannot place model {model!r} in a family; known: {FAMILIES}")
    return prefix


def score_key(dataset, model, beta):
    """Cache key for one grid point. Stable across runs and machines."""
    return f"{dataset}|{model}|{beta_tag(beta)}"


def average_ranks(values):
    """
    Ranks of `values`, 1 = largest, ties sharing their average rank.

    Ties matter here rather than being a formality: beta=0 and a small
    beta can produce bit-identical validation metrics on a sparse dataset
    where the exploration term never moves a top-10 list, and awarding
    one of them rank 1 by list order would be an artifact of dict
    ordering.
    """
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0.0] * len(values)

    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1  # 1-based, averaged over the tied block
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1

    return ranks


def _row_scores(records, dataset, model, grid, metric, source="full_val"):
    """The metric at each grid point for one (dataset, model), or None."""
    out = []
    for beta in grid:
        record = records.get(score_key(dataset, model, beta))
        if record is None:
            return None
        metrics = record.get(source) or {}
        if metric not in metrics:
            return None
        out.append(float(metrics[metric]))
    return out


def select_for_family(family, records, grid, metric, rows=None):
    """
    Choose beta for one family.

    `records` maps score_key -> {"full_val": {...}, "subsample_at_best":
    {...}, ...} as written by select_beta.py. `rows` restricts which
    (dataset, model) pairs are considered; by default every pair present.

    The returned dict is the audit trail, not just the answer: per-row
    scores and ranks, the rows that had to be dropped and why, and how
    often the per-epoch subsampled score would have chosen differently.
    """
    if rows is None:
        rows = sorted({
            (r["dataset"], r["model"]) for r in records.values()
            if family_of(r["model"]) == family
        })

    usable, excluded = [], []
    for dataset, model in rows:
        scores = _row_scores(records, dataset, model, grid, metric)
        if scores is None:
            excluded.append({"dataset": dataset, "model": model,
                             "reason": f"missing {metric} at one or more grid points"})
            continue
        if max(scores) <= 0:
            # Every grid point scored zero: the row carries no signal, and
            # normalising it would divide by zero. Sparse datasets with a
            # model that never lands a hit at any beta do reach this.
            excluded.append({"dataset": dataset, "model": model,
                             "reason": f"{metric} is 0 at every grid point"})
            continue
        usable.append((dataset, model, scores))

    if not usable:
        return {
            "family": family, "metric": metric, "grid": grid,
            "selected": None, "rows": [], "excluded_rows": excluded,
            "reason": "no complete row",
        }

    row_reports, rank_totals, norm_totals = [], [0.0] * len(grid), [0.0] * len(grid)
    subsample_disagreements = 0

    for dataset, model, scores in usable:
        ranks = average_ranks(scores)
        best = max(scores)
        normalised = [s / best for s in scores]

        for i in range(len(grid)):
            rank_totals[i] += ranks[i]
            norm_totals[i] += normalised[i]

        # What the cheap per-epoch signal would have chosen, had it been
        # used directly. Every disagreement here is a beta that the
        # 1,000-user subsample picks and the full validation set does not.
        subsampled = _row_scores(records, dataset, model, grid, metric,
                                 source="subsample_at_best")
        subsample_winner = None
        if subsampled is not None:
            subsample_winner = grid[subsampled.index(max(subsampled))]
            if subsample_winner != grid[scores.index(max(scores))]:
                subsample_disagreements += 1

        row_reports.append({
            "dataset": dataset,
            "model": model,
            "scores": {beta_tag(b): s for b, s in zip(grid, scores)},
            "ranks": {beta_tag(b): r for b, r in zip(grid, ranks)},
            "winner": grid[scores.index(max(scores))],
            "subsample_winner": subsample_winner,
        })

    n = len(usable)
    mean_rank = [total / n for total in rank_totals]
    mean_norm = [total / n for total in norm_totals]

    # Lowest mean rank wins; mean normalised score breaks ties. Ties are
    # common with five grid points over five rows, so the tie-break is a
    # working part, not decoration.
    best_index = min(range(len(grid)), key=lambda i: (mean_rank[i], -mean_norm[i]))
    others = sorted((i for i in range(len(grid)) if i != best_index),
                    key=lambda i: (mean_rank[i], -mean_norm[i]))
    runner_up = others[0] if others else None

    return {
        "family": family,
        "metric": metric,
        "grid": grid,
        "selected": grid[best_index],
        "runner_up": grid[runner_up] if runner_up is not None else None,
        "mean_rank": {beta_tag(b): r for b, r in zip(grid, mean_rank)},
        "mean_normalised": {beta_tag(b): v for b, v in zip(grid, mean_norm)},
        "rank_margin": (mean_rank[runner_up] - mean_rank[best_index]
                        if runner_up is not None else None),
        "rows": row_reports,
        "excluded_rows": excluded,
        "num_rows": n,
        "subsample_disagreements": subsample_disagreements,
    }


def select_all(records, grid, metric, families=FAMILIES):
    """`select_for_family` over every family that has any record."""
    present = {family_of(r["model"]) for r in records.values()}
    return {f: select_for_family(f, records, grid, metric)
            for f in families if f in present}


def witness_selections(records, grid, families=FAMILIES, metrics=WITNESS_METRICS):
    """
    The same selection under each of `metrics`.

    Printed next to the real choice so that a beta which only wins under
    the selection metric is visible as such. HR@10 under full-catalog
    evaluation lands 5-40 hits per 1,000 users (base.yaml), so a win by a
    couple of hits is worth cross-checking against HR@100 before it is
    written into the config.
    """
    return {
        metric: {f: s.get("selected")
                 for f, s in select_all(records, grid, metric, families).items()}
        for metric in metrics
    }


# ---------------------------------------------------------------- report

def render_report(selections, witnesses=None):
    """The human-readable selection report. Returns a string."""
    lines = []

    for family, sel in selections.items():
        grid = sel["grid"]
        lines.append("")
        lines.append(f"=== {FAMILY_DISPLAY.get(family, family)} "
                     f"-- selection metric {sel['metric']} ===")

        if sel["selected"] is None:
            lines.append(f"  no selection: {sel.get('reason', 'unknown')}")
            for row in sel["excluded_rows"]:
                lines.append(f"    excluded {row['dataset']}/{row['model']}: {row['reason']}")
            continue

        head = f"{'row':<28}" + "".join(f"{beta_tag(b):>12}" for b in grid)
        lines.append(head)
        lines.append("-" * len(head))

        for row in sel["rows"]:
            label = f"{row['dataset']}/{row['model']}"
            cells = ""
            for b in grid:
                value = row["scores"][beta_tag(b)]
                marker = "*" if b == row["winner"] else " "
                cells += f"{value:>11.4f}{marker}"
            lines.append(f"{label:<28}{cells}")

        lines.append("-" * len(head))
        lines.append(f"{'mean rank (lower better)':<28}"
                     + "".join(f"{sel['mean_rank'][beta_tag(b)]:>12.2f}" for b in grid))
        lines.append(f"{'mean normalised':<28}"
                     + "".join(f"{sel['mean_normalised'][beta_tag(b)]:>12.3f}" for b in grid))

        margin = sel["rank_margin"]
        lines.append("")
        lines.append(f"  selected beta = {sel['selected']:g}   "
                     f"(runner-up {sel['runner_up']:g}, mean-rank margin "
                     f"{margin:.2f} over {sel['num_rows']} rows)")

        if margin is not None and margin < 0.5:
            lines.append("  NOTE: margin under half a rank. The grid is close to flat "
                         "here; record that in the deviations register rather than "
                         "reporting a tuned optimum.")

        if sel["subsample_disagreements"]:
            lines.append(f"  {sel['subsample_disagreements']} of {sel['num_rows']} rows "
                         f"would have chosen a different beta from the per-epoch "
                         f"1,000-user subsample. That gap is why selection re-evaluates "
                         f"on the full validation set.")

        for row in sel["excluded_rows"]:
            lines.append(f"  excluded {row['dataset']}/{row['model']}: {row['reason']}")

    if witnesses:
        lines.append("")
        lines.append("=== same selection under other metrics ===")
        families = sorted({f for per in witnesses.values() for f in per})
        head = f"{'metric':<12}" + "".join(f"{f:>16}" for f in families)
        lines.append(head)
        lines.append("-" * len(head))
        for metric, per_family in witnesses.items():
            row = f"{metric:<12}"
            for f in families:
                value = per_family.get(f)
                row += f"{'--' if value is None else format(value, 'g'):>16}"
            lines.append(row)
        lines.append("")
        lines.append("  A family whose row is not constant was decided by the choice of "
                     "metric, not by beta. Say so in the writeup.")

    return "\n".join(lines)


# ---------------------------------------------------------------- config

BETA_LINE = re.compile(r"^beta:[ \t]*\S+[ \t]*$", re.MULTILINE)


def config_updates(selections, config_root="configs/models"):
    """
    The (path, old, new) edits implied by `selections`, without applying
    them. Every masked member of a family gets its family's value.
    """
    updates = []
    for family, sel in selections.items():
        if sel["selected"] is None:
            continue
        for model in MASKED_MODELS.get(family, []):
            path = Path(config_root) / f"{model}.yaml"
            if not path.is_file():
                continue
            text = path.read_text()
            match = BETA_LINE.search(text)
            if match is None:
                updates.append({"path": str(path), "old": None,
                                "new": sel["selected"], "problem": "no beta: line"})
                continue
            old = float(match.group(0).split(":", 1)[1].strip())
            updates.append({"path": str(path), "old": old, "new": sel["selected"],
                            "model": model, "family": family})
    return updates


def apply_config_updates(updates):
    """
    Rewrite the `beta:` line in each config, leaving everything else --
    comments included -- untouched.

    Only the single top-level `beta:` line is replaced, by regex rather
    than a YAML round-trip, because a round-trip would drop the comments
    in these files and those comments are the record of why each model
    exists.
    """
    written = []
    for update in updates:
        if update.get("problem"):
            continue
        path = Path(update["path"])
        text = path.read_text()
        new_text, count = BETA_LINE.subn(f"beta: {update['new']:g}", text, count=1)
        if count != 1:
            raise RuntimeError(f"{path}: expected exactly one top-level beta: line")
        if not new_text.endswith("\n"):
            new_text += "\n"
        path.write_text(new_text)
        written.append(str(path))
    return written