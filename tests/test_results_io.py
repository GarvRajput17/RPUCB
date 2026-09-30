"""
The results layer and the beta selection built on top of it.

Nothing here touches a model. These are the paths where a wrong answer
does not crash: a dry-run result kept as though it were a real one, a
beta sweep overwriting a matrix cell, a corrupt file taking out a table,
a selection rule that lets the densest dataset decide alone.
"""

import json

import pytest

from src.tuning import (
    apply_config_updates,
    average_ranks,
    beta_tag,
    config_updates,
    family_of,
    parse_grid,
    parse_tag,
    score_key,
    select_for_family,
    witness_selections,
)
from src.utils import (
    StaleResultError,
    _seed_path,
    aggregate_results,
    available_tags,
    check_existing_run,
    iter_result_files,
    load_all_results,
    paired_ttest,
    run_slug,
    save_run_result,
    split_slug,
)

FULL = {"dataset": "AMusic", "model": "deepcf", "seed": 42, "max_epochs": 100,
        "beta": 1.0, "tie_policy": "mid", "lr": 0.001}


def write_result(root, dataset, model, seed, config, tag=None, hr=0.1, **extra):
    path = _seed_path(dataset, model, seed, root, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "seed": seed,
               "test_metrics": {"HR@10": hr, "HR@100": hr * 4, "NDCG@10": hr * 0.6,
                                "Coverage@10": 0.2, "ILD@10": 0.5}}
    payload.update(extra)
    path.write_text(json.dumps(payload))
    return path


# --------------------------------------------------------------- layout

def test_run_slug_round_trips():
    assert run_slug("deepcf_rpucb") == "deepcf_rpucb"
    assert run_slug("deepcf_rpucb", "beta0.5") == "deepcf_rpucb__beta0.5"
    assert split_slug("deepcf_rpucb__beta0.5") == ("deepcf_rpucb", "beta0.5")
    assert split_slug("deepcf_rpucb") == ("deepcf_rpucb", None)


def test_no_model_key_contains_the_separator():
    """`__` is only safe as the tag separator while that holds."""
    from src.models import MATRIX_MODELS
    assert not any("__" in key for key in MATRIX_MODELS)


def test_tagged_and_untagged_runs_land_on_different_paths(tmp_path):
    plain = _seed_path("AMusic", "deepcf", 42, tmp_path)
    tagged = _seed_path("AMusic", "deepcf", 42, tmp_path, "dryrun")
    assert plain != tagged


# ------------------------------------------------- stale-result refusal

def test_identical_config_is_skippable(tmp_path):
    write_result(tmp_path, "AMusic", "deepcf", 42, FULL)
    assert check_existing_run(FULL, "AMusic", "deepcf", 42, tmp_path) is True


def test_absent_result_is_not_skipped(tmp_path):
    assert check_existing_run(FULL, "AMusic", "dcm", 42, tmp_path) is False


def test_dry_run_result_cannot_pass_for_a_real_one(tmp_path):
    """
    The sharpest case. The dry run writes seed-42 results at
    max_epochs=2, and the full run's first seed is also 42 -- without
    this check, --skip-existing would keep 2-epoch numbers for a third
    of the matrix and say nothing.
    """
    write_result(tmp_path, "AMusic", "deepcf", 42, {**FULL, "max_epochs": 2})
    with pytest.raises(StaleResultError, match="max_epochs"):
        check_existing_run(FULL, "AMusic", "deepcf", 42, tmp_path)


def test_a_changed_beta_is_refused(tmp_path):
    write_result(tmp_path, "AMusic", "deepcf", 42, {**FULL, "beta": 0.5})
    with pytest.raises(StaleResultError, match="beta"):
        check_existing_run(FULL, "AMusic", "deepcf", 42, tmp_path)


def test_refusal_names_what_differs(tmp_path):
    write_result(tmp_path, "AMusic", "deepcf", 42, {**FULL, "beta": 0.5, "max_epochs": 2})
    with pytest.raises(StaleResultError) as excinfo:
        check_existing_run(FULL, "AMusic", "deepcf", 42, tmp_path)
    message = str(excinfo.value)
    assert "beta" in message and "max_epochs" in message and "--tag" in message


def test_throughput_knobs_do_not_count_as_a_different_experiment(tmp_path):
    """
    A batch size changed to fit worker1's memory does not change what is
    being measured, and refusing on it would make the dry run's whole
    purpose unusable.
    """
    write_result(tmp_path, "AMusic", "deepcf", 42,
                 {**FULL, "user_batch_size": 16, "num_workers": 2})
    assert check_existing_run({**FULL, "user_batch_size": 64, "num_workers": 8},
                               "AMusic", "deepcf", 42, tmp_path) is True


def test_an_unreadable_result_is_treated_as_absent(tmp_path):
    path = _seed_path("AMusic", "deepcf", 42, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{truncated")
    assert check_existing_run(FULL, "AMusic", "deepcf", 42, tmp_path) is False


def test_tagged_run_at_the_same_seed_is_independent(tmp_path):
    write_result(tmp_path, "AMusic", "deepcf", 42, FULL)
    dry = {**FULL, "max_epochs": 2}
    assert check_existing_run(dry, "AMusic", "deepcf", 42, tmp_path, tag="dryrun") is False
    write_result(tmp_path, "AMusic", "deepcf", 42, dry, tag="dryrun")
    assert check_existing_run(dry, "AMusic", "deepcf", 42, tmp_path, tag="dryrun") is True


# ------------------------------------------------------------ the tree

def test_tags_are_discovered_from_directory_names(tmp_path):
    write_result(tmp_path, "AMusic", "deepcf", 42, FULL)
    write_result(tmp_path, "AMusic", "deepcf", 42, FULL, tag="beta0.5")
    write_result(tmp_path, "lastfm", "deepcf", 42, FULL, tag="dryrun")
    assert available_tags(tmp_path) == ["beta0.5", "dryrun"]


def test_aggregation_never_mixes_a_tag_into_the_matrix(tmp_path):
    """
    A status pass that folded the beta sweep in with the matrix would
    report eight seeds for cells that have three.
    """
    for seed in (42, 43, 44):
        write_result(tmp_path, "AMusic", "deepcf", seed, FULL, hr=0.1)
    for seed in (42, 43):
        write_result(tmp_path, "AMusic", "deepcf", seed, FULL, tag="beta0.5", hr=0.9)

    assert aggregate_results("AMusic", "deepcf", tmp_path)["num_seeds"] == 3
    assert aggregate_results("AMusic", "deepcf", tmp_path)["mean_HR@10"] == pytest.approx(0.1)
    assert aggregate_results("AMusic", "deepcf", tmp_path, tag="beta0.5")["num_seeds"] == 2

    assert len(load_all_results(tmp_path)) == 3
    assert len(load_all_results(tmp_path, tag="beta0.5")) == 2


def test_iter_result_files_sees_runs_nobody_asked_for(tmp_path):
    """
    A typo'd model key is invisible to a lookup keyed on the expected
    matrix, and its absence from the tables would go unexplained.
    """
    write_result(tmp_path, "AMusic", "deepcf_rpcub", 42, FULL)
    found = list(iter_result_files(tmp_path))
    assert found and found[0][1] == "deepcf_rpcub"


def test_one_corrupt_file_does_not_take_out_the_table(tmp_path):
    """
    A truncated JSON anywhere under results/ used to abort aggregation
    for all 150 runs.
    """
    for seed in (42, 43):
        write_result(tmp_path, "AMusic", "deepcf", seed, FULL, hr=0.2)
    bad = _seed_path("AMusic", "deepcf", 44, tmp_path)
    bad.write_text("{truncated")

    agg = aggregate_results("AMusic", "deepcf", tmp_path)
    assert agg["num_seeds"] == 2 and agg["mean_HR@10"] == pytest.approx(0.2)

    unreadable = [r for r in load_all_results(tmp_path) if r.get("unreadable")]
    assert len(unreadable) == 1, "the bad file was skipped without being reported"


def test_paired_ttest_uses_only_shared_seeds(tmp_path):
    for seed, hr in ((42, 0.10), (43, 0.11), (44, 0.12)):
        write_result(tmp_path, "AMusic", "deepcf", seed, FULL, hr=hr)
    for seed, hr in ((42, 0.12), (43, 0.13)):
        write_result(tmp_path, "AMusic", "deepcf_rpucb", seed, FULL, hr=hr)

    result = paired_ttest("AMusic", "deepcf", "deepcf_rpucb", root=tmp_path)
    assert result["n"] == 2 and result["seeds"] == [42, 43]
    assert result["delta"] == pytest.approx(0.02)


def test_paired_ttest_returns_none_below_two_pairs(tmp_path):
    write_result(tmp_path, "AMusic", "deepcf", 42, FULL)
    write_result(tmp_path, "AMusic", "deepcf_rpucb", 42, FULL)
    assert paired_ttest("AMusic", "deepcf", "deepcf_rpucb", root=tmp_path) is None


def test_save_run_result_writes_atomically_and_records_the_tag(tmp_path):
    result = {"test_metrics": {"HR@10": 0.3}}
    save_run_result(result, "AMusic", "deepcf", 42, config=FULL,
                     provenance={"git_sha": "abc"}, root=tmp_path, tag="dryrun")
    path = _seed_path("AMusic", "deepcf", 42, tmp_path, "dryrun")
    payload = json.loads(path.read_text())

    assert payload["tag"] == "dryrun" and payload["config"] == FULL
    assert payload["provenance"]["git_sha"] == "abc"
    assert not list(path.parent.glob("*.tmp"))


# ----------------------------------------------------- beta selection

def test_beta_tag_round_trips():
    """
    tune_beta.sh derives the tag by calling this, so the sweep and the
    selector cannot disagree about whether 1.0 is "beta1" or "beta1.0".
    """
    for value in (0, 0.25, 0.5, 1, 1.0, 2, 0.05):
        assert parse_tag(beta_tag(value)) == float(value)
    assert beta_tag(1.0) == beta_tag(1) == "beta1"


def test_parse_grid_needs_at_least_two_points():
    assert parse_grid("0,0.5,1") == [0.0, 0.5, 1.0]
    assert parse_grid("1,0.5,1") == [0.5, 1.0]        # deduplicated, sorted
    with pytest.raises(ValueError):
        parse_grid("1")


def test_family_of_matches_the_key_prefix():
    assert family_of("dcm_rpucb_kd") == "dcm"
    assert family_of("mind_rpucb_multi") == "mind"
    with pytest.raises(ValueError):
        family_of("pinterest_base")


def test_average_ranks_shares_ties():
    assert average_ranks([0.5, 0.9, 0.7]) == [3.0, 1.0, 2.0]
    assert average_ranks([0.5, 0.5, 0.9]) == [2.5, 2.5, 1.0]
    assert average_ranks([1, 1, 1]) == [2.0, 2.0, 2.0]


def make_records(truth, subsample=None):
    records = {}
    for (dataset, model), by_beta in truth.items():
        for beta, value in by_beta.items():
            sub = (subsample or {}).get((dataset, model), {}).get(beta)
            records[score_key(dataset, model, beta)] = {
                "dataset": dataset, "model": model, "beta": beta,
                "full_val": {"HR@10": value, "HR@100": value * 5,
                             "NDCG@10": value * 0.6},
                "subsample_at_best": None if sub is None else {
                    "HR@10": sub, "HR@100": sub * 5, "NDCG@10": sub * 0.6},
            }
    return records


GRID = [0.0, 0.25, 0.5, 1.0, 2.0]


def test_selection_is_not_decided_by_the_densest_dataset():
    """
    The reason selection is by mean rank rather than mean score. Here
    beta=0.5 wins narrowly on three rows and beta=2.0 wins enormously on
    one; averaging raw HR@10 would hand it to 2.0, because ml-1m's scale
    is an order of magnitude above the sparse datasets'.
    """
    truth = {
        ("ml-1m", "dcm_rpucb_multi"): {0.0: 0.30, 0.25: 0.31, 0.5: 0.32, 1.0: 0.31, 2.0: 0.90},
        ("lastfm", "dcm_rpucb_multi"): {0.0: 0.05, 0.25: 0.05, 0.5: 0.06, 1.0: 0.05, 2.0: 0.04},
        ("AMusic", "dcm_rpucb_multi"): {0.0: 0.02, 0.25: 0.02, 0.5: 0.03, 1.0: 0.02, 2.0: 0.01},
        ("AToy", "dcm_rpucb_multi"): {0.0: 0.01, 0.25: 0.02, 0.5: 0.01, 1.0: 0.01, 2.0: 0.01},
    }
    selection = select_for_family("dcm", make_records(truth), GRID, "HR@10")

    naive_mean_winner = max(
        GRID, key=lambda b: sum(row[b] for row in truth.values()) / len(truth))
    assert naive_mean_winner == 2.0, "the fixture no longer exercises the scale problem"
    assert selection["selected"] == 0.5
    assert selection["num_rows"] == 4


def test_incomplete_rows_are_excluded_not_ranked_over():
    """A missing grid point is an absence, not a loss."""
    truth = {("citeulike-a", "dcm_rpucb_multi"): {0.0: 0.04, 0.25: 0.04, 0.5: 0.05}}
    selection = select_for_family("dcm", make_records(truth), GRID, "HR@10")
    assert selection["selected"] is None
    assert "missing" in selection["excluded_rows"][0]["reason"]


def test_an_all_zero_row_is_excluded():
    truth = {("AToy", "mind_rpucb_multi"): {b: 0.0 for b in GRID}}
    selection = select_for_family("mind", make_records(truth), GRID, "HR@10")
    assert selection["selected"] is None
    assert "0 at every grid point" in selection["excluded_rows"][0]["reason"]


def test_subsample_disagreement_is_counted():
    """
    The evidence for why selection re-evaluates on the full validation
    set instead of reading val_metrics_at_best.
    """
    truth = {
        ("ml-1m", "dcm_rpucb_multi"): {0.0: 0.30, 0.25: 0.31, 0.5: 0.32, 1.0: 0.31, 2.0: 0.29},
        ("AMusic", "dcm_rpucb_multi"): {0.0: 0.02, 0.25: 0.02, 0.5: 0.03, 1.0: 0.02, 2.0: 0.01},
    }
    subsample = {
        ("ml-1m", "dcm_rpucb_multi"): {0.0: 0.30, 0.25: 0.31, 0.5: 0.32, 1.0: 0.99, 2.0: 0.29},
        ("AMusic", "dcm_rpucb_multi"): {0.0: 0.02, 0.25: 0.02, 0.5: 0.03, 1.0: 0.02, 2.0: 0.01},
    }
    selection = select_for_family("dcm", make_records(truth, subsample), GRID, "HR@10")
    assert selection["subsample_disagreements"] == 1


def test_witness_metrics_expose_a_metric_dependent_choice():
    """
    A beta that wins only under the selection metric should be visible
    as such before it goes into the config.
    """
    truth = {("ml-1m", "dcm_rpucb_multi"): {
        0.0: 0.10, 0.25: 0.20, 0.5: 0.30, 1.0: 0.25, 2.0: 0.05}}
    records = make_records(truth)
    # HR@100 and NDCG@10 are monotone transforms here, so all three agree.
    witnesses = witness_selections(records, GRID, families=("dcm",))
    assert {v["dcm"] for v in witnesses.values()} == {0.5}


def test_config_updates_cover_every_masked_member_of_a_family(tmp_path):
    from src.tuning import MASKED_MODELS

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for family, members in MASKED_MODELS.items():
        for model in members:
            (models_dir / f"{model}.yaml").write_text(
                f"# comment that must survive\nmodel: {model}\nbeta: 1.0\n")

    selections = {"deepcf": {"selected": 0.5}, "mind": {"selected": 1.0},
                  "dcm": {"selected": 0.25}}
    updates = config_updates(selections, config_root=models_dir)
    assert len(updates) == sum(len(v) for v in MASKED_MODELS.values())

    apply_config_updates(updates)
    text = (models_dir / "dcm_rpucb_kd.yaml").read_text()
    assert "beta: 0.25" in text
    assert "# comment that must survive" in text, "a YAML round-trip ate the comments"


def test_applying_updates_twice_is_a_no_op(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "deepcf_rpucb.yaml").write_text("model: deepcf_rpucb\nbeta: 1.0\n")

    selections = {"deepcf": {"selected": 0.5}}
    apply_config_updates(config_updates(selections, config_root=models_dir))
    once = (models_dir / "deepcf_rpucb.yaml").read_text()
    apply_config_updates(config_updates(selections, config_root=models_dir))
    assert (models_dir / "deepcf_rpucb.yaml").read_text() == once


def test_a_config_without_a_beta_line_is_reported_not_written(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "deepcf_rpucb.yaml").write_text("model: deepcf_rpucb\n")

    updates = config_updates({"deepcf": {"selected": 0.5}}, config_root=models_dir)
    assert updates[0]["problem"] == "no beta: line"
    apply_config_updates(updates)
    assert "beta" not in (models_dir / "deepcf_rpucb.yaml").read_text()
