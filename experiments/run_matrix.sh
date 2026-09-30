#!/usr/bin/env bash
#
# Matrix driver: 5 datasets x 10 models x N seeds.
#
# Phase 5 (dry run) -- 1 seed, tight epoch cap. This is the gate that
# produces the real compute/disk estimate and settles where checkpoints
# live (§10: /home on worker1 was full at 19 GB, /mnt/prof-* had ~2.6 TB,
# and whether the group can write there is still open):
#
#     ./experiments/run_matrix.sh --dry-run
#
# Phase 6 (full run) -- 3 seeds, resumable:
#
#     tmux new -s matrix
#     ./experiments/run_matrix.sh --checkpoint-root /mnt/prof-XXX/checkpoints
#
# --skip-existing is always on, so a killed run resumes by re-invoking the
# same command: anything with a results JSON already on disk is skipped at
# seed granularity -- but only after main.py has confirmed the stored run
# used the same settings, so a config change cannot be silently inherited. save_run_result and save_checkpoint both write via
# temp-file-and-rename, so a job killed mid-write leaves either a complete
# artifact or none -- never a truncated one that skip-existing would
# mistake for finished.
#
# One log per run under logs/. Failures do not abort the matrix: one model
# that OOMs on the largest dataset should not cost you the other 149 runs.
# Check the summary at the end for what to re-run.

set -uo pipefail

DATASETS=${DATASETS:-"ml-1m lastfm citeulike-a AMusic AToy"}
MODELS=${MODELS:-"deepcf deepcf_rpucb deepcf_rpucb_attn mind mind_rpucb_multi mind_rpucb dcm dcm_rpucb_multi dcm_rpucb_kd dcm_rpucb_d"}
RUNS=${RUNS:-3}

CHECKPOINT_ROOT=""
RESULTS_ROOT=""
LOG_DIR=""
TAG=""
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)          DRY_RUN=1; shift ;;
    --checkpoint-root)  CHECKPOINT_ROOT="$2"; shift 2 ;;
    --results-root)     RESULTS_ROOT="$2"; shift 2 ;;
    --log-dir)          LOG_DIR="$2"; shift 2 ;;
    --tag)              TAG="$2"; shift 2 ;;
    --runs)             RUNS="$2"; shift 2 ;;
    *)                  EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ $DRY_RUN -eq 1 ]]; then
  # The dry run must not be able to pass for real results. Its first seed is
  # 42, which is also the full run's first seed, so sharing a results root
  # would let --skip-existing keep 2-epoch numbers for a third of the matrix
  # and say nothing. Separate roots plus a tag; either alone would do, and
  # both together make it impossible to mix by accident. main.py's stale
  # check is the third layer.
  RUNS=1
  TAG="${TAG:-dryrun}"
  EXTRA_ARGS+=(--max-epochs 2)
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints_dryrun}"
  RESULTS_ROOT="${RESULTS_ROOT:-results_dryrun}"
  LOG_DIR="${LOG_DIR:-logs_dryrun}"
  echo "DRY RUN: 1 seed, max_epochs=2, tag='${TAG}', roots ${RESULTS_ROOT}/ and ${CHECKPOINT_ROOT}/."
  echo "         Produces the compute/disk estimate, not results."
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
LOG_DIR="${LOG_DIR:-logs}"
[[ -n "$TAG" ]] && EXTRA_ARGS+=(--tag "$TAG")

mkdir -p "$LOG_DIR" "$RESULTS_ROOT" "$CHECKPOINT_ROOT"

n_total=0
n_failed=0
failed_runs=()
started=$(date +%s)

for dataset in $DATASETS; do
  if [[ ! -f "configs/datasets/${dataset}.yaml" ]]; then
    echo "SKIP  ${dataset}: no configs/datasets/${dataset}.yaml"
    continue
  fi
  # A config file is not data. lastfm and AToy ship complete configs but
  # their rating files are not in the repo yet, so guard on what main.py
  # will actually try to open.
  data_path=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data_path'])"               "configs/datasets/${dataset}.yaml" 2>/dev/null)
  if [[ -z "$data_path" || ! -f "${data_path}/train.rating" || ! -f "${data_path}/test.rating" ]]; then
    echo "SKIP  ${dataset}: no rating files under ${data_path:-<unset>} -- run the "
    echo "      preprocessing script in src/data/preprocess/ first"
    continue
  fi
  for model in $MODELS; do
    n_total=$((n_total + 1))
    log="${LOG_DIR}/${dataset}__${model}${TAG:+__$TAG}.log"
    printf '[%s] %-14s %-20s -> %s\n' "$(date +%H:%M:%S)" "$dataset" "$model" "$log"

    if ! python main.py \
        --dataset "$dataset" \
        --model "$model" \
        --runs "$RUNS" \
        --checkpoint-root "$CHECKPOINT_ROOT" \
        --results-root "$RESULTS_ROOT" \
        --skip-existing \
        "${EXTRA_ARGS[@]}" > "$log" 2>&1; then
      n_failed=$((n_failed + 1))
      failed_runs+=("${dataset}/${model}")
      echo "      FAILED -- see $log"
    fi
  done
done

elapsed=$(( $(date +%s) - started ))
echo
echo "=================================================================="
printf 'combos: %d   failed: %d   elapsed: %dh %dm\n' \
  "$n_total" "$n_failed" $((elapsed / 3600)) $(((elapsed % 3600) / 60))

if [[ $n_failed -gt 0 ]]; then
  echo "failed:"
  printf '  %s\n' "${failed_runs[@]}"
fi

echo
echo "checkpoint disk usage:"
du -sh "$CHECKPOINT_ROOT" 2>/dev/null || echo "  (none written)"
du -sh "$RESULTS_ROOT" 2>/dev/null

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "Scale these for Phase 6: multiply checkpoint size by the seed count,"
  echo "and wall time by roughly (real max_epochs / 2) x seed count -- runs are"
  echo "early-stopped so per-run epoch counts are data-dependent, which is"
  echo "exactly why this estimate has to come from a real pass rather than"
  echo "arithmetic."
fi

[[ $n_failed -eq 0 ]]