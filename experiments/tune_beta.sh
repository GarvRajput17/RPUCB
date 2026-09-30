#!/usr/bin/env bash
#
# Beta sweep (§3/§5: "only beta tuned, on validation only").
#
#     ./experiments/tune_beta.sh                 # the full grid
#     ./experiments/tune_beta.sh --grid 0.5,1,2  # a narrower one
#     python select_beta.py --write              # then choose and commit
#
# Shape: 5 betas x 3 models x 5 datasets x 1 seed = 75 runs.
#
# Three models, not seven. Beta is a single per-family value, so sweeping
# the other masked members of a family would multiply the cost without
# adding a degree of freedom -- the representative is the "pure mask
# effect" variant of each family (§2), and select_beta.py writes the
# chosen value to every masked member.
#
# One seed. The sweep is a selection, not a measurement: nothing from
# here is reported. The selection is made on the full validation set by
# select_beta.py, which is where the noise this would otherwise average
# over is actually dealt with.
#
# Everything lands under results_tuning/ and checkpoints_tuning/ with a
# per-beta tag, so no tuning run can be mistaken for -- or overwrite -- a
# main-matrix result. main.py enforces the same thing from the other
# side: --beta without --tag is a hard error.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

GRID=${GRID:-"0,0.25,0.5,1,2"}
DATASETS=${DATASETS:-"ml-1m lastfm citeulike-a AMusic AToy"}
MODELS=${MODELS:-"deepcf_rpucb mind_rpucb_multi dcm_rpucb_multi"}

RESULTS_ROOT="results_tuning"
CHECKPOINT_ROOT="checkpoints_tuning"
LOG_DIR="logs_tuning"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grid)             GRID="$2"; shift 2 ;;
    --models)           MODELS="$2"; shift 2 ;;
    --datasets)         DATASETS="$2"; shift 2 ;;
    --results-root)     RESULTS_ROOT="$2"; shift 2 ;;
    --checkpoint-root)  CHECKPOINT_ROOT="$2"; shift 2 ;;
    --log-dir)          LOG_DIR="$2"; shift 2 ;;
    *)                  EXTRA_ARGS+=("$1"); shift ;;
  esac
done

echo "beta grid:  ${GRID}"
echo "models:     ${MODELS}"
echo "datasets:   ${DATASETS}"
echo "writing to: ${RESULTS_ROOT}/ and ${CHECKPOINT_ROOT}/"
echo

started=$(date +%s)

for beta in ${GRID//,/ }; do
  # The tag comes from src/tuning.py rather than being formatted here.
  # It names the results and checkpoint directories that select_beta.py
  # then has to find again, and two independent float formatters that
  # agree today ("1" vs "1.0") is a silent empty-report waiting to
  # happen.
  tag=$(python -c "from src.tuning import beta_tag; print(beta_tag($beta))") || {
    echo "could not derive a tag for beta=$beta"; exit 1; }

  echo "=================================================================="
  echo "beta=${beta}  tag=${tag}"
  echo "=================================================================="

  DATASETS="$DATASETS" MODELS="$MODELS" RUNS=1 \
  ./experiments/run_matrix.sh \
      --results-root "$RESULTS_ROOT" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --log-dir "$LOG_DIR" \
      --tag "$tag" \
      --beta "$beta" \
      "${EXTRA_ARGS[@]}"
done

elapsed=$(( $(date +%s) - started ))
echo
echo "=================================================================="
printf 'sweep finished in %dh %dm\n' $((elapsed / 3600)) $(((elapsed % 3600) / 60))
du -sh "$CHECKPOINT_ROOT" "$RESULTS_ROOT" 2>/dev/null
echo
echo "Next:  python select_beta.py --results-root ${RESULTS_ROOT} \\"
echo "                            --checkpoint-root ${CHECKPOINT_ROOT} --grid ${GRID}"
echo
echo "It re-scores every one of these checkpoints on the FULL validation set"
echo "before choosing -- the per-epoch numbers in the logs above are a 1,000"
echo "user subsample and are not what the selection is made on."