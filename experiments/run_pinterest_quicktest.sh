#!/bin/bash
set -e

DEVICE=${1:-cuda}  # your laptop has a GPU -- default to it; pass 'cpu' to override

# Control + core rows only (gameplan §6), skips the stretch row
# (pinterest_dcm_rpucb) intentionally -- this script is for confirming the
# pipeline runs end-to-end, not for producing real numbers.
MODELS="pinterest_base pinterest_base_rpucb pinterest_base_rpucb_kd pinterest_dcm"

for model in $MODELS; do
    echo "=========================================="
    echo "Running: model=$model  dataset=AMusic (quicktest)"
    echo "=========================================="
    python main.py \
        --model $model \
        --dataset AMusic \
        --config configs/AMusic_quicktest.yaml \
        --device $DEVICE \
        --runs 1
done

echo ""
echo "Pipeline check complete. If all 4 models printed an Epoch line with a"
echo "non-NaN loss and a results/AMusic_<model>_results.json file exists for"
echo "each, the pipeline is running correctly end-to-end."