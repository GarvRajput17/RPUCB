#!/bin/bash
set -e

DEVICE=${1:-cpu}  # Pass 'cuda' as first arg to use GPU

# ── 10 models x 3 datasets = 30 experiment combos ───────────────────────────
# Ordered per gameplan §5: existing models first, then controls, then core
# rows, then stretch -- if compute runs out, the stretch row drops first.
MODELS="deepcf static_mask rpucb rpucb_attn rpucb_attn_full pinterest_base pinterest_base_rpucb pinterest_base_rpucb_kd pinterest_dcm pinterest_dcm_rpucb"
DATASETS="ml-1m AMusic citeulike"

for dataset in $DATASETS; do
    for model in $MODELS; do
        echo "=========================================="
        echo "Running: model=$model  dataset=$dataset"
        echo "=========================================="
        python main.py \
            --model $model \
            --dataset $dataset \
            --config configs/$dataset.yaml \
            --device $DEVICE \
            --runs 3
    done
done

python -c "from src.utils import print_results_table; print_results_table()"