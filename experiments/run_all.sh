#!/bin/bash
set -e

DEVICE=${1:-cpu}  # Pass 'cuda' as first arg to use GPU

# ── 5 models x 3 datasets = 15 experiment combos ────────────────────────────
MODELS="deepcf static_mask rpucb rpucb_attn rpucb_attn_full"
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
