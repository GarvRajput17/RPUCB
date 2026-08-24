#!/bin/bash
set -e

DEVICE=${1:-cuda}  # pass 'cpu' as first arg to override

# Selected rows from the results table: 1, 5, 6, 7, 8, 9
#   1 = DeepCF                        -> deepcf
#   5 = RP-UCB + Attn (User+Item)     -> rpucb_attn_full
#   6 = Pinterest Tower (control)     -> pinterest_base
#   7 = Pinterest + RP-UCB (d)        -> pinterest_base_rpucb
#   8 = Pinterest + RP-UCB (K·d)      -> pinterest_base_rpucb_kd
#   9 = Pinterest DCM (K=7)           -> pinterest_dcm
#
# Skipped on purpose: static_mask, rpucb, rpucb_attn (rows 2/3/4), and the
# stretch model pinterest_dcm_rpucb (row 10).
#
# epochs=10 for all three datasets is set in configs/*.yaml directly, not
# here -- main.py has no CLI flag for it, only reads from the config file.
MODELS="deepcf rpucb_attn_full pinterest_base pinterest_base_rpucb pinterest_base_rpucb_kd pinterest_dcm"
DATASETS="ml-1m AMusic citeulike"

# 6 models x 3 datasets x 3 runs = 54 total training runs
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