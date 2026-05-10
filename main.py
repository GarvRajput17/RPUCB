import argparse
import sys
import yaml
import torch
import numpy as np

from src.data.dataset import RecDataset
from src.train import train_model
from src.models import DeepCF, DeepCFStaticMaskAttn, DeepCFRPUCB, RPUCBAttn, RPUCBAttnFull
from src.utils import save_results, print_results_table

# ── Valid model / dataset options ────────────────────────────────────────────
MODEL_CHOICES = ['deepcf', 'static_mask', 'rpucb', 'rpucb_attn', 'rpucb_attn_full']
DATASET_CHOICES = ['ml-1m', 'AMusic', 'citeulike']


def build_model(model_name, num_users, num_items, config, dataset):
    embed_dim  = config['embed_dim']
    rl_layers  = config['rl_layers']
    ml_layers  = config['ml_layers']
    attn_heads = config.get('attn_heads', 2)
    dropout    = config.get('dropout', 0.0)
    gamma_init = config.get('gamma_init', 2.0)
    beta       = config.get('beta', 1.0)

    if model_name == 'deepcf':
        return DeepCF(num_users, num_items, embed_dim,
                      rl_layers, ml_layers, dropout=dropout)

    elif model_name == 'static_mask':
        return DeepCFStaticMaskAttn(num_users, num_items, embed_dim,
                                    rl_layers, ml_layers, attn_heads=attn_heads, dropout=dropout)

    elif model_name == 'rpucb':
        return DeepCFRPUCB(
            num_users, num_items, embed_dim, rl_layers, ml_layers,
            user_interaction_counts=dataset.user_interaction_counts,
            item_interaction_counts=dataset.item_interaction_counts,
            dropout=dropout,
        )

    elif model_name == 'rpucb_attn':
        return RPUCBAttn(
            num_users, num_items, embed_dim, rl_layers, ml_layers,
            user_interaction_counts=dataset.user_interaction_counts,
            attn_heads=attn_heads, dropout=dropout,
            gamma_init=gamma_init, beta=beta,
        )

    elif model_name == 'rpucb_attn_full':
        return RPUCBAttnFull(
            num_users, num_items, embed_dim, rl_layers, ml_layers,
            user_interaction_counts=dataset.user_interaction_counts,
            item_interaction_counts=dataset.item_interaction_counts,
            attn_heads=attn_heads, dropout=dropout,
            gamma_init=gamma_init, beta=beta,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_experiment(model_name, dataset_name, config_path, device, num_runs):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['model'] = model_name

    all_results = []

    for run_id in range(num_runs):
        # Set seeds
        seed = config.get('seed', 42) + run_id
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load dataset
        dataset = RecDataset(config['data_path'], config['num_negatives'], seed=seed)

        num_users = dataset.num_users
        num_items = dataset.num_items

        # Instantiate model
        model = build_model(model_name, num_users, num_items, config, dataset)
        model = model.to(device)

        print(f"--- Running {model_name} on {dataset_name} (Run {run_id+1}/{num_runs}) ---")
        run_results = train_model(model, dataset, config, device, run_id=run_id)
        all_results.append(run_results)

    save_results(all_results, dataset_name, model_name, config)


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive User Representation Capacity in Deep Collaborative Filtering"
    )
    parser.add_argument('--model',   type=str, choices=MODEL_CHOICES)
    parser.add_argument('--dataset', type=str, choices=DATASET_CHOICES)
    parser.add_argument('--config',  type=str)
    parser.add_argument('--device',  type=str, default='auto',
                        help="cuda or cpu (default: auto-detect)")
    parser.add_argument('--runs',    type=int, default=3,
                        help="number of independent runs per experiment")
    parser.add_argument('--all',     action='store_true',
                        help=f"runs all {len(MODEL_CHOICES) * len(DATASET_CHOICES)} experiment combos")

    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    if args.all:
        for dataset in DATASET_CHOICES:
            config_path = f"configs/{dataset}.yaml"
            for model in MODEL_CHOICES:
                run_experiment(model, dataset, config_path, device, args.runs)

        print_results_table()
    else:
        if not args.model or not args.dataset or not args.config:
            print("Error: If not using --all, you must explicitly provide "
                  "--model, --dataset, and --config arguments.")
            sys.exit(1)

        run_experiment(args.model, args.dataset, args.config, device, args.runs)
        print_results_table()


if __name__ == '__main__':
    main()
