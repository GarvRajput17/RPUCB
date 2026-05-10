import argparse
import sys

import numpy as np
import torch
import yaml

from src.data.dataset import RecDataset
from src.train_pretrained import (
    train_deepcf_pretrained,
    train_rpucb_user_item_pretrained,
)
from src.utils import save_results


MODEL_CHOICES = ['deepcf_pretrained', 'rpucb_user_item_pretrained']
DATASET_CHOICES = ['ml-1m', 'AMusic', 'citeulike']


def run_experiment(model_name, dataset_name, config_path, device, num_runs):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['model'] = model_name
    all_results = []

    for run_id in range(num_runs):
        seed = config.get('seed', 42) + run_id
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = RecDataset(config['data_path'], config['num_negatives'], seed=seed)

        print(f"--- Running {model_name} on {dataset_name} "
              f"(Run {run_id + 1}/{num_runs}) ---")

        if model_name == 'deepcf_pretrained':
            run_results = train_deepcf_pretrained(dataset, config, device)
        elif model_name == 'rpucb_user_item_pretrained':
            run_results = train_rpucb_user_item_pretrained(dataset, config, device)
        else:
            raise ValueError(f"Unknown pretrained model: {model_name}")

        all_results.append(run_results)

    save_results(all_results, dataset_name, model_name, config)


def main():
    parser = argparse.ArgumentParser(
        description="Pretrained DeepCF and user+item RP-UCB experiments"
    )
    parser.add_argument('--model', type=str, choices=MODEL_CHOICES)
    parser.add_argument('--dataset', type=str, choices=DATASET_CHOICES)
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--all', action='store_true')

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
    else:
        if not args.model or not args.dataset or not args.config:
            print("Error: provide --model, --dataset, and --config unless using --all.")
            sys.exit(1)
        run_experiment(args.model, args.dataset, args.config, device, args.runs)


if __name__ == '__main__':
    main()
