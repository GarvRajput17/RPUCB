"""
Experiment entry point: config resolution, registry-driven model
construction, centralised seeding, provenance, per-seed skip-existing.

Models are named, not pointed at: `--dataset citeulike-a --model dcm` is
resolved against configs/base.yaml + configs/datasets/citeulike-a.yaml +
configs/models/dcm.yaml by src/config.py, which also enforces the
per-layer key whitelists. The old `--config path/to.yaml` flag is gone --
a single flat file per dataset is what allowed the pre-refactor
`citeulike.yaml: beta: 0.05` override to hide.

Choices come from what is actually on disk (configs/datasets/*.yaml,
configs/models/*.yaml), so adding a dataset or model needs no edit here.
"""

import argparse
import json
import platform
import subprocess
import time

import torch
from pathlib import Path

from src.config import (
    available_datasets,
    available_models,
    dataset_is_available,
    load_config,
    resolve_seeds,
)
from src.data.dataset import RecDataset
from src.models.registry import build_model, model_summary
from src.reproducibility import describe_determinism, set_global_seed
from src.train import train_model
from src.utils import print_results_table, run_all_significance_tests, run_result_exists, save_run_result

# The main matrix. dcm_rpucb_shared has a config file but is deliberately
# excluded (§2: available for a symmetric comparison, not a matrix row).
MATRIX_MODEL_KEYS = [
    "deepcf", "deepcf_rpucb", "deepcf_rpucb_attn",
    "mind", "mind_rpucb_multi", "mind_rpucb",
    "dcm", "dcm_rpucb_multi", "dcm_rpucb_kd", "dcm_rpucb_d",
]


def read_dataset_manifest(data_path):
    """
    The sha256 checksums and stats written by the preprocessing scripts.

    §6 requires dataset checksums in every results JSON. Returning None
    when the manifest is absent rather than raising, so a run is still
    possible before preprocessing has been run -- but the absence is
    recorded in the results file, which is what a provenance audit needs
    to see.
    """
    manifest_path = Path(data_path) / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "stats": manifest.get("stats"),
        "files": manifest.get("files"),
        "source": manifest.get("source"),
    }


def gather_provenance(config, seed, deterministic, duration_s):
    """§6: git SHA, resolved config, seed, torch/CUDA versions, GPU name,
    hostname, duration, determinism mode, dataset checksums."""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = None

    return {
        "git_sha": git_sha,
        "seed": seed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": platform.node(),
        "duration_s": duration_s,
        "determinism": describe_determinism(deterministic),
        "resolved_config": config,
        "model_spec": model_summary(config["model"]),
        "dataset_manifest": read_dataset_manifest(config["data_path"]),
    }


def run_experiment(model_name, dataset_name, device, seeds, checkpoint_root,
                    results_root, deterministic, skip_existing, overrides=None):
    for seed in seeds:
        if skip_existing and run_result_exists(dataset_name, model_name, seed, results_root):
            print(f"[skip] {dataset_name}/{model_name}/seed{seed} already has a result")
            continue

        config = load_config(dataset_name, model_name, overrides=overrides)
        config["seed"] = seed

        set_global_seed(seed, deterministic=deterministic)

        dataset = RecDataset(config["data_path"], config["num_negatives"], seed=seed)
        model = build_model(model_name, dataset, config).to(device)

        print(f"--- {dataset_name}/{model_name}  seed={seed} ---")
        t0 = time.time()
        result = train_model(model, model_name, dataset, dataset_name, config, device, seed,
                              checkpoint_root=checkpoint_root)
        duration = time.time() - t0

        provenance = gather_provenance(config, seed, deterministic, duration)
        save_run_result(result, dataset_name, model_name, seed, config=config,
                         provenance=provenance, root=results_root)


def main():
    datasets_on_disk = available_datasets()
    models_on_disk = available_models()

    parser = argparse.ArgumentParser(
        description="Adaptive User Representation Capacity in Deep Collaborative Filtering"
    )
    parser.add_argument("--model", choices=models_on_disk)
    parser.add_argument("--dataset", choices=datasets_on_disk)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs", type=int, default=3,
                         help="number of seeds, base_seed..base_seed+runs-1")
    parser.add_argument("--seeds", default=None,
                         help="explicit comma-separated seeds, overrides --runs")
    parser.add_argument("--all", action="store_true",
                         help="run the full matrix: every dataset x the 10 matrix models")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                         help="bit-exact at a throughput cost; see src/reproducibility.py")
    parser.add_argument("--max-epochs", type=int, default=None,
                         help="override; the Phase 5 dry run uses a small value")
    parser.add_argument("--beta", type=float, default=None,
                         help="override for validation tuning (§3)")
    parser.add_argument("--mask-init", type=float, default=None,
                         help="override; ~1.0 gives the parity-start robustness run")
    parser.add_argument("--loss-override", default=None, choices=["bpr"],
                         help="§3 robustness check")

    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Using device: {device}")

    overrides = {
        "max_epochs": args.max_epochs,
        "beta": args.beta,
        "mask_init": args.mask_init,
        "loss_override": args.loss_override,
    }

    if args.all:
        usable = []
        for dataset in datasets_on_disk:
            available, reason = dataset_is_available(dataset)
            if available:
                usable.append(dataset)
            else:
                print(f"[skip dataset] {dataset}: {reason}")
        if not usable:
            parser.error("no dataset has its rating files on disk; run the "
                          "preprocessing scripts in src/data/preprocess/ first")
        targets = [(d, m) for d in usable for m in MATRIX_MODEL_KEYS]
    else:
        if not args.model or not args.dataset:
            parser.error("without --all you must pass --model and --dataset")
        available, reason = dataset_is_available(args.dataset)
        if not available:
            parser.error(
                f"{args.dataset}: {reason}. See configs/datasets/{args.dataset}.yaml "
                f"for where the data comes from, then run the matching script in "
                f"src/data/preprocess/."
            )
        targets = [(args.dataset, args.model)]

    # Seeds come from base.yaml's base_seed, which any resolved config carries.
    probe = load_config(targets[0][0], targets[0][1])
    seeds = resolve_seeds(probe, runs=args.runs, explicit=args.seeds)
    print(f"Seeds: {seeds}  ({len(targets)} model/dataset combos "
          f"= {len(targets) * len(seeds)} runs)")

    for dataset_name, model_name in targets:
        run_experiment(model_name, dataset_name, device, seeds,
                        args.checkpoint_root, args.results_root,
                        args.deterministic, args.skip_existing, overrides)

    if args.all:
        ran = sorted({d for d, _ in targets})
        print_results_table(datasets=ran)
        run_all_significance_tests(datasets=ran)
    else:
        print_results_table(models=[args.model], datasets=[args.dataset])


if __name__ == "__main__":
    main()