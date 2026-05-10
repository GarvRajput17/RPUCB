import os
import json
import numpy as np

# ── Canonical model list ──────────────────────────────────────────────────────
ALL_MODELS = ['neumf', 'deepcf', 'static_mask', 'rpucb', 'rpucb_attn', 'rpucb_attn_full']
ALL_DATASETS = ['ml-1m', 'AMusic', 'citeulike']

MODEL_DISPLAY = {
    'neumf':           'NeuMF',
    'deepcf':          'DeepCF',
    'static_mask':     'DeepCF + Static Mask',
    'rpucb':           'DeepCF + RP-UCB (User+Item)',
    'rpucb_attn':      'RP-UCB + Attn (User only)',
    'rpucb_attn_full': 'RP-UCB + Attn (User+Item)',
}

DATASET_DISPLAY = {
    'ml-1m':     'ML-1M',
    'AMusic':    'AMusic',
    'citeulike': 'CiteULike',
}


def save_results(results, dataset_name, model_name, config=None):
    """
    Saves per-run and aggregated results to results/{dataset}_{model}_results.json

    results: list of dicts, one per run, each with:
        - 'best_hr'   : float
        - 'best_ndcg' : float
        - 'epoch_logs': list of {epoch, train_loss, hr, ndcg}
    """
    hrs   = [res['best_hr']   for res in results]
    ndcgs = [res['best_ndcg'] for res in results]

    mean_hr   = float(np.mean(hrs))
    std_hr    = float(np.std(hrs))
    mean_ndcg = float(np.mean(ndcgs))
    std_ndcg  = float(np.std(ndcgs))

    display = MODEL_DISPLAY.get(model_name, model_name)
    print(f"\nResults for {display} on {dataset_name} over {len(results)} runs:")
    print(f"HR@10:   {mean_hr:.4f} ± {std_hr:.4f}")
    print(f"NDCG@10: {mean_ndcg:.4f} ± {std_ndcg:.4f}\n")

    out_file = f"results/{dataset_name}_{model_name}_results.json"
    os.makedirs('results', exist_ok=True)

    out_dict = {
        'model':      model_name,
        'dataset':    dataset_name,
        'num_runs':   len(results),
        'mean_hr':    mean_hr,
        'std_hr':     std_hr,
        'mean_ndcg':  mean_ndcg,
        'std_ndcg':   std_ndcg,
        'runs':       results,
    }
    if config is not None:
        out_dict['config'] = config

    with open(out_file, 'w') as f:
        json.dump(out_dict, f, indent=4)


def print_results_table():
    print("\n--- Final Results Table ---")

    # Build header
    header = f"{'Model':<30}"
    for ds in ALL_DATASETS:
        name = DATASET_DISPLAY.get(ds, ds)
        header += f"| {name} HR@10 | {name} NDCG@10 "
    print(header)
    print("-" * len(header))

    for model in ALL_MODELS:
        row = f"{MODEL_DISPLAY[model]:<30}"
        for ds in ALL_DATASETS:
            file_path = f"results/{ds}_{model}_results.json"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                hr_str   = f"{data['mean_hr']:.4f}±{data['std_hr']:.4f}"
                ndcg_str = f"{data['mean_ndcg']:.4f}±{data['std_ndcg']:.4f}"
            else:
                hr_str   = "    ---    "
                ndcg_str = "    ---    "

            row += f"| {hr_str:<13} | {ndcg_str:<16}"
        print(row)

    print("\nValues reported as: mean ± std over independent runs")
