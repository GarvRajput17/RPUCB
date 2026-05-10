# Adaptive User Representation Capacity in Deep Collaborative Filtering

**Garv Rajput, Pushkar Kulkarni, Aaryan Antala**  
Department of Computer Science, IIIT Bangalore

---

## Overview

Standard collaborative filtering models assign a fixed-dimensional embedding to every user and item, treating all latent dimensions as equally reliable regardless of how much interaction evidence supports them. This is problematic in sparse datasets where many users and items have very few interactions: high-variance gradients can drive useful embedding dimensions toward zero, causing representation collapse.

This work proposes **RP-UCB** (Relative Parametric Upper Confidence Bound), an uncertainty-aware masking mechanism that adaptively gates the latent dimensions of learned user and item representations before interaction scoring. The exploration bonus is derived from each entity's interaction count relative to the dataset average, keeping dimensions active for under-observed users and items while allowing the model to rely on learned parameters once sufficient evidence is available.

The mask for a user `u` (and analogously for an item `i`) is:

```
m_u = clamp( sigmoid(w_u) + beta * sigmoid(gamma_U) * max(0, ln(N_bar / N_u)),  0, 1 )
```

where `w_u` is a per-user learnable vector, `gamma_U` is a global learnable confidence vector, `N_u` is the interaction count, and `N_bar` is the dataset average.

---

## Models

| Key | Description |
|---|---|
| `deepcf` | Deep Collaborative Filtering — interaction-profile embeddings, RL + ML branches |
| `static_mask` | DeepCF with a learned sigmoid mask on user embeddings, no exploration term |
| `rpucb` | DeepCF + RP-UCB mask applied to both user and item embeddings |
| `rpucb_attn` | RP-UCB (user side) + Self-Attention interaction module |
| `rpucb_attn_full` | RP-UCB (user + item) + Self-Attention interaction module |

---

## Results (HR@10 / NDCG@10)

| Model | Scope | MovieLens-1M | Amazon Music | CiteULike-a |
|---|---|---|---|---|
| DeepCF | --- | 0.6760 / 0.3958 | 0.4465 / 0.2626 | 0.6893 / 0.4365 |
| DeepCF + Static Mask + Attn | U | 0.6808 / 0.3993 | 0.4476 / 0.2471 | 0.6772 / 0.4306 |
| DeepCF + RP-UCB | U+I | 0.6740 / 0.3980 | 0.4240 / 0.2475 | 0.6671 / 0.4223 |
| RP-UCB + Attn | U | 0.6965 / **0.4182** | 0.4336 / 0.2510 | 0.6812 / 0.4337 |
| RP-UCB + Attn | **U+I** | **0.6997** / 0.4173 | **0.4944** / **0.2731** | **0.6911** / **0.4470** |

All values are means over multiple independent runs. Improvements are relative to the DeepCF baseline.

---

## Repository Structure

```
.
├── configs/                  # Per-dataset hyperparameter configs (YAML)
│   ├── ml-1m.yaml
│   ├── AMusic.yaml
│   └── citeulike.yaml
├── data/                     # Dataset files (not tracked by git)
│   ├── ml-1m/
│   ├── AMusic/
│   └── citeulike/
├── experiments/
│   └── run_all.sh            # Run all 6 x 3 experiment combinations
├── src/
│   ├── models/
│   │   ├── neumf.py
│   │   ├── deepcf.py
│   │   ├── deepcf_static_mask.py
│   │   ├── deepcf_rpucb.py
│   │   ├── rpucb_attn.py
│   │   └── rpucb_attn_full.py
│   ├── data/
│   │   └── dataset.py        # Leave-one-out dataset with negative sampling
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── main.py                   # Experiment entry point
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dataset files (`.rating` and `.negative` format) should be placed under `data/<dataset-name>/`.

---

## Running Experiments

Single experiment:

```bash
python main.py --model rpucb_attn_full --dataset ml-1m --config configs/ml-1m.yaml --runs 3
```

All 18 combinations (6 models x 3 datasets):

```bash
bash experiments/run_all.sh cuda   # or cpu
```

Available model keys: `neumf`, `deepcf`, `static_mask`, `rpucb`, `rpucb_attn`, `rpucb_attn_full`  
Available datasets: `ml-1m`, `AMusic`, `citeulike`

---

## Datasets

| Dataset | Users | Items | Interactions | Sparsity |
|---|---|---|---|---|
| MovieLens-1M | 6,040 | 3,706 | 1,000,209 | 95.53% |
| CiteULike-a | 5,551 | 16,980 | 204,986 | 99.78% |
| Amazon Music | 1,776 | 12,929 | 46,087 | 99.80% |

Preprocessing: ratings thresholded to binary implicit feedback; users and items with fewer than 5 interactions filtered out.

---

## Evaluation Protocol

Leave-one-out evaluation: the most recent interaction per user is held out for testing. The held-out item is ranked against 99 randomly sampled negatives. Metrics reported at cutoff K=10: **HR@10** and **NDCG@10**.

Training uses pairwise BPR loss with 8 negative samples per positive interaction, Adam optimizer, and Cosine Annealing learning rate schedule.

---


