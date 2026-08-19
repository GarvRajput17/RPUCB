"""
Stage A integration smoke test -- run AFTER all modifications to existing
files are in place.

Tests that:
  1. Every new model key can be instantiated via build_model()
  2. The training branch for each new model_type runs one forward+backward
     pass without error (including DCM's argmax condition association)
  3. evaluate_model returns HR, NDCG, Coverage, and ILD for each model
  4. The new metrics (Coverage@10, ILD@10) are present and sane

Uses a tiny fake dataset constructed in-memory (no data files needed).
Should finish in <30 seconds on CPU.
"""

import torch
import numpy as np
from collections import defaultdict

# ── Minimal mock dataset matching RecDataset's public interface ─────────────
class FakeDataset:
    def __init__(self, num_users=80, num_items=60, interactions_per_user=10, seed=42):
        rng = np.random.RandomState(seed)
        self.num_users = num_users
        self.num_items = num_items

        # Build interaction matrix
        self.user_train_items = defaultdict(set)
        self.train_pairs = []
        for u in range(num_users):
            items = rng.choice(num_items, size=interactions_per_user, replace=False)
            for i in items:
                self.user_train_items[u].add(int(i))
                self.train_pairs.append((u, int(i)))

        import scipy.sparse as sp
        mat = sp.lil_matrix((num_users, num_items), dtype=np.float32)
        for u, i in self.train_pairs:
            mat[u, i] = 1.0
        csr = mat.tocsr()

        counts = np.array(csr.sum(axis=1)).flatten()
        self.user_interaction_counts = torch.LongTensor(counts)
        item_counts = np.array(csr.sum(axis=0)).flatten()
        self.item_interaction_counts = torch.LongTensor(item_counts)

        row_sums = np.maximum(1.0, counts)
        row_norm = csr.copy()
        for u in range(num_users):
            s, e = row_norm.indptr[u], row_norm.indptr[u + 1]
            row_norm.data[s:e] /= row_sums[u]
        self.interaction_rows = torch.from_numpy(row_norm.toarray()).float()

        col_sums = np.maximum(1.0, item_counts)
        col_norm = mat.tocsc()
        for i in range(num_items):
            s, e = col_norm.indptr[i], col_norm.indptr[i + 1]
            col_norm.data[s:e] /= col_sums[i]
        self.interaction_cols = torch.from_numpy(col_norm.transpose().toarray()).float()

        # Fake test data: one held-out positive + 99 negatives per user
        self.test_ratings = []
        self.test_negatives = [[] for _ in range(num_users)]
        for u in range(num_users):
            pos = list(self.user_train_items[u])[0]
            self.test_ratings.append((u, pos))
            negs = []
            while len(negs) < 99:
                j = rng.randint(0, num_items)
                if j not in self.user_train_items[u] and j != pos:
                    negs.append(j)
            self.test_negatives[u] = negs


# ── Helpers ──────────────────────────────────────────────────────────────────
def fake_config():
    return {
        'embed_dim': 32,
        'rl_layers': [64, 32],
        'ml_layers': [64, 32],
        'attn_heads': 2,
        'dropout': 0.0,
        'gamma_init': 2.0,
        'beta': 1.0,
        'lambda_l1': 1e-4,
        'K': 4,               # smaller K for speed
        'max_hist_len': 8,
        'summarization_hidden': 32,
        'n_fields': 4,
        'n_heads_dhen': 2,
        'transformer_layers': 1,
        'routing_iters': 2,
    }


def one_train_step(model, model_type, dataset, device):
    """Run one forward + backward pass through the correct training branch."""
    from src.losses import bpr_loss

    B = 6
    num_neg = 2
    rng = np.random.RandomState(0)
    users = torch.LongTensor(rng.randint(0, dataset.num_users, B)).to(device)
    pos_items = torch.LongTensor([
        list(dataset.user_train_items[u.item()])[0] for u in users
    ]).to(device)

    interaction_rows_gpu = dataset.interaction_rows.to(device)
    interaction_cols_gpu = dataset.interaction_cols.to(device)

    user_row = interaction_rows_gpu[users]
    pos_item_col = interaction_cols_gpu[pos_items]

    neg_ids = []
    for _ in range(num_neg):
        neg = torch.LongTensor(rng.randint(0, dataset.num_items, B)).to(device)
        neg_ids.append(neg)

    user_row_tiled = user_row.repeat(num_neg, 1)
    neg_cols = torch.cat([interaction_cols_gpu[n] for n in neg_ids], dim=0)
    user_ids_tiled = users.repeat(num_neg)
    neg_item_ids_flat = torch.cat(neg_ids, dim=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    if model_type == 'pinterest_base':
        pos_scores = model(user_row, pos_item_col, users)
        all_neg = model(user_row_tiled, neg_cols, user_ids_tiled)
        neg_scores = all_neg.view(num_neg, B).t()
        loss = bpr_loss(pos_scores, neg_scores)

    elif model_type in ('pinterest_base_rpucb', 'pinterest_base_rpucb_kd'):
        pos_scores, _ = model(user_row, pos_item_col, users)
        all_neg, _ = model(user_row_tiled, neg_cols, user_ids_tiled)
        neg_scores = all_neg.view(num_neg, B).t()
        loss = bpr_loss(pos_scores, neg_scores)

    elif model_type in ('pinterest_dcm', 'pinterest_dcm_rpucb'):
        pos_k, _ = model.score_multi(user_row, pos_item_col, users)
        j_star = pos_k.argmax(dim=1)
        pos_scores = pos_k[torch.arange(B, device=device), j_star]

        all_neg_k, _ = model.score_multi(user_row_tiled, neg_cols, user_ids_tiled)
        j_star_t = j_star.repeat(num_neg)
        all_neg = all_neg_k[torch.arange(B * num_neg, device=device), j_star_t]
        neg_scores = all_neg.view(num_neg, B).t()
        loss = bpr_loss(pos_scores, neg_scores)

    else:
        raise ValueError(f"Unexpected model_type: {model_type}")

    loss.backward()
    optimizer.step()
    return loss.item()


# ── Main test ────────────────────────────────────────────────────────────────
def main():
    import sys
    sys.path.insert(0, '.')
    from main import build_model
    from src.evaluate import evaluate_model

    device = 'cpu'
    dataset = FakeDataset(num_users=80, num_items=60)
    config = fake_config()

    new_models = [
        'pinterest_base',
        'pinterest_base_rpucb',
        'pinterest_base_rpucb_kd',
        'pinterest_dcm',
        'pinterest_dcm_rpucb',
    ]

    for name in new_models:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        # 1. Build
        config['model'] = name
        model = build_model(name, dataset.num_users, dataset.num_items, config, dataset)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Built OK — {n_params:,} parameters")

        # 2. Train one step
        loss = one_train_step(model, name, dataset, device)
        print(f"  Train step OK — loss={loss:.4f}")

        # 3. Evaluate
        test_ratings = dataset.test_ratings
        test_negatives = dataset.test_negatives
        results = evaluate_model(model, test_ratings, test_negatives, dataset, device, K=10)

        assert 'HR@10' in results
        assert 'NDCG@10' in results
        assert 'Coverage@10' in results
        assert 'ILD@10' in results
        assert 0.0 <= results['Coverage@10'] <= 1.0
        assert 0.0 <= results['ILD@10'] <= 1.0

        print(f"  Eval OK — HR={results['HR@10']:.4f}  NDCG={results['NDCG@10']:.4f}  "
              f"Cov={results['Coverage@10']:.4f}  ILD={results['ILD@10']:.4f}")

    print(f"\n{'='*60}")
    print("All integration smoke tests passed.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()