import torch
import torch.nn.functional as F
import math


def evaluate_model(model, test_ratings, test_negatives, dataset, device, K=10,
                   interaction_rows_gpu=None, interaction_cols_gpu=None):
    """
    Leave-one-out evaluation with HR@K, NDCG@K, catalog coverage, and
    intra-list diversity (ILD).

    Coverage@K : fraction of the item catalog that appears in *any* user's
                 top-K list across the test set.
    ILD@K      : mean intra-list diversity, defined as the average pairwise
                 cosine *distance* (1 - cosine_sim) among items in each
                 user's top-K, averaged across users. Uses the raw
                 interaction_cols vectors (metadata-free, identical across
                 all three datasets -- gameplan Fork 6's locked decision).
    """
    model.eval()

    if interaction_rows_gpu is None:
        interaction_rows_gpu = dataset.interaction_rows.to(device)
    if interaction_cols_gpu is None:
        interaction_cols_gpu = dataset.interaction_cols.to(device)

    hr_list   = []
    ndcg_list = []
    ild_list  = []
    all_topk_items = set()

    with torch.no_grad():
        eval_batch_size = 512
        for i in range(0, len(test_ratings), eval_batch_size):
            batch_ratings = test_ratings[i:i + eval_batch_size]

            all_user_rows  = []
            all_item_cols  = []
            all_user_ids   = []
            all_item_ids   = []
            batch_candidates = []   # track candidate IDs per user for coverage

            for u, pos_item in batch_ratings:
                neg_items   = test_negatives[u]
                candidates  = [pos_item] + neg_items
                num_cands   = len(candidates)
                batch_candidates.append(candidates)

                all_user_rows.append(
                    interaction_rows_gpu[u].unsqueeze(0).expand(num_cands, -1)
                )
                all_item_cols.append(interaction_cols_gpu[candidates])
                all_user_ids.extend([u] * num_cands)
                all_item_ids.extend(candidates)

            batch_user_rows = torch.cat(all_user_rows, dim=0)
            batch_item_cols = torch.cat(all_item_cols, dim=0)
            batch_user_ids  = torch.LongTensor(all_user_ids).to(device)
            batch_item_ids  = torch.LongTensor(all_item_ids).to(device)

            # ── Forward pass ────────────────────────────────────────────────
            scores = model(batch_user_rows, batch_item_cols,
                           batch_user_ids, batch_item_ids)
            if isinstance(scores, tuple):
                scores = scores[0]

            scores = scores.cpu()

            # ── Process per-user ─────────────────────────────────────────────
            idx = 0
            for bi in range(len(batch_ratings)):
                candidates = batch_candidates[bi]
                num_cands = len(candidates)
                user_scores = scores[idx:idx + num_cands]
                idx += num_cands

                # HR / NDCG (existing logic)
                pos_score = user_scores[0]
                rank      = 1 + (user_scores[1:] > pos_score).sum().item()

                if rank <= K:
                    hr_list.append(1.0)
                    ndcg_list.append(1.0 / math.log2(rank + 1.0))
                else:
                    hr_list.append(0.0)
                    ndcg_list.append(0.0)

                # Top-K items for coverage and diversity
                _, topk_idx = torch.topk(user_scores, min(K, num_cands))
                topk_items = [candidates[j] for j in topk_idx.tolist()]
                all_topk_items.update(topk_items)

                # Intra-list diversity (ILD)
                if len(topk_items) >= 2:
                    item_ids_t = torch.LongTensor(topk_items).to(device)
                    vecs = interaction_cols_gpu[item_ids_t]             # [K, num_users]
                    vecs = F.normalize(vecs.float(), dim=-1)
                    sim = vecs @ vecs.t()                              # [K, K]
                    n = sim.size(0)
                    mask = 1.0 - torch.eye(n, device=sim.device)
                    avg_sim = (sim * mask).sum() / (n * (n - 1))
                    ild_list.append((1.0 - avg_sim).item())
                else:
                    ild_list.append(0.0)

    hr_mean   = sum(hr_list)   / len(hr_list)   if hr_list   else 0.0
    ndcg_mean = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    ild_mean  = sum(ild_list)  / len(ild_list)  if ild_list  else 0.0
    coverage  = len(all_topk_items) / dataset.num_items if dataset.num_items > 0 else 0.0

    return {
        'HR@10':       hr_mean,
        'NDCG@10':     ndcg_mean,
        'Coverage@10': coverage,
        'ILD@10':      ild_mean,
    }