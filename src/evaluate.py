import torch
import math


def evaluate_model(model, test_ratings, test_negatives, dataset, device, K=10,
                   interaction_rows_gpu=None, interaction_cols_gpu=None):
    """
    Leave-one-out evaluation.

    Args:
        model                  : any of the supported model instances (in eval mode)
        test_ratings           : list of (user_id, pos_item_id) tuples
        test_negatives         : list of lists — test_negatives[i] has 99 neg item IDs
        dataset                : RecDataset instance
        device                 : torch.device
        K                      : cutoff for HR and NDCG (default 10)
        interaction_rows_gpu   : pre-uploaded interaction rows (optional)
        interaction_cols_gpu   : pre-uploaded interaction cols (optional)

    Returns:
        dict with keys 'HR@K' and 'NDCG@K', float values
    """
    model.eval()

    if interaction_rows_gpu is None:
        interaction_rows_gpu = dataset.interaction_rows.to(device)
    if interaction_cols_gpu is None:
        interaction_cols_gpu = dataset.interaction_cols.to(device)


    hr_list   = []
    ndcg_list = []

    with torch.no_grad():
        eval_batch_size = 512
        for i in range(0, len(test_ratings), eval_batch_size):
            batch_ratings = test_ratings[i:i + eval_batch_size]

            all_user_rows  = []
            all_item_cols  = []
            all_user_ids   = []
            all_item_ids   = []   # needed for item-masked models

            for u, pos_item in batch_ratings:
                neg_items   = test_negatives[u]
                candidates  = [pos_item] + neg_items
                num_cands   = len(candidates)

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

            scores = scores.cpu().numpy()

            # ── Process per-user ─────────────────────────────────────────────
            idx = 0
            for _ in range(len(batch_ratings)):
                user_scores = scores[idx:idx + 100]
                idx += 100

                pos_score = user_scores[0]
                rank      = 1 + (user_scores[1:] > pos_score).sum()

                if rank <= K:
                    hr_list.append(1.0)
                    ndcg_list.append(1.0 / math.log2(rank + 1.0))
                else:
                    hr_list.append(0.0)
                    ndcg_list.append(0.0)

    hr_mean   = sum(hr_list)   / len(hr_list)   if hr_list   else 0.0
    ndcg_mean = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0

    return {'HR@10': hr_mean, 'NDCG@10': ndcg_mean}
