import torch
import torch.optim as optim
from tqdm import tqdm
from .losses import bpr_loss, mask_l1_loss
from .evaluate import evaluate_model


def train_one_epoch(model, dataloader, optimizer, device, lambda_l1, model_type,
                    interaction_rows_gpu, interaction_cols_gpu):
    """
    model_type: one of ['neumf', 'deepcf', 'static_mask', 'rpucb', 'rpucb_attn', 'rpucb_attn_full']
    Returns: float mean training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for step, batch in pbar:
        user_ids     = batch['user'].to(device, non_blocking=True)              # [B]
        pos_items    = batch['pos_item'].to(device, non_blocking=True)
        neg_items_idx = [n.to(device, non_blocking=True) for n in batch['neg_items']]

        user_row      = interaction_rows_gpu[user_ids]          # [B, num_items]
        pos_item_col  = interaction_cols_gpu[pos_items]         # [B, num_users]
        neg_item_cols = [interaction_cols_gpu[neg] for neg in neg_items_idx]

        optimizer.zero_grad()

        num_negatives  = len(neg_item_cols)
        B              = user_row.size(0)

        # Tiled representations for negative sampling
        user_row_tiled    = user_row.repeat(num_negatives, 1)             # [B*K, num_items]
        neg_cols_stacked  = torch.cat(neg_item_cols, dim=0)              # [B*K, num_users]
        user_ids_tiled    = user_ids.repeat(num_negatives)               # [B*K]
        neg_item_ids_flat = torch.cat([n for n in neg_items_idx], dim=0) # [B*K]

        # ── NeuMF ────────────────────────────────────────────────────────────
        if model_type == 'neumf':
            pos_scores     = model.score_neumf(user_ids, pos_items)
            all_neg_scores = model.score_neumf(user_ids_tiled, neg_item_ids_flat)
            neg_scores     = all_neg_scores.view(num_negatives, B).t()
            loss           = bpr_loss(pos_scores, neg_scores)

        # ── DeepCF (no mask) ─────────────────────────────────────────────────
        elif model_type == 'deepcf':
            pos_scores     = model(user_row, pos_item_col)
            all_neg_scores = model(user_row_tiled, neg_cols_stacked)
            neg_scores     = all_neg_scores.view(num_negatives, B).t()
            loss           = bpr_loss(pos_scores, neg_scores)

        # ── Models that need item_ids for both-side RP-UCB ──────────────────
        elif model_type in ('rpucb', 'rpucb_attn_full'):
            pos_scores, pos_mask = model(user_row, pos_item_col, user_ids, item_ids=pos_items)
            all_neg_scores, _   = model(user_row_tiled, neg_cols_stacked,
                                        user_ids_tiled, item_ids=neg_item_ids_flat)
            neg_scores = all_neg_scores.view(num_negatives, B).t()
            loss       = bpr_loss(pos_scores, neg_scores)

            # L1 sparsity on dense users' user mask component
            if lambda_l1 > 0:
                if isinstance(pos_mask, tuple):
                    u_mask = pos_mask[0]   # user mask component
                else:
                    u_mask = pos_mask
                N_u = model.user_counts[user_ids].float()
                dense_mask = u_mask[N_u >= model.n_bar_user] if hasattr(model, 'n_bar_user') \
                             else u_mask[N_u >= model.n_bar]
                if dense_mask.numel() > 0:
                    loss = loss + lambda_l1 * dense_mask.abs().mean()

        # ── Models with user-side masking only (static_mask, rpucb_attn) ────
        else:
            pos_scores, pos_mask = model(user_row, pos_item_col, user_ids)
            all_neg_scores, _   = model(user_row_tiled, neg_cols_stacked, user_ids_tiled)
            neg_scores = all_neg_scores.view(num_negatives, B).t()
            loss       = bpr_loss(pos_scores, neg_scores)

            if lambda_l1 > 0 and pos_mask is not None:
                if hasattr(model, 'n_bar'):
                    N_u = model.user_counts[user_ids].float()
                    dense_mask = pos_mask[N_u >= model.n_bar]
                    if dense_mask.numel() > 0:
                        loss = loss + lambda_l1 * dense_mask.abs().mean()
                else:
                    loss = loss + lambda_l1 * mask_l1_loss(pos_mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss_so_far = total_loss / (step + 1)
        pbar.set_postfix({'loss': f"{avg_loss_so_far:.4f}"})

    return total_loss / len(dataloader)


def train_model(model, dataset, config, device, run_id=0):
    """Full training loop with evaluation every epoch."""
    dataloader = dataset.get_train_dataloader(batch_size=config['batch_size'])
    optimizer  = optim.Adam(model.parameters(), lr=config['lr'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config.get('lr_min', 1e-5),
    )

    test_ratings, test_negatives = dataset.get_test_data()

    model_type = config.get('model', 'deepcf')
    lambda_l1  = config.get('lambda_l1', 1e-4)

    best_hr   = 0.0
    best_ndcg = 0.0
    epoch_logs = []

    # Pre-load matrices to GPU to bypass PCIe bottleneck
    interaction_rows_gpu = dataset.interaction_rows.to(device)
    interaction_cols_gpu = dataset.interaction_cols.to(device)

    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(
            model, dataloader, optimizer, device, lambda_l1, model_type,
            interaction_rows_gpu, interaction_cols_gpu,
        )
        eval_results = evaluate_model(
            model, test_ratings, test_negatives, dataset, device, K=10,
            interaction_rows_gpu=interaction_rows_gpu,
            interaction_cols_gpu=interaction_cols_gpu,
        )

        hr   = eval_results['HR@10']
        ndcg = eval_results['NDCG@10']

        if hr   > best_hr:   best_hr   = hr
        if ndcg > best_ndcg: best_ndcg = ndcg

        print(f"Epoch {epoch+1:02d}/{config['epochs']} | "
              f"Loss: {train_loss:.4f} | HR@10: {hr:.4f} | NDCG@10: {ndcg:.4f}")
        scheduler.step()

        epoch_logs.append({
            'epoch':      epoch + 1,
            'train_loss': train_loss,
            'hr':         hr,
            'ndcg':       ndcg,
        })

    return {
        'best_hr':    best_hr,
        'best_ndcg':  best_ndcg,
        'epoch_logs': epoch_logs,
    }
