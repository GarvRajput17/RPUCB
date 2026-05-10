import torch
import torch.optim as optim
from tqdm import tqdm

from .evaluate import evaluate_model
from .losses import bpr_loss, mask_l1_loss
from .models.deepcf_pretrained import (
    DeepCFMLPretrain,
    DeepCFPretrained,
    DeepCFRLPretrain,
)
from .models.rpucb_user_item_pretrained import RPUCBUserItemPretrained


def _score_batch(model, model_type, user_row, item_col, user_ids, item_ids):
    if model_type == 'deepcf_pretrained':
        return model(user_row, item_col)
    scores = model(user_row, item_col, user_ids, item_ids)
    if isinstance(scores, tuple):
        return scores[0]
    return scores


def train_bpr_epoch(model, dataloader, optimizer, device, model_type,
                    interaction_rows_gpu, interaction_cols_gpu, lambda_l1=0.0):
    model.train()
    total_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for step, batch in pbar:
        user_ids = batch['user'].to(device, non_blocking=True)
        pos_items = batch['pos_item'].to(device, non_blocking=True)
        neg_items_idx = [n.to(device, non_blocking=True) for n in batch['neg_items']]

        user_row = interaction_rows_gpu[user_ids]
        pos_item_col = interaction_cols_gpu[pos_items]
        neg_item_cols = [interaction_cols_gpu[neg] for neg in neg_items_idx]

        num_negatives = len(neg_item_cols)
        batch_size = user_row.size(0)

        user_row_tiled = user_row.repeat(num_negatives, 1)
        user_ids_tiled = user_ids.repeat(num_negatives)
        neg_cols_stacked = torch.cat(neg_item_cols, dim=0)
        neg_item_ids_flat = torch.cat(neg_items_idx, dim=0)

        optimizer.zero_grad()

        pos_scores = _score_batch(
            model, model_type, user_row, pos_item_col, user_ids, pos_items
        )
        all_neg_scores = _score_batch(
            model, model_type, user_row_tiled, neg_cols_stacked,
            user_ids_tiled, neg_item_ids_flat,
        )
        neg_scores = all_neg_scores.view(num_negatives, batch_size).t()
        loss = bpr_loss(pos_scores, neg_scores)

        if model_type == 'rpucb_user_item_pretrained' and lambda_l1 > 0:
            _, masks = model.score_with_mask(user_row, pos_item_col, user_ids, pos_items)
            user_mask, item_mask = masks
            loss = loss + lambda_l1 * 0.5 * (
                mask_l1_loss(user_mask) + mask_l1_loss(item_mask)
            )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{total_loss / (step + 1):.4f}"})

    return total_loss / len(dataloader)


def _evaluate_and_log(model, dataset, config, device, epoch, phase,
                      train_loss, best_hr, best_ndcg,
                      interaction_rows_gpu, interaction_cols_gpu):
    test_ratings, test_negatives = dataset.get_test_data()
    eval_results = evaluate_model(
        model, test_ratings, test_negatives, dataset, device, K=10,
        interaction_rows_gpu=interaction_rows_gpu,
        interaction_cols_gpu=interaction_cols_gpu,
    )
    hr = eval_results['HR@10']
    ndcg = eval_results['NDCG@10']
    best_hr = max(best_hr, hr)
    best_ndcg = max(best_ndcg, ndcg)

    print(f"{phase} Epoch {epoch:02d} | "
          f"Loss: {train_loss:.4f} | HR@10: {hr:.4f} | NDCG@10: {ndcg:.4f}")

    return best_hr, best_ndcg, {
        'phase': phase,
        'epoch': epoch,
        'train_loss': train_loss,
        'hr': hr,
        'ndcg': ndcg,
    }


def train_deepcf_pretrained(dataset, config, device):
    dataloader = dataset.get_train_dataloader(batch_size=config['batch_size'])
    interaction_rows_gpu = dataset.interaction_rows.to(device)
    interaction_cols_gpu = dataset.interaction_cols.to(device)

    embed_dim = config['embed_dim']
    rl_layers = config['rl_layers']
    ml_layers = config['ml_layers']
    dropout = config.get('dropout', 0.0)
    pretrain_epochs = config.get('pretrain_epochs', max(2, config['epochs'] // 2))
    finetune_epochs = config.get('finetune_epochs', config['epochs'])
    pretrain_lr = config.get('pretrain_lr', config['lr'])

    rl_model = DeepCFRLPretrain(
        dataset.num_users, dataset.num_items, embed_dim, rl_layers, dropout
    ).to(device)
    ml_model = DeepCFMLPretrain(
        dataset.num_users, dataset.num_items, embed_dim, ml_layers, dropout
    ).to(device)

    for name, branch_model in [('DeepCF-RL-Pretrain', rl_model),
                               ('DeepCF-ML-Pretrain', ml_model)]:
        optimizer = optim.Adam(branch_model.parameters(), lr=pretrain_lr)
        for epoch in range(1, pretrain_epochs + 1):
            loss = train_bpr_epoch(
                branch_model, dataloader, optimizer, device,
                'deepcf_pretrained', interaction_rows_gpu, interaction_cols_gpu,
            )
            print(f"{name} Epoch {epoch:02d} | Loss: {loss:.4f}")

    model = DeepCFPretrained(
        dataset.num_users, dataset.num_items, embed_dim,
        rl_layers, ml_layers, dropout,
    ).to(device)
    model.init_from_pretrain(rl_model, ml_model)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs, eta_min=config.get('lr_min', 1e-5)
    )

    best_hr = 0.0
    best_ndcg = 0.0
    epoch_logs = []
    for epoch in range(1, finetune_epochs + 1):
        loss = train_bpr_epoch(
            model, dataloader, optimizer, device,
            'deepcf_pretrained', interaction_rows_gpu, interaction_cols_gpu,
        )
        best_hr, best_ndcg, log = _evaluate_and_log(
            model, dataset, config, device, epoch, 'DeepCF-Finetune',
            loss, best_hr, best_ndcg, interaction_rows_gpu, interaction_cols_gpu,
        )
        epoch_logs.append(log)
        scheduler.step()

    return {'best_hr': best_hr, 'best_ndcg': best_ndcg, 'epoch_logs': epoch_logs}


def train_rpucb_user_item_pretrained(dataset, config, device):
    dataloader = dataset.get_train_dataloader(batch_size=config['batch_size'])
    interaction_rows_gpu = dataset.interaction_rows.to(device)
    interaction_cols_gpu = dataset.interaction_cols.to(device)

    hidden_layers = config.get('rpucb_hidden_layers', config['ml_layers'])
    pretrain_epochs = config.get('pretrain_epochs', max(2, config['epochs'] // 2))
    warmup_epochs = config.get('mask_warmup_epochs', 2)
    finetune_epochs = config.get('finetune_epochs', config['epochs'])
    lambda_l1 = config.get('lambda_l1', 1e-4)

    model = RPUCBUserItemPretrained(
        dataset.num_users,
        dataset.num_items,
        embed_dim=config['embed_dim'],
        hidden_layers=hidden_layers,
        user_interaction_counts=dataset.user_interaction_counts,
        item_interaction_counts=dataset.item_interaction_counts,
        dropout=config.get('dropout', 0.0),
        gamma_init=config.get('pretrained_gamma_init', 0.1),
        beta=config.get('beta', 1.0),
        identity_mask_value=config.get('identity_mask_value', 0.95),
        pretrain_mode=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.get('pretrain_lr', config['lr']))
    for epoch in range(1, pretrain_epochs + 1):
        loss = train_bpr_epoch(
            model, dataloader, optimizer, device,
            'rpucb_user_item_pretrained',
            interaction_rows_gpu, interaction_cols_gpu,
        )
        print(f"RPUCB-Backbone-Pretrain Epoch {epoch:02d} | Loss: {loss:.4f}")

    model.set_pretrain_mode(False)
    model.init_identity_masks(
        config.get('identity_mask_value', 0.95),
        config.get('pretrained_gamma_init', 0.1),
    )

    if warmup_epochs > 0:
        model.freeze_backbone()
        optimizer = optim.Adam(model.mask_parameters(), lr=config.get('mask_warmup_lr', config['lr']))
        for epoch in range(1, warmup_epochs + 1):
            loss = train_bpr_epoch(
                model, dataloader, optimizer, device,
                'rpucb_user_item_pretrained',
                interaction_rows_gpu, interaction_cols_gpu,
                lambda_l1=lambda_l1,
            )
            print(f"RPUCB-Mask-Warmup Epoch {epoch:02d} | Loss: {loss:.4f}")

    model.unfreeze_all()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs, eta_min=config.get('lr_min', 1e-5)
    )

    best_hr = 0.0
    best_ndcg = 0.0
    epoch_logs = []
    for epoch in range(1, finetune_epochs + 1):
        loss = train_bpr_epoch(
            model, dataloader, optimizer, device,
            'rpucb_user_item_pretrained',
            interaction_rows_gpu, interaction_cols_gpu,
            lambda_l1=lambda_l1,
        )
        best_hr, best_ndcg, log = _evaluate_and_log(
            model, dataset, config, device, epoch, 'RPUCB-Finetune',
            loss, best_hr, best_ndcg, interaction_rows_gpu, interaction_cols_gpu,
        )
        epoch_logs.append(log)
        scheduler.step()

    return {'best_hr': best_hr, 'best_ndcg': best_ndcg, 'epoch_logs': epoch_logs}
