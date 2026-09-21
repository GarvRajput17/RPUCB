"""
Training loop: per-family loss dispatch, validation-based early stopping,
best-checkpoint restoration, and a single full-catalog test pass.

Replaces the pre-refactor loop, which ran a fixed `config['epochs']`,
evaluated on the *test* set every epoch, and reported
`max(metric) over all epochs` independently for HR, NDCG, Coverage and
ILD. That leaked model selection through the test set, and meant the
reported row for a given (dataset, model) did not correspond to any
checkpoint that ever actually existed -- each metric could come from a
different epoch. Now: early stopping watches val HR@10 (§5), the best
epoch's weights are the only thing ever evaluated on test, and every
metric in the returned result comes from that one restored checkpoint,
evaluated once.

Dispatch is by loss family (3 branches -- bce, sampled_softmax,
sampled_softmax_logq), not by the 10 model keys, because every model now
shares `score_with_mask` / `score_multi` (models/base.py). The three
branches below are the three genuinely different ways a batch's loss gets
assembled, not a stand-in for "which of the 10 classes is this":

  bce                    single embedding, pointwise vs sampled negatives
  sampled_softmax_logq   K condition embeddings, argmax-selected (DCM)
  sampled_softmax        K interest embeddings, label-aware-pooled (MIND)

Negative-batch tiling matches the pre-refactor convention exactly --
`.repeat` block-tiling, not `.repeat_interleave`: for `num_negatives`
negative slots and batch size B, flat index r = k*B + b is negative slot
k of batch element b. Every family relies on this layout being identical
across `user_row_tiled`, `neg_cols_stacked`, `user_ids_tiled`,
`neg_item_ids_flat`, and (for the multi-interest families)
`j_star_tiled` / `user_vec_tiled` -- changing the tiling scheme in one
place without the others would silently misalign which negative gets
scored against which user.
"""

import torch
import torch.optim as optim
from tqdm import tqdm

from .checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from .early_stopping import EarlyStopping
from .evaluate import evaluate_split
from .losses import LOSS_FAMILY, bce_loss, bpr_loss, sampled_softmax_loss, uniform_log_q


def _mask_l1_penalty(model, user_ids, item_ids):
    """
    L1 sparsity on 'dense' entities' masks, summed across whichever of
    `user_mask` / `item_mask` the model has. Generic over all ten models
    via attribute lookup rather than a per-model-name branch: every masked
    model in this codebase names its RP-UCB masks these two things (see
    each model file's constructor) -- unmasked models have neither
    attribute and this contributes 0.0 for them automatically.
    """
    penalty = 0.0
    if hasattr(model, "user_mask"):
        penalty = penalty + model.user_mask.l1_penalty(user_ids)
    if hasattr(model, "item_mask"):
        penalty = penalty + model.item_mask.l1_penalty(item_ids)
    return penalty


def _tile_negatives(user_row, user_ids, neg_item_cols, neg_items_idx):
    num_negatives = len(neg_item_cols)
    user_row_tiled = user_row.repeat(num_negatives, 1)
    neg_cols_stacked = torch.cat(neg_item_cols, dim=0)
    user_ids_tiled = user_ids.repeat(num_negatives)
    neg_item_ids_flat = torch.cat(neg_items_idx, dim=0)
    return num_negatives, user_row_tiled, neg_cols_stacked, user_ids_tiled, neg_item_ids_flat


def compute_loss(
    model, model_name, user_row, pos_item_col, user_ids, pos_items,
    user_row_tiled, neg_cols_stacked, user_ids_tiled, neg_item_ids_flat,
    num_negatives, B, device, config,
):
    family = LOSS_FAMILY[model_name]
    loss_override = config.get("loss_override")
    lambda_l1 = config.get("lambda_l1", 0.0)

    if family == "bce":
        pos_scores, _ = model.score_with_mask(user_row, pos_item_col, user_ids, pos_items)
        neg_flat, _ = model.score_with_mask(
            user_row_tiled, neg_cols_stacked, user_ids_tiled, neg_item_ids_flat
        )
        neg_scores = neg_flat.view(num_negatives, B).t()
        native = bce_loss(pos_scores, neg_scores)

    elif family == "sampled_softmax_logq":
        # Sec 3.2.3's condition-association rule: the positive selects
        # which of the K conditions gets the gradient (j*); negatives
        # reuse that same j* rather than each picking their own argmax,
        # matching the paper's stated computational shortcut.
        pos_scores_k, _ = model.score_multi(user_row, pos_item_col, user_ids, pos_items)
        j_star = pos_scores_k.argmax(dim=1)
        pos_scores = pos_scores_k[torch.arange(B, device=device), j_star]

        neg_scores_k, _ = model.score_multi(
            user_row_tiled, neg_cols_stacked, user_ids_tiled, neg_item_ids_flat
        )
        j_star_tiled = j_star.repeat(num_negatives)
        neg_flat = neg_scores_k[torch.arange(B * num_negatives, device=device), j_star_tiled]
        neg_scores = neg_flat.view(num_negatives, B).t()

        pos_log_q = uniform_log_q(model.num_items, pos_scores.shape, device)
        neg_log_q = uniform_log_q(model.num_items, neg_scores.shape, device)
        native = sampled_softmax_loss(pos_scores, neg_scores, pos_log_q, neg_log_q)

    elif family == "sampled_softmax":
        # MIND's label-aware pooling: the positive item selects one pooled
        # user vector (mind_model.py's pool_interests), reused for scoring
        # both the positive and its negatives -- the MIND analogue of the
        # DCM branch's j* reuse above.
        interests, slot_valid = model.get_user_interests(user_ids)
        interests = model.mask_interests(interests, user_ids)
        pos_embed = model.item_embedding(pos_item_col, pos_items)
        user_vec = model.pool_interests(interests, slot_valid, pos_embed)
        pos_scores = (user_vec * pos_embed).sum(dim=-1)

        user_vec_tiled = user_vec.repeat(num_negatives, 1)
        neg_embed = model.item_embedding(neg_cols_stacked, neg_item_ids_flat)
        neg_flat = (user_vec_tiled * neg_embed).sum(dim=-1)
        neg_scores = neg_flat.view(num_negatives, B).t()

        native = sampled_softmax_loss(pos_scores, neg_scores)  # no logQ for MIND, per §3

    else:
        raise ValueError(f"unknown loss family {family!r} for model {model_name!r}")

    # BPR stays available as the §3 robustness check, composed on top of
    # whichever family derived (pos_scores, neg_scores) above.
    loss = bpr_loss(pos_scores, neg_scores) if loss_override == "bpr" else native

    if lambda_l1 > 0:
        loss = loss + lambda_l1 * _mask_l1_penalty(model, user_ids, pos_items)

    return loss


def train_one_epoch(
    model, model_name, dataloader, optimizer, device, config,
    interaction_rows_gpu, interaction_cols_gpu,
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        user_ids = batch["user"].to(device, non_blocking=True)
        pos_items = batch["pos_item"].to(device, non_blocking=True)
        neg_items_idx = [n.to(device, non_blocking=True) for n in batch["neg_items"]]

        user_row = interaction_rows_gpu[user_ids]
        pos_item_col = interaction_cols_gpu[pos_items]
        neg_item_cols = [interaction_cols_gpu[neg] for neg in neg_items_idx]

        B = user_row.size(0)
        (num_negatives, user_row_tiled, neg_cols_stacked,
         user_ids_tiled, neg_item_ids_flat) = _tile_negatives(
            user_row, user_ids, neg_item_cols, neg_items_idx
        )

        optimizer.zero_grad()
        loss = compute_loss(
            model, model_name, user_row, pos_item_col, user_ids, pos_items,
            user_row_tiled, neg_cols_stacked, user_ids_tiled, neg_item_ids_flat,
            num_negatives, B, device, config,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})

    return total_loss / len(dataloader)


def train_model(
    model, model_name, dataset, dataset_name, config, device, seed,
    checkpoint_root="checkpoints",
):
    """
    Full run for one (dataset, model, seed): train with early stopping on
    val HR@10, restore the best checkpoint, evaluate that checkpoint on
    test full-catalog exactly once.

    `max_epochs` falls back to the old `epochs` config key so existing
    YAML files keep working until Checkpoint 3's config restructure
    settles on `max_epochs: 100` explicitly per §5.
    """
    dataloader = dataset.get_train_dataloader(config["batch_size"], seed=seed)

    optimizer_name = config.get("optimizer", "adam").lower()
    if optimizer_name != "adam":
        raise ValueError(
            f"config sets optimizer={optimizer_name!r}, but §5 locks Adam and no "
            f"other optimiser is wired up. Remove the key or implement it -- do "
            f"not leave a config value that silently does nothing."
        )
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    max_epochs = config.get("max_epochs", config.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=config.get("lr_min", 1e-5)
    )
    stopper = EarlyStopping(
        patience=config.get("patience", 10),
        min_delta=config.get("min_delta", 1e-4),
        mode="max",
    )

    ckpt_path = checkpoint_path(dataset_name, model_name, seed, root=checkpoint_root)
    val_subsample_size = config.get("val_subsample_size", 1000)

    # Evaluation memory/speed knobs. These come from base.yaml rather than
    # evaluate_split's defaults so the Phase 5 dry run can tune them per
    # dataset -- AToy's 33,951-item catalog is the binding case -- without
    # editing code. Declared-but-unread config keys are worse than absent
    # ones: they look like a lever and are not.
    eval_kwargs = {
        "user_batch_size": config.get("user_batch_size", 64),
        "item_chunk_size": config.get("item_encode_chunk", 8192),
        "pair_chunk": config.get("pair_chunk", 262144),
    }

    # Preloaded once per run so val evaluation (every epoch) and training
    # both index the same resident tensors instead of re-transferring the
    # full interaction matrices on every call.
    interaction_rows_gpu = dataset.interaction_rows.to(device)
    interaction_cols_gpu = dataset.interaction_cols.to(device)

    epoch_logs = []
    final_epoch = -1

    for epoch in range(max_epochs):
        final_epoch = epoch
        train_loss = train_one_epoch(
            model, model_name, dataloader, optimizer, device, config,
            interaction_rows_gpu, interaction_cols_gpu,
        )

        val_metrics = evaluate_split(
            model, dataset, "val", device,
            max_users=val_subsample_size, seed=seed,
            interaction_rows_gpu=interaction_rows_gpu,
            interaction_cols_gpu=interaction_cols_gpu,
            **eval_kwargs,
        )
        val_hr10 = val_metrics["HR@10"]
        improved = stopper.step(val_hr10, epoch)

        if improved:
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, epoch, val_metrics,
                config, seed, early_stopping_state=stopper.state_dict(),
            )

        print(
            f"[{dataset_name}/{model_name}/seed{seed}] "
            f"epoch {epoch + 1}/{max_epochs}  loss={train_loss:.4f}  "
            f"val_HR@10={val_hr10:.4f}  best={stopper.best_score:.4f}@{stopper.best_epoch}"
        )

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        scheduler.step()

        if stopper.should_stop:
            print(f"  early stopping (no improvement for {stopper.patience} epochs)")
            break

    restored = load_checkpoint(ckpt_path, model, map_location=device)
    model.to(device)

    test_metrics = evaluate_split(
        model, dataset, "test", device, max_users=None,
        interaction_rows_gpu=interaction_rows_gpu,
        interaction_cols_gpu=interaction_cols_gpu,
        **eval_kwargs,
    )

    return {
        "test_metrics": test_metrics,
        "val_metrics_at_best": restored["val_metrics"],
        "best_epoch": restored["epoch"],
        "final_epoch": final_epoch,
        "stopping": stopper.summary(),
        "epoch_logs": epoch_logs,
    }