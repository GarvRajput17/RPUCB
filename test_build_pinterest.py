import torch
from collections import defaultdict

from src.models.pinterest_base import PinterestBase
from src.models.pinterest_base_rpucb import PinterestBaseRPUCB
from src.models.pinterest_dcm import PinterestDCM
from src.models.pinterest_dcm_rpucb import PinterestDCMRPUCB


def _fake_user_train_items(num_users, num_items, min_items=6, max_items=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = defaultdict(set)
    for u in range(num_users):
        n = int(torch.randint(min_items, max_items, (1,), generator=g).item())
        items = torch.randint(0, num_items, (n,), generator=g).tolist()
        out[u].update(items)
    return out


def _fake_batch(num_users, num_items, batch_size):
    user_row = torch.rand(batch_size, num_items)
    item_col = torch.rand(batch_size, num_users)
    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    return user_row, item_col, user_ids, item_ids


def test_pinterest_base():
    print("\n[PinterestBase] Initializing...")
    num_users, num_items, embed_dim, batch_size = 100, 50, 64, 8
    model = PinterestBase(num_users, num_items, embed_dim=embed_dim)

    user_row, item_col, user_ids, item_ids = _fake_batch(num_users, num_items, batch_size)
    scores = model.score(user_row, item_col, user_ids, item_ids)
    print("Output shape:", scores.shape)
    assert scores.shape == (batch_size,)
    print("[PinterestBase] OK")


def test_pinterest_base_rpucb():
    print("\n[PinterestBaseRPUCB] Initializing...")
    num_users, num_items, embed_dim, batch_size = 100, 50, 64, 8
    counts = torch.randint(1, 30, (num_users,)).long()
    model = PinterestBaseRPUCB(
        num_users, num_items, embed_dim=embed_dim,
        user_interaction_counts=counts,
    )

    user_row, item_col, user_ids, item_ids = _fake_batch(num_users, num_items, batch_size)
    scores, mask = model.score_with_mask(user_row, item_col, user_ids, item_ids)
    print("Score shape:", scores.shape, "| Mask shape:", mask.shape)
    assert scores.shape == (batch_size,)
    assert mask.shape == (batch_size, embed_dim)
    assert (mask >= 0).all() and (mask <= 1).all()

    # Also test at K*d width (the matched-capacity config from the target table)
    K, d = 7, 64
    model_wide = PinterestBaseRPUCB(
        num_users, num_items, embed_dim=K * d,
        user_interaction_counts=counts,
    )
    scores_wide, mask_wide = model_wide.score_with_mask(user_row, item_col, user_ids, item_ids)
    assert scores_wide.shape == (batch_size,)
    assert mask_wide.shape == (batch_size, K * d)
    print("[PinterestBaseRPUCB] OK (both d=64 and K*d=448)")


def test_pinterest_dcm():
    print("\n[PinterestDCM] Initializing...")
    num_users, num_items, embed_dim, K, batch_size = 100, 50, 64, 7, 8
    user_train_items = _fake_user_train_items(num_users, num_items)
    interaction_cols = torch.rand(num_items, num_users)

    model = PinterestDCM(
        num_users, num_items, embed_dim=embed_dim, K=K,
        user_train_items=user_train_items, interaction_cols=interaction_cols,
        max_hist_len=15,
    )

    user_row, item_col, user_ids, item_ids = _fake_batch(num_users, num_items, batch_size)

    scores = model.score(user_row, item_col, user_ids, item_ids)
    print("score() output shape:", scores.shape)
    assert scores.shape == (batch_size,)

    scores_k, conditions = model.score_multi(user_row, item_col, user_ids, item_ids)
    print("score_multi() scores shape:", scores_k.shape, "| conditions shape:", conditions.shape)
    assert scores_k.shape == (batch_size, K)
    assert conditions.shape == (batch_size, K, embed_dim)

    # score() should equal max over K of score_multi()
    assert torch.allclose(scores, scores_k.max(dim=1).values, atol=1e-5)

    # Sanity: a user with an empty/never-seen history shouldn't crash
    # (all-invalid mask edge case, see docstring note in dcm.py).
    empty_user = torch.tensor([num_users - 1])
    _ = model.score(
        user_row[:1], item_col[:1], empty_user, item_ids[:1]
    )
    print("[PinterestDCM] OK")


def test_pinterest_dcm_rpucb():
    print("\n[PinterestDCMRPUCB] Initializing...")
    num_users, num_items, embed_dim, K, batch_size = 100, 50, 64, 7, 8
    user_train_items = _fake_user_train_items(num_users, num_items)
    interaction_cols = torch.rand(num_items, num_users)
    counts = torch.randint(1, 30, (num_users,)).long()

    model = PinterestDCMRPUCB(
        num_users, num_items, embed_dim=embed_dim, K=K,
        user_train_items=user_train_items, interaction_cols=interaction_cols,
        max_hist_len=15, user_interaction_counts=counts,
    )

    user_row, item_col, user_ids, item_ids = _fake_batch(num_users, num_items, batch_size)

    scores = model.score(user_row, item_col, user_ids, item_ids)
    assert scores.shape == (batch_size,)

    masks = model.get_condition_masks(user_ids)
    print("Per-(user,k) mask shape:", masks.shape)
    assert masks.shape == (batch_size, K, embed_dim)
    assert (masks >= 0).all() and (masks <= 1).all()

    # Different k-slots for the same user should generally get different
    # masks (independent embedding rows) -- spot check slot 0 vs slot 1.
    assert not torch.allclose(masks[:, 0, :], masks[:, 1, :])
    print("[PinterestDCMRPUCB] OK")


def test_gradients_flow():
    """Cheap check that backward() doesn't error and reaches the towers."""
    print("\n[Gradient flow] Checking backward pass on PinterestDCM...")
    num_users, num_items, embed_dim, K, batch_size = 60, 40, 32, 7, 6
    user_train_items = _fake_user_train_items(num_users, num_items, min_items=6, max_items=12)
    interaction_cols = torch.rand(num_items, num_users)

    model = PinterestDCM(
        num_users, num_items, embed_dim=embed_dim, K=K,
        user_train_items=user_train_items, interaction_cols=interaction_cols,
        max_hist_len=10,
    )
    user_row, item_col, user_ids, item_ids = _fake_batch(num_users, num_items, batch_size)
    scores = model.score(user_row, item_col, user_ids, item_ids)
    loss = scores.sum()
    loss.backward()

    grad_found = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.item_tower.parameters()
    )
    assert grad_found, "No gradient reached item_tower -- routing graph may be detached"
    print("[Gradient flow] OK")


if __name__ == "__main__":
    test_pinterest_base()
    test_pinterest_base_rpucb()
    test_pinterest_dcm()
    test_pinterest_dcm_rpucb()
    test_gradients_flow()
    print("\nAll Pinterest-family smoke tests passed.")