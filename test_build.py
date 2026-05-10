import torch
from src.models.deepcf_static_mask_attn import DeepCFStaticMaskAttn

def test_model():
    print("Initializing model...")
    num_users = 100
    num_items = 50
    embed_dim = 64
    batch_size = 8

    model = DeepCFStaticMaskAttn(num_users=num_users, num_items=num_items, embed_dim=embed_dim)

    # user_row is expected to be [batch_size, num_items]
    user_row = torch.rand(batch_size, num_items)
    # item_col is expected to be [batch_size, num_users]
    item_col = torch.rand(batch_size, num_users)
    # user_ids is expected to be [batch_size]
    user_ids = torch.randint(0, num_users, (batch_size,))

    print("Running forward pass...")
    scores = model.score(user_row=user_row, item_col=item_col, user_ids=user_ids)

    print("Forward pass successful!")
    print("Output shape:", scores.shape)
    assert scores.shape == (batch_size,)

if __name__ == "__main__":
    test_model()
