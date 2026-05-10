import torch

def bpr_loss(pos_scores, neg_scores):
    """
    pos_scores: FloatTensor [batch_size]
    neg_scores: FloatTensor [batch_size, num_negatives]
                OR [batch_size] if num_negatives=1
    Returns: scalar mean BPR loss
    """
    # For each positive, compute loss against each negative
    # loss = -mean( log( sigmoid( pos_score - neg_score ) ) )
    if neg_scores.dim() == 1:
        neg_scores = neg_scores.unsqueeze(1)
    pos_scores = pos_scores.unsqueeze(1).expand_as(neg_scores)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
    return loss

def mask_l1_loss(mask_values):
    """
    mask_values: FloatTensor of any shape containing mask values in [0, 1]
    Returns: scalar mean L1 norm of mask values
    """
    return mask_values.abs().mean()
