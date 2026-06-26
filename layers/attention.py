import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention — Section 3.2.1 of the paper.

    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Why scale by sqrt(d_k)?
    For large d_k, dot products grow large in magnitude, pushing softmax
    into regions with extremely small gradients. Scaling counteracts this.

    Args:
        query : (batch, heads, seq_len, d_k)
        key   : (batch, heads, seq_len, d_k)
        value : (batch, heads, seq_len, d_k)
        mask  : optional (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)

    Returns:
        output  : (batch, heads, seq_len, d_k)
        weights : (batch, heads, seq_len, seq_len)  -- for visualization later
    """
    d_k = query.size(-1)

    # QK^T — similarity scores between every query and every key
    # (batch, heads, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask before softmax
    # Masked positions get -inf so they become 0 after softmax
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax over last dim — each query attends over all key positions
    weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(weights, value)

    return output, weights