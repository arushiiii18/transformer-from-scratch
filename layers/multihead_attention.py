import torch
import torch.nn as nn
from layers.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention — Section 3.2.2 of the paper.

    Instead of one attention function with d_model dimensions,
    we project Q, K, V h times with different learned projections
    to d_k = d_model / h dimensions each, run attention in parallel,
    then concatenate and project back.

    Why? Each head can specialize in different relationships
    (syntactic, semantic, positional). One big attention head
    averages all of that into a single representation.

    Paper: d_model=512, h=8, d_k=d_v=64
    """

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # dimension per head

        # Single linear layers that project for ALL heads at once
        # then we split — more efficient than h separate linear layers
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # store for visualization later

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split d_model into h heads.
        (batch, seq_len, d_model) -> (batch, h, seq_len, d_k)
        """
        batch, seq_len, d_model = x.size()
        x = x.view(batch, seq_len, self.h, self.d_k)
        return x.transpose(1, 2)  # (batch, h, seq_len, d_k)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_heads.
        (batch, h, seq_len, d_k) -> (batch, seq_len, d_model)
        """
        batch, h, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query : (batch, seq_len, d_model)
            key   : (batch, seq_len, d_model)
            value : (batch, seq_len, d_model)
            mask  : optional

        Note: Q, K, V come from different sources depending on context:
            - Encoder self-attention:     Q=K=V=encoder output
            - Decoder self-attention:     Q=K=V=decoder output (masked)
            - Encoder-Decoder attention:  Q=decoder, K=V=encoder output
        """
        # Project and split into heads
        Q = self.split_heads(self.W_q(query))  # (batch, h, seq_len, d_k)
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # Attention across all heads simultaneously
        x, self.attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and project
        x = self.combine_heads(x)       # (batch, seq_len, d_model)
        x = self.W_o(x)                 # final projection

        return x