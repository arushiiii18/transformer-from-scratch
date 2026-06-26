import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding — Section 3.5 of the paper.

    Since we removed recurrence, the model has no idea about word order.
    We inject position information by adding a deterministic signal to embeddings.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why sinusoidal? Two reasons:
    1. The model can learn to attend by relative positions because
       PE(pos+k) can be represented as a linear function of PE(pos).
    2. It generalizes to sequence lengths unseen during training.

    Why add instead of concatenate?
    Keeps d_model unchanged. The model learns to disentangle
    position from content through training.
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the positional encoding matrix once, reuse forever
        pe = torch.zeros(max_seq_len, d_model)  # (max_seq_len, d_model)

        position = torch.arange(0, max_seq_len).unsqueeze(1).float()  # (max_seq_len, 1)

        # Compute the div_term in log space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model) — batch dim

        # Register as buffer: saved in state_dict but not a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding, same shape
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)