import torch
import torch.nn as nn


class SubLayerConnection(nn.Module):
    """
    Residual connection + Layer Normalization — Section 3.1 of the paper.

    output = LayerNorm(x + Sublayer(x))

    Note: Paper applies LayerNorm AFTER the residual (post-norm).
    Modern implementations often use pre-norm (LayerNorm before sublayer)
    which trains more stably. We follow the paper exactly here.

    Why residual connections?
    Without them, gradients vanish through deep networks.
    The skip connection gives gradients a highway back to early layers.

    Why LayerNorm and not BatchNorm?
    BatchNorm normalizes across the batch dimension — unstable for
    variable-length sequences and small batches. LayerNorm normalizes
    across the feature dimension independently per token. Sequence-length
    agnostic and batch-size independent.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        """
        Apply residual connection to any sublayer with same d_model I/O.
        The sublayer is passed as a function — keeps this class generic.
        """
        return self.norm(x + self.dropout(sublayer(x)))