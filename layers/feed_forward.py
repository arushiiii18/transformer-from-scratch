import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network — Section 3.3 of the paper.

    FFN(x) = max(0, xW1 + b1)W2 + b2

    Applied identically and independently to each position.
    Paper: d_model=512, d_ff=2048 (4x expansion).

    Why does this exist after attention already mixed context?
    Attention decides WHAT information to gather from where.
    FFN decides WHAT TO DO with that information — it's where
    the model stores factual knowledge and applies transformations.
    Two different jobs. Both necessary.

    Why independent per position?
    Because after attention, each token already has the context it needs.
    The FFN just processes each enriched token representation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand -> ReLU -> Dropout -> Contract
        return self.linear2(self.dropout(self.relu(self.linear1(x))))