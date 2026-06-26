import torch
import torch.nn as nn
from layers.multihead_attention import MultiHeadAttention
from layers.feed_forward import FeedForward
from layers.sublayer import SubLayerConnection


class EncoderBlock(nn.Module):
    """
    Single Encoder Layer — Section 3.1 of the paper.

    Each layer has two sublayers:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network

    Each sublayer wrapped with: output = LayerNorm(x + Dropout(Sublayer(x)))
    Paper uses N=6 identical encoder layers stacked.
    """

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SubLayerConnection(d_model, dropout)
        self.sublayer2 = SubLayerConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention: Q=K=V=x (every token attends to every other token)
        x = self.sublayer1(x, lambda x: self.self_attention(x, x, x, src_mask))
        # Feed-forward: applied independently per position
        x = self.sublayer2(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """
    Full Encoder: stack of N EncoderBlocks.
    """

    def __init__(self, d_model: int, h: int, d_ff: int, N: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)  # final norm after all layers

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)