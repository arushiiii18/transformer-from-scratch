import torch
import torch.nn as nn
from layers.multihead_attention import MultiHeadAttention
from layers.feed_forward import FeedForward
from layers.sublayer import SubLayerConnection


class DecoderBlock(nn.Module):
    """
    Single Decoder Layer — Section 3.1 of the paper.

    Each layer has THREE sublayers:
    1. Masked Multi-Head Self-Attention (causal — can't see future tokens)
    2. Encoder-Decoder Cross-Attention (Q from decoder, K/V from encoder)
    3. Position-wise Feed-Forward Network

    Why masked self-attention in decoder?
    During training we feed the full target sequence at once (for efficiency).
    But the model must not cheat by seeing future tokens — it must predict
    token t using only tokens 0..t-1. The causal mask enforces this.

    Why cross-attention?
    This is where the decoder queries the encoder's output —
    it learns what source information is relevant for each target position.
    """

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, h, dropout)
        self.cross_attention = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SubLayerConnection(d_model, dropout)
        self.sublayer2 = SubLayerConnection(d_model, dropout)
        self.sublayer3 = SubLayerConnection(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # 1. Masked self-attention — decoder attends to itself, causally
        x = self.sublayer1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        # 2. Cross-attention — Q from decoder, K and V from encoder
        x = self.sublayer2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        # 3. Feed-forward
        x = self.sublayer3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """
    Full Decoder: stack of N DecoderBlocks.
    """

    def __init__(self, d_model: int, h: int, d_ff: int, N: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)