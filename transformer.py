import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from layers.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    Full Transformer — Section 3 of the paper.

    Assembles:
    - Source embedding + positional encoding
    - Target embedding + positional encoding
    - Encoder stack
    - Decoder stack
    - Final linear projection to vocabulary
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        h: int = 8,
        N: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encodings
        self.src_pe = PositionalEncoding(d_model, max_seq_len, dropout)
        self.tgt_pe = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder and Decoder stacks
        self.encoder = Encoder(d_model, h, d_ff, N, dropout)
        self.decoder = Decoder(d_model, h, d_ff, N, dropout)

        # Final projection: d_model -> tgt_vocab_size
        self.projection = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model

        # Initialize parameters — paper section 5.4
        # Xavier uniform keeps variance stable across layers at init
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_pe(self.src_embedding(src) * (self.d_model ** 0.5))
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.tgt_pe(self.tgt_embedding(tgt) * (self.d_model ** 0.5))
        return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.projection(decoder_output)  # (batch, tgt_seq_len, tgt_vocab_size)


def make_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    h: int = 8,
    N: int = 6,
    d_ff: int = 2048,
    dropout: float = 0.1
) -> Transformer:
    """Factory function — clean way to instantiate the model."""
    return Transformer(src_vocab_size, tgt_vocab_size, d_model, h, N, d_ff, dropout=dropout)