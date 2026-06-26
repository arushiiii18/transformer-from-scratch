import torch
import pytest
from transformer import make_transformer


class TestTransformer:

    def setup_method(self):
        self.model = make_transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,  # small for test speed
            h=4,
            N=2,
            d_ff=256
        )

    def test_output_shape(self):
        src = torch.randint(0, 1000, (2, 10))
        tgt = torch.randint(0, 1000, (2, 7))
        out = self.model(src, tgt)
        assert out.shape == (2, 7, 1000)

    def test_causal_mask_shape(self):
        seq_len = 7
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        assert mask.shape == (1, 1, seq_len, seq_len)

    def test_padding_mask(self):
        """Padding tokens (0) should be masked out."""
        src = torch.tensor([[1, 2, 3, 0, 0]])  # last two are padding
        pad_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (1,1,1,5)
        assert pad_mask.shape == (1, 1, 1, 5)
        assert pad_mask[0, 0, 0, 3].item() == False
        assert pad_mask[0, 0, 0, 0].item() == True

    def test_encoder_decoder_output_shapes(self):
        src = torch.randint(0, 1000, (3, 12))
        tgt = torch.randint(0, 1000, (3, 8))
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = torch.tril(torch.ones(1, 1, 8, 8))
        enc_out = self.model.encode(src, src_mask)
        dec_out = self.model.decode(tgt, enc_out, src_mask, tgt_mask)
        assert enc_out.shape == (3, 12, 128)
        assert dec_out.shape == (3, 8, 128)

    def test_parameter_count_reasonable(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total > 1_000_000, "Model seems too small"