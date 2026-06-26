import torch
import pytest
from layers.attention import scaled_dot_product_attention
from layers.multihead_attention import MultiHeadAttention


class TestScaledDotProductAttention:

    def test_output_shape(self):
        Q = torch.randn(2, 8, 10, 64)
        K = torch.randn(2, 8, 10, 64)
        V = torch.randn(2, 8, 10, 64)
        out, weights = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (2, 8, 10, 64)
        assert weights.shape == (2, 8, 10, 10)

    def test_attention_weights_sum_to_one(self):
        Q = torch.randn(2, 8, 10, 64)
        K = torch.randn(2, 8, 10, 64)
        V = torch.randn(2, 8, 10, 64)
        _, weights = scaled_dot_product_attention(Q, K, V)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_causal_mask_blocks_future(self):
        """Masked positions must have exactly 0 attention weight."""
        Q = torch.randn(1, 1, 4, 64)
        K = torch.randn(1, 1, 4, 64)
        V = torch.randn(1, 1, 4, 64)
        # Causal mask: lower triangular — position i can only see 0..i
        mask = torch.tril(torch.ones(1, 1, 4, 4))
        _, weights = scaled_dot_product_attention(Q, K, V, mask)
        # Upper triangle must be zero
        upper = weights[0, 0]
        for i in range(4):
            for j in range(i + 1, 4):
                assert upper[i, j].item() == pytest.approx(0.0, abs=1e-6), \
                    f"Position ({i},{j}) should be masked but got {upper[i,j].item()}"

    def test_no_mask_no_zeros_in_weights(self):
        """Without mask, all weights should be positive."""
        Q = torch.randn(2, 4, 5, 32)
        K = torch.randn(2, 4, 5, 32)
        V = torch.randn(2, 4, 5, 32)
        _, weights = scaled_dot_product_attention(Q, K, V)
        assert (weights > 0).all()


class TestMultiHeadAttention:

    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=512, h=8)
        x = torch.randn(2, 10, 512)
        out = mha(x, x, x)
        assert out.shape == (2, 10, 512)

    def test_different_qkv_sources(self):
        """Cross-attention: Q and K/V from different sources."""
        mha = MultiHeadAttention(d_model=512, h=8)
        q = torch.randn(2, 7, 512)   # decoder
        kv = torch.randn(2, 10, 512) # encoder
        out = mha(q, kv, kv)
        assert out.shape == (2, 7, 512)

    def test_d_model_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=513, h=8)