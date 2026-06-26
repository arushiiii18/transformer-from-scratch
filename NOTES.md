# Implementation Notes — Transformer from Scratch

## Why I built this
After reading "Attention Is All You Need" (Vaswani et al., 2017), I wanted to
verify my understanding by implementing every component from scratch in PyTorch —
no HuggingFace, no shortcuts.

---

## Key design decisions

### Scaled Dot-Product Attention
- Scaling by √d_k prevents softmax saturation for large d_k values
- Used `-inf` masking (not large negatives) — guarantees exact 0 after softmax
- Attention weights stored on the module for visualization

### Multi-Head Attention
- Single W_q, W_k, W_v projection then split — more efficient than h separate linears
- No bias on projection matrices — follows paper exactly
- Q, K, V source differs by context:
  - Encoder self-attention: Q=K=V=encoder output
  - Decoder self-attention: Q=K=V=decoder output (causally masked)
  - Cross-attention: Q=decoder, K=V=encoder output

### Positional Encoding
- Sinusoidal, not learned — generalizes to unseen sequence lengths
- Computed in log space for numerical stability
- Added (not concatenated) to preserve d_model dimensionality
- Registered as buffer — saved in checkpoint but not a trainable parameter

### Residual Connections + LayerNorm
- Post-norm (paper exact): LayerNorm(x + Sublayer(x))
- LayerNorm over BatchNorm — sequence-length agnostic, batch-size independent
- SubLayerConnection is generic — accepts any sublayer as a callable

### Feed-Forward Network
- d_ff = 2048 (4x d_model) as in paper, scaled to 512 for this experiment
- Independent per position — attention handles mixing, FFN handles transformation
- ReLU activation (paper exact — modern models use GELU)

### Masking
- Padding mask: (src != pad_idx) — ignores padding tokens in attention
- Causal mask: torch.tril — prevents decoder from seeing future tokens
- Both ANDed together for decoder self-attention

### Training
- Adam with β1=0.9, β2=0.98, ε=1e-9 — paper exact values
- Warmup schedule: lr increases linearly for 4000 steps, then decays
- Label smoothing ε=0.1 — prevents overconfident predictions
- Gradient clipping at 1.0 — prevents exploding gradients
- Teacher forcing during training — feed ground truth as decoder input

---

## What I'd do differently with more compute
- Train for 20+ epochs (only 2 completed on CPU)
- Use beam search decoding instead of greedy
- BPE tokenization instead of whitespace split
- Larger model: d_model=512, N=6 as in paper
- Pre-norm instead of post-norm for training stability

---

## Results (2 epochs, CPU, d_model=256, N=3)
- Train Loss: 4.37
- Val Loss: 3.79
- Sample: "ein mann spielt gitarre ." → "a man playing the playing the guitar."

---

## References
- Vaswani et al., "Attention Is All You Need", 2017
- The Annotated Transformer (Harvard NLP)