# Transformer from Scratch

A complete, paper-accurate implementation of the Transformer architecture
from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

Built in PyTorch with no HuggingFace dependencies. Every component traces
directly to a specific section of the paper.

---

## What's implemented

| Component | Paper Section | File |
|---|---|---|
| Scaled Dot-Product Attention | 3.2.1 | `layers/attention.py` |
| Multi-Head Attention | 3.2.2 | `layers/multihead_attention.py` |
| Positional Encoding | 3.5 | `layers/positional_encoding.py` |
| Feed-Forward Network | 3.3 | `layers/feed_forward.py` |
| Residual + LayerNorm | 3.1 | `layers/sublayer.py` |
| Encoder | 3.1 | `encoder.py` |
| Decoder | 3.1 | `decoder.py` |
| Full Transformer | 3 | `transformer.py` |
| Warmup LR Schedule | 5.3 | `train.py` |
| Label Smoothing | 5.4 | `train.py` |

---

## Results

Trained on Multi30k (De→En), 2 epochs, CPU, d_model=256, N=3 layers.

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 6.17 | 4.69 |
| 2 | 4.37 | 3.79 |

**Sample translations after 2 epochs:**
```
DE: ein mann spielt gitarre .
EN: a man playing the playing the guitar.

DE: eine frau läuft durch den park .
EN: a woman running through the water.

DE: ein hund rennt über das feld .
EN: a dog running through the grass.
```

---

## Project structure

```
transformer-from-scratch/
├── layers/
│   ├── attention.py          # Scaled Dot-Product Attention
│   ├── multihead_attention.py
│   ├── positional_encoding.py
│   ├── feed_forward.py
│   └── sublayer.py           # Residual + LayerNorm
├── encoder.py
├── decoder.py
├── transformer.py
├── train.py                  # Training loop, masks, LR schedule
├── inference.py              # Greedy decoding
├── data.py                   # Custom dataset, vocab, dataloader
├── tests/
│   ├── test_attention.py
│   └── test_model.py
├── notebooks/
│   ├── attention_visualization.ipynb
│   └── positional_encoding.ipynb
└── NOTES.md                  # Implementation decisions explained
```

---

## Setup

```bash
git clone https://github.com/arushiiii18/transformer-from-scratch
cd transformer-from-scratch
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Download Multi30k data:
```bash
python -c "from data import build_dataloaders; build_dataloaders()"
```

---

## Run

**Train:**
```bash
python train.py
```

**Translate:**
```bash
python inference.py
```

**Tests:**
```bash
python -m pytest tests/ -v
```

---

## Key implementation notes

**Why `-inf` masking and not `-1000`?**
`-inf` guarantees exactly `0` after softmax mathematically. Large negatives
leak small non-zero values and lose the semantic meaning of "this position
does not exist."

**Why sinusoidal positional encodings?**
PE(pos+k) can be represented as a linear function of PE(pos), allowing the
model to attend by relative position. Also generalizes to sequence lengths
unseen during training.

**Why feed-forward after attention?**
Attention decides *what* information to gather. The FFN decides *what to do*
with it. Two different jobs — both necessary.

See `NOTES.md` for full implementation reasoning.

---

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017