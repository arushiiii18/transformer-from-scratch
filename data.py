import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import List


# ── Special token indices ──────────────────────────────────────────────────────
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIALS = ['<pad>', '<sos>', '<eos>', '<unk>']


# ── Vocabulary ─────────────────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = {}
        for i, s in enumerate(SPECIALS):
            self.token2idx[s] = i
            self.idx2token[i] = s

    def build(self, sentences: List[List[str]]):
        counter = Counter()
        for tokens in sentences:
            counter.update(tokens)
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2idx.get(t, UNK_IDX) for t in tokens]

    def __len__(self):
        return len(self.token2idx)


def tokenize(sentence: str) -> List[str]:
    """Basic whitespace + lowercase tokenizer."""
    return sentence.lower().strip().split()


def read_file(path: str) -> List[str]:
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


# ── Dataset ────────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_len: int = 100
    ):
        assert len(src_sentences) == len(tgt_sentences)
        self.pairs = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_tokens = tokenize(src)
            tgt_tokens = tokenize(tgt)
            # Skip overly long sentences — stabilizes training
            if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
                continue
            src_ids = [SOS_IDX] + src_vocab.encode(src_tokens) + [EOS_IDX]
            tgt_ids = [SOS_IDX] + tgt_vocab.encode(tgt_tokens) + [EOS_IDX]
            self.pairs.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch


# ── Entry point ────────────────────────────────────────────────────────────────

def build_dataloaders(batch_size: int = 32):
    # Read raw files
    train_src = read_file('data_files/train.de')
    train_tgt = read_file('data_files/train.en')
    val_src   = read_file('data_files/val.de')
    val_tgt   = read_file('data_files/val.en')

    # Build vocabularies from training data only — never from val
    src_vocab = Vocabulary(min_freq=2)
    tgt_vocab = Vocabulary(min_freq=2)
    src_vocab.build([tokenize(s) for s in train_src])
    tgt_vocab.build([tokenize(s) for s in train_tgt])

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_dataset   = TranslationDataset(val_src,   val_tgt,   src_vocab, tgt_vocab)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, src_vocab, tgt_vocab