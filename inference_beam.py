import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional
from transformer import make_transformer
from data import build_dataloaders, tokenize, SOS_IDX, EOS_IDX, PAD_IDX


@dataclass
class BeamHypothesis:
    """A single beam candidate."""
    token_ids: List[int]
    score: float          # cumulative log probability
    is_complete: bool     # True if EOS has been generated

    def __len__(self):
        return len(self.token_ids)

    @property
    def normalized_score(self, alpha: float = 0.7) -> float:
        """
        Length-normalized score.
        Without this, beam search prefers shorter sequences because
        summing more negative log probs always decreases the total score.
        alpha=0.7 is standard (Wu et al., Google NMT 2016).
        """
        length_penalty = ((5 + len(self.token_ids)) / 6) ** alpha
        return self.score / length_penalty


def beam_search(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    beam_size: int = 4,
    max_len: int = 50,
    alpha: float = 0.7,
    device: torch.device = torch.device('cpu')
) -> List[int]:
    """
    Beam search decoding.

    At each step:
    1. Expand every active beam by all vocab tokens
    2. Score each candidate by cumulative log prob
    3. Keep top beam_size candidates
    4. Repeat until all beams complete or max_len reached

    Args:
        beam_size : number of beams (k=1 reduces to greedy)
        alpha     : length penalty exponent (0=no penalty, 1=full penalty)

    Returns:
        Best token sequence as list of ints (excluding SOS)
    """
    model.eval()

    with torch.no_grad():
        # Encode source once — reused by all beams
        encoder_output = model.encode(src, src_mask)  # (1, src_len, d_model)

        # Expand encoder output for all beams: (beam_size, src_len, d_model)
        encoder_output = encoder_output.expand(beam_size, -1, -1)
        src_mask = src_mask.expand(beam_size, -1, -1, -1)

        # Initialize beams — all start with SOS token
        beams = [BeamHypothesis(
            token_ids=[SOS_IDX],
            score=0.0,
            is_complete=False
        )]
        completed = []

        for step in range(max_len):
            # Filter active (incomplete) beams
            active_beams = [b for b in beams if not b.is_complete]
            if not active_beams:
                break

            n_active = len(active_beams)

            # Build decoder input from all active beams: (n_active, current_len)
            tgt = torch.tensor(
                [b.token_ids for b in active_beams],
                dtype=torch.long, device=device
            )
            tgt_len = tgt.size(1)
            tgt_mask = torch.tril(
                torch.ones(1, 1, tgt_len, tgt_len, device=device)
            ).bool()

            # Use only as many encoder outputs as active beams
            enc_out = encoder_output[:n_active]
            s_mask = src_mask[:n_active]

            # Forward through decoder
            dec_out = model.decode(tgt, enc_out, s_mask, tgt_mask)
            # Take last token's logits: (n_active, vocab_size)
            logits = model.projection(dec_out[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)  # (n_active, vocab_size)

            # Expand each beam by all vocab tokens
            vocab_size = log_probs.size(-1)
            candidates = []

            for i, beam in enumerate(active_beams):
                beam_log_probs = log_probs[i]  # (vocab_size,)

                # Top-k tokens for efficiency (full vocab search is slow)
                topk_log_probs, topk_tokens = beam_log_probs.topk(beam_size * 2)

                for log_prob, token in zip(topk_log_probs.tolist(), topk_tokens.tolist()):
                    new_ids = beam.token_ids + [token]
                    new_score = beam.score + log_prob
                    is_done = (token == EOS_IDX)
                    candidates.append(BeamHypothesis(
                        token_ids=new_ids,
                        score=new_score,
                        is_complete=is_done
                    ))

            # Sort all candidates by length-normalized score
            candidates.sort(
                key=lambda h: h.score / ((5 + len(h)) / 6) ** alpha,
                reverse=True
            )

            # Keep top beam_size, separate completed from active
            beams = []
            for candidate in candidates:
                if len(beams) >= beam_size:
                    break
                if candidate.is_complete:
                    completed.append(candidate)
                else:
                    beams.append(candidate)

            # Early stop: best active beam can't beat best completed
            if completed:
                best_completed = max(
                    completed,
                    key=lambda h: h.score / ((5 + len(h)) / 6) ** alpha
                )
                if beams:
                    best_active = max(
                        beams,
                        key=lambda h: h.score / ((5 + len(h)) / 6) ** alpha
                    )
                    if best_completed.normalized_score >= best_active.normalized_score:
                        break

        # If nothing completed, take best active beam
        if not completed:
            completed = beams if beams else [BeamHypothesis([SOS_IDX], 0.0, False)]

        # Return best completed sequence
        best = max(
            completed,
            key=lambda h: h.score / ((5 + len(h)) / 6) ** alpha
        )

        # Strip SOS token
        return best.token_ids[1:]


def translate_beam(
    sentence: str,
    model,
    src_vocab,
    tgt_vocab,
    device,
    beam_size: int = 4,
    max_len: int = 50
) -> str:
    model.eval()
    tokens = tokenize(sentence)
    src_ids = [SOS_IDX] + src_vocab.encode(tokens) + [EOS_IDX]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    output_ids = beam_search(model, src, src_mask, beam_size, max_len, device=device)

    translated = []
    for idx in output_ids:
        if idx == EOS_IDX:
            break
        if idx in (SOS_IDX, PAD_IDX):
            continue
        translated.append(tgt_vocab.idx2token.get(idx, '<unk>'))

    return ' '.join(translated)


if __name__ == "__main__":
    device = torch.device('cpu')
    _, _, src_vocab, tgt_vocab = build_dataloaders(batch_size=32)

    model = make_transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256, h=8, N=3, d_ff=512, dropout=0.0
    ).to(device)

    model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=device))
    model.eval()
    print('Model loaded.\n')

    test_sentences = [
        "ein mann spielt gitarre .",
        "eine frau läuft durch den park .",
        "zwei kinder spielen im garten .",
        "ein hund rennt über das feld .",
    ]

    print(f"{'German':<45} {'Greedy':<35} {'Beam (k=4)'}")
    print("-" * 115)

    # Import greedy for comparison
    from inference import translate

    for sentence in test_sentences:
        greedy = translate(sentence, model, src_vocab, tgt_vocab, device)
        beam = translate_beam(sentence, model, src_vocab, tgt_vocab, device, beam_size=4)
        print(f"{sentence:<45} {greedy:<35} {beam}")