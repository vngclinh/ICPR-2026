"""Post-processing utilities for OCR decoding."""
import math
from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import torch


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str]
) -> List[Tuple[str, float]]:
    """CTC decode predictions with confidence scores using greedy decoding.
    
    Args:
        preds: Log-softmax predictions of shape [batch_size, time_steps, num_classes].
        idx2char: Index to character mapping.
    
    Returns:
        List of (predicted_string, confidence_score) tuples.
    """
    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)
    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()
    
    batch_size, time_steps = indices_np.shape
    results: List[Tuple[str, float]] = []
    
    for batch_idx in range(batch_size):
        path = indices_np[batch_idx]
        probs_b = max_probs_np[batch_idx]
        
        # Group consecutive identical characters and filter blanks
        # groupby returns (key, group_iterator) pairs
        pred_chars = []
        confidences = []
        time_idx = 0
        
        for char_idx, group in groupby(path):
            group_list = list(group)
            group_size = len(group_list)
            
            if char_idx != 0:  # Skip blank
                pred_chars.append(idx2char.get(char_idx, ''))
                # Get maximum probability from this group
                group_probs = probs_b[time_idx:time_idx + group_size]
                confidences.append(float(np.max(group_probs)))
            
            time_idx += group_size
        
        pred_str = "".join(pred_chars)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))

    return results


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    NEG_INF = -1e30
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    hi = max(a, b)
    return hi + math.log1p(math.exp(min(a, b) - hi))


def ctc_beam_search(
    log_probs: torch.Tensor,
    idx2char: Dict[int, str],
    beam_width: int = 20,
) -> Tuple[str, float]:
    """CTC prefix beam search.

    Args:
        log_probs: [T, C] log-softmax probabilities (blank = index 0).
        idx2char: mapping from class index to character string.
        beam_width: number of hypotheses to keep per step.

    Returns:
        (best_string, score) where score = log P(best_string | log_probs).
    """
    NEG_INF = -1e30
    T, C = log_probs.shape
    lp = log_probs.cpu().numpy()

    # Beam: dict{ prefix_str -> [log_pb, log_pnb] }
    # log_pb  = log P(path ending with blank   that decodes to prefix)
    # log_pnb = log P(path ending with non-blank that decodes to prefix)
    beam: Dict[str, List[float]] = {"": [0.0, NEG_INF]}

    for t in range(T):
        new_beam: Dict[str, List[float]] = {}

        for prefix, (log_pb, log_pnb) in beam.items():
            last = prefix[-1] if prefix else None
            log_p_tot = _log_add(log_pb, log_pnb)

            # 1. Extend with blank — prefix unchanged
            new_log_pb = lp[t, 0] + log_p_tot
            if prefix not in new_beam:
                new_beam[prefix] = [new_log_pb, NEG_INF]
            else:
                new_beam[prefix][0] = _log_add(new_beam[prefix][0], new_log_pb)

            # 2. Extend with each non-blank character
            for c in range(1, C):
                ch = idx2char.get(c)
                if ch is None:
                    continue
                new_prefix = prefix + ch

                if ch == last:
                    # Same char as last non-blank: only blank-ending paths can restart it
                    new_log_pnb = lp[t, c] + log_pb
                else:
                    new_log_pnb = lp[t, c] + log_p_tot

                if new_prefix not in new_beam:
                    new_beam[new_prefix] = [NEG_INF, new_log_pnb]
                else:
                    new_beam[new_prefix][1] = _log_add(new_beam[new_prefix][1], new_log_pnb)

        # Prune to beam_width
        beam = dict(
            sorted(
                new_beam.items(),
                key=lambda kv: -_log_add(kv[1][0], kv[1][1]),
            )[:beam_width]
        )

    # Pick best hypothesis
    best = max(beam.items(), key=lambda kv: _log_add(kv[1][0], kv[1][1]))
    score = _log_add(best[1][0], best[1][1])
    return best[0], float(score)


def decode_beam_search_batch(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    beam_width: int = 20,
) -> List[Tuple[str, float]]:
    """Beam-search decode a batch of CTC log-probs.

    Args:
        preds: [B, T, C] log-softmax predictions.
        idx2char: index-to-char mapping.
        beam_width: beam size.

    Returns:
        List of (predicted_string, confidence) tuples.
    """
    results = []
    for b in range(preds.size(0)):
        text, score = ctc_beam_search(preds[b], idx2char, beam_width)
        # Convert log-score to a pseudo-confidence in [0, 1]
        conf = float(math.exp(max(score / max(len(text), 1), -20.0)))
        results.append((text, conf))
    return results
