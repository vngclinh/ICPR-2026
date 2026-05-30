"""Format-constrained decoding for Brazilian license plates.

Two layouts are valid (both exactly 7 characters):

* **Brazil-old:**  ``L L L  N N N N``      pattern ``[A-Z]{3}[0-9]{4}``
* **Mercosur:**    ``L L L  N  L  N N``    pattern ``[A-Z]{3}[0-9][A-Z][0-9]{2}``

We project the model's CTC log-probs onto whichever pattern best fits, by
masking out illegal character classes per position and running constrained
beam search over the 7 fixed slots. Among the two pattern hypotheses we
keep the higher-scoring one.

This is invoked **after** ensemble averaging — input is a single
``[T, C]`` log-prob tensor (CTC blank at index 0).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

BRAZIL_OLD_PATTERN: Tuple[str, ...] = ("L", "L", "L", "N", "N", "N", "N")
MERCOSUR_PATTERN: Tuple[str, ...] = ("L", "L", "L", "N", "L", "N", "N")

ALL_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "brazil_old": BRAZIL_OLD_PATTERN,
    "mercosur": MERCOSUR_PATTERN,
}


def _allowed_indices(slot: str, char2idx: Dict[str, int]) -> List[int]:
    chars = LETTERS if slot == "L" else DIGITS
    return [char2idx[c] for c in chars if c in char2idx]


def _per_slot_score(log_probs: torch.Tensor, slot_indices: Iterable[int]) -> Tuple[int, float]:
    """For one CTC timestep, return (best_class_idx, log_score) over allowed indices."""
    idxs = list(slot_indices)
    vals = log_probs[idxs]
    best = int(vals.argmax().item())
    return idxs[best], float(vals[best].item())


def _ctc_greedy_with_timesteps(
    log_probs: torch.Tensor, blank: int = 0
) -> Tuple[List[int], List[int]]:
    """CTC greedy decode that also returns the timestep each char was emitted at.

    Returns ``(class_indices, timesteps)`` of equal length.
    """
    path = log_probs.argmax(dim=-1).tolist()        # [T]
    classes: List[int] = []
    timesteps: List[int] = []
    prev = blank
    for t, c in enumerate(path):
        if c != blank and c != prev:
            classes.append(c)
            timesteps.append(t)
        prev = c
    return classes, timesteps


def _matches_pattern(chars: List[int], pattern: Tuple[str, ...], idx2char: Dict[int, str]) -> bool:
    if len(chars) != len(pattern):
        return False
    for c, slot in zip(chars, pattern):
        ch = idx2char.get(c, "")
        if slot == "L" and not ch.isalpha():
            return False
        if slot == "N" and not ch.isdigit():
            return False
    return True


def format_constrained_decode(
    log_probs: torch.Tensor,
    char2idx: Dict[str, int],
    idx2char: Dict[int, str],
    target_length: int = 7,
) -> Tuple[str, float]:
    """Decode a plate; only project to a valid pattern when greedy violates one.

    Strategy:

    1. CTC-greedy-decode the output with per-char timesteps.
    2. If greedy already matches Brazil-old or Mercosur, return greedy
       unchanged (it is by definition optimal under CTC).
    3. Otherwise, for each candidate pattern, swap only the offending
       positions to the highest-prob valid class in a small timestep window
       (±1) around where the violating char was emitted. Pick the higher-
       scoring projection.
    4. If greedy has the wrong length, fall back to a length-fix: keep the
       most-confident ``target_length`` emissions (truncate) or insert from
       the next-best class at gaps (pad).

    Args:
        log_probs: [T, C] log-softmax. Index 0 is the CTC blank.
        char2idx: char -> class index (without blank → 0).
        idx2char: inverse.
        target_length: plate length (7 for Brazilian plates).

    Returns:
        (predicted_string, log_score)
    """
    t = log_probs.shape[0]
    classes, timesteps = _ctc_greedy_with_timesteps(log_probs, blank=0)

    # Greedy already matches a valid pattern → return it as-is.
    if len(classes) == target_length:
        for pattern in ALL_PATTERNS.values():
            if _matches_pattern(classes, pattern, idx2char):
                text = "".join(idx2char.get(c, "") for c in classes)
                score = sum(float(log_probs[ts, c].item()) for c, ts in zip(classes, timesteps))
                return text, score

    # Length fix: truncate to top-confidence or pad placeholders.
    if len(classes) == target_length:
        aligned = list(zip(classes, timesteps))
    elif len(classes) > target_length:
        scores = [float(log_probs[ts, c].item()) for c, ts in zip(classes, timesteps)]
        keep = sorted(sorted(range(len(classes)), key=lambda i: -scores[i])[:target_length])
        aligned = [(classes[i], timesteps[i]) for i in keep]
    else:
        aligned = list(zip(classes, timesteps))
        last_t = timesteps[-1] if timesteps else t - 1
        while len(aligned) < target_length:
            aligned.append((0, last_t))     # blank → forces re-pick

    best_total = -math.inf
    best_string = ""
    for pattern in ALL_PATTERNS.values():
        chars: List[str] = []
        total = 0.0
        for slot_idx, slot in enumerate(pattern):
            allowed = _allowed_indices(slot, char2idx)
            if not allowed:
                continue
            cls, ts = aligned[slot_idx]
            if cls in allowed:
                chars.append(idx2char.get(cls, ""))
                total += float(log_probs[ts, cls].item())
            else:
                lo = max(0, ts - 1)
                hi = min(t, ts + 2)
                neigh = log_probs[lo:hi][:, allowed]
                best_per_t, best_class = neigh.max(dim=1)
                best_t = int(best_per_t.argmax().item())
                best_idx = allowed[int(best_class[best_t].item())]
                chars.append(idx2char.get(best_idx, ""))
                total += float(best_per_t[best_t].item())
        if total > best_total:
            best_total = total
            best_string = "".join(chars)

    return best_string, best_total
