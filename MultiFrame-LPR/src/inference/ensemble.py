"""Logits ensemble for the 4 LPR variants.

Three mechanisms (as in Section 6.1 of the design doc):
1. **Weighted log-probability averaging** — when variants share the same
   ``[T, C]`` time grid we average ``log_probs`` (numerically equivalent to
   geometric mean of probabilities).
2. **Voting** — each variant decodes a string; we pick the majority string.
3. **Single-model fallback** — when models disagree heavily and the chosen
   ensemble winner has low confidence, fall back to the predesignated
   strongest model.

The three are composed by ``ensemble_predictions``: average + format-decode,
and only when the chosen string's confidence is below ``fallback_threshold``
or voting disagrees, we use the fallback.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.inference.format_decode import format_constrained_decode


def _align_time_axis(logits: torch.Tensor, target_t: int) -> torch.Tensor:
    """Resample [T, C] log-probs to length ``target_t`` along time.

    Different variants may produce slightly different sequence lengths (e.g.
    V4 has a fixed ``num_queries``). We linearly interpolate the *probability*
    domain (then re-log) to align them before averaging.
    """
    t, c = logits.shape
    if t == target_t:
        return logits
    probs = logits.exp().t().unsqueeze(0)              # [1, C, T]
    probs_resized = F.interpolate(probs, size=target_t, mode="linear", align_corners=False)
    probs_resized = probs_resized.squeeze(0).t()        # [target_t, C]
    probs_resized = probs_resized.clamp_min(1e-12)
    return probs_resized.log()


def weighted_logprob_average(
    log_probs_list: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Weighted average of a list of ``[T_i, C]`` log-prob tensors.

    All tensors must share the channel dim ``C``. Time is aligned to the
    *maximum* T across inputs by linear interpolation.
    """
    if not log_probs_list:
        raise ValueError("Expected at least one log-prob tensor")

    if weights is None:
        weights = [1.0] * len(log_probs_list)
    if len(weights) != len(log_probs_list):
        raise ValueError("weights length must match log_probs_list length")
    weights_t = torch.tensor(weights, dtype=log_probs_list[0].dtype)
    weights_t = weights_t / weights_t.sum().clamp_min(1e-12)

    target_t = max(lp.shape[0] for lp in log_probs_list)
    aligned = [_align_time_axis(lp, target_t) for lp in log_probs_list]
    stacked = torch.stack(aligned, dim=0)               # [N, T, C]
    w = weights_t.view(-1, 1, 1).to(stacked.device, stacked.dtype)
    return (stacked * w).sum(dim=0)


def voting_predictions(strings: Sequence[str]) -> Tuple[str, int]:
    """Return the most common string and its vote count."""
    if not strings:
        return "", 0
    counts = Counter(strings)
    winner, votes = counts.most_common(1)[0]
    return winner, votes


def fallback_predictions(
    per_variant_strings: Sequence[str],
    per_variant_scores: Sequence[float],
    fallback_index: int,
) -> Tuple[str, float]:
    """Return (string, score) for the strongest predesignated variant."""
    return per_variant_strings[fallback_index], per_variant_scores[fallback_index]


def ensemble_predictions(
    per_variant_log_probs: Sequence[torch.Tensor],   # each [T_i, C]
    char2idx: Dict[str, int],
    idx2char: Dict[int, str],
    weights: Optional[Sequence[float]] = None,
    fallback_index: int = 0,
    fallback_threshold: float = -1.5,
    target_length: int = 7,
) -> Tuple[str, float, Dict[str, object]]:
    """Run the full Section-6 ensemble pipeline.

    Returns:
        (final_string, log_score, debug) where ``debug`` reports the chosen
        path so the caller can log how often fallback fires.
    """
    # Per-variant constrained decodes (used for voting and fallback).
    per_variant_decoded: List[Tuple[str, float]] = [
        format_constrained_decode(lp, char2idx, idx2char, target_length=target_length)
        for lp in per_variant_log_probs
    ]
    per_variant_strings = [s for s, _ in per_variant_decoded]
    per_variant_scores = [s for _, s in per_variant_decoded]

    # 1. Weighted log-prob average.
    avg = weighted_logprob_average(per_variant_log_probs, weights=weights)
    avg_string, avg_score = format_constrained_decode(
        avg, char2idx, idx2char, target_length=target_length
    )

    # 2. Voting on the per-variant decodes.
    vote_string, vote_count = voting_predictions(per_variant_strings)

    # 3. Decide which to keep.
    debug: Dict[str, object] = {
        "avg_string": avg_string,
        "avg_score": avg_score,
        "vote_string": vote_string,
        "vote_count": vote_count,
        "per_variant": per_variant_decoded,
        "path": "average",
    }

    # If voting majority disagrees with the average AND average is weak, fall
    # back to the strongest predesignated model.
    per_token_score = avg_score / max(target_length, 1)
    if (
        vote_string != avg_string
        and vote_count >= (len(per_variant_strings) // 2 + 1)
    ):
        debug["path"] = "vote"
        return vote_string, sum(per_variant_scores) / len(per_variant_scores), debug

    if per_token_score < fallback_threshold:
        fb_string, fb_score = fallback_predictions(
            per_variant_strings, per_variant_scores, fallback_index
        )
        debug["path"] = "fallback"
        return fb_string, fb_score, debug

    return avg_string, avg_score, debug
