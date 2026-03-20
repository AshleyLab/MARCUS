"""
Per-Category Evaluation Breakdown

Computes evaluation metrics stratified by clinical category (e.g., arrhythmia,
ventricular function, valve disease) as reported in the MARCUS paper (Figure 3,
Supplementary Figure S5).

Each prediction record is expected to carry a ``category`` field (free-form
string assigned during dataset curation) and a ``modality`` field (e.g.,
"ECG", "Echo", "CMR").  Both MCQ and VQA tasks are supported.

Bootstrap CIs use the percentile method at 95% coverage.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "CategoryStats",
    "compute_per_category_stats",
    "format_category_table",
    "save_category_stats",
]


@dataclass
class CategoryStats:
    """Aggregated statistics for a single (category, modality) stratum."""

    category: str
    modality: str
    n: int
    # MCQ stats
    accuracy: float | None
    accuracy_ci_lower: float | None
    accuracy_ci_upper: float | None
    # VQA stats
    mean_likert: float | None
    median_likert: float | None
    likert_ci_lower: float | None
    likert_ci_upper: float | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(
    values: list[float],
    n_bootstrap: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap percentile CI for the mean."""
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(math.floor((alpha / 2) * n_bootstrap))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    hi_idx = min(hi_idx, n_bootstrap - 1)
    return means[lo_idx], means[hi_idx]


def _safe_median(values: list[float]) -> float:
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_v[mid])
    return (sorted_v[mid - 1] + sorted_v[mid]) / 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_per_category_stats(
    predictions: list[dict[str, Any]],
    task: str = "mcq",
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, CategoryStats]:
    """Compute per-category statistics from a list of prediction records.

    Each record must contain:

    - ``category`` (str): clinical category label.
    - ``modality`` (str): imaging/signal modality label.

    For MCQ tasks (``task="mcq"``), each record must also contain one of:

    - ``correct`` (bool): whether the prediction was correct, **or**
    - ``eval_label`` (str): ``"Correct"`` / ``"Incorrect"`` / ``"Excluded"``.

    For VQA tasks (``task="vqa"``), each record must contain:

    - ``likert_score`` (int 1–5): judge-assigned Likert score.

    Records missing the required scoring field are silently skipped for that
    category/modality bucket but are counted in ``n``.

    Args:
        predictions: List of prediction dicts (see above).
        task: ``"mcq"`` or ``"vqa"``.
        n_bootstrap: Number of bootstrap resamples for CIs.
        seed: RNG seed for reproducible CIs.

    Returns:
        Dict keyed by ``"{category}|{modality}"`` → :class:`CategoryStats`.
    """
    if task not in ("mcq", "vqa"):
        raise ValueError(f"task must be 'mcq' or 'vqa', got {task!r}")

    # Group records by (category, modality)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in predictions:
        cat = str(rec.get("category", "Unknown"))
        mod = str(rec.get("modality", "Unknown"))
        buckets[(cat, mod)].append(rec)

    results: dict[str, CategoryStats] = {}

    for (cat, mod), recs in sorted(buckets.items()):
        n = len(recs)
        key = f"{cat}|{mod}"

        if task == "mcq":
            # Collect correct/incorrect flags; exclude "Excluded" rows
            correct_flags: list[float] = []
            for rec in recs:
                if "correct" in rec:
                    val = rec["correct"]
                    if isinstance(val, bool):
                        correct_flags.append(1.0 if val else 0.0)
                    elif isinstance(val, (int, float)):
                        correct_flags.append(float(val))
                elif "eval_label" in rec:
                    label = str(rec["eval_label"])
                    if label == "Correct":
                        correct_flags.append(1.0)
                    elif label == "Incorrect":
                        correct_flags.append(0.0)
                    # "Excluded" is skipped

            if correct_flags:
                accuracy = sum(correct_flags) / len(correct_flags)
                if len(correct_flags) >= 2:
                    ci_lo, ci_hi = _bootstrap_mean_ci(correct_flags, n_bootstrap, seed)
                else:
                    ci_lo = ci_hi = accuracy
            else:
                accuracy = ci_lo = ci_hi = None

            results[key] = CategoryStats(
                category=cat,
                modality=mod,
                n=n,
                accuracy=accuracy,
                accuracy_ci_lower=ci_lo,
                accuracy_ci_upper=ci_hi,
                mean_likert=None,
                median_likert=None,
                likert_ci_lower=None,
                likert_ci_upper=None,
            )

        else:  # vqa
            likert_scores: list[float] = []
            for rec in recs:
                score = rec.get("likert_score")
                if score is not None:
                    try:
                        s = float(score)
                        if 1.0 <= s <= 5.0:
                            likert_scores.append(s)
                    except (TypeError, ValueError):
                        pass

            if likert_scores:
                mean_l = sum(likert_scores) / len(likert_scores)
                median_l = _safe_median(likert_scores)
                if len(likert_scores) >= 2:
                    ci_lo, ci_hi = _bootstrap_mean_ci(likert_scores, n_bootstrap, seed)
                else:
                    ci_lo = ci_hi = mean_l
            else:
                mean_l = median_l = ci_lo = ci_hi = None

            results[key] = CategoryStats(
                category=cat,
                modality=mod,
                n=n,
                accuracy=None,
                accuracy_ci_lower=None,
                accuracy_ci_upper=None,
                mean_likert=mean_l,
                median_likert=median_l,
                likert_ci_lower=ci_lo,
                likert_ci_upper=ci_hi,
            )

    return results


def format_category_table(
    stats: dict[str, CategoryStats],
    modality: str | None = None,
) -> str:
    """Format per-category statistics as a Markdown table.

    Args:
        stats: Output of :func:`compute_per_category_stats`.
        modality: If given, only include rows for this modality.

    Returns:
        A Markdown-formatted table string.
    """
    rows = list(stats.values())
    if modality is not None:
        rows = [r for r in rows if r.modality == modality]
    rows.sort(key=lambda r: (r.modality, r.category))

    if not rows:
        return "(no data)"

    # Determine which columns to show
    has_mcq = any(r.accuracy is not None for r in rows)
    has_vqa = any(r.mean_likert is not None for r in rows)

    lines: list[str] = []

    if has_mcq:
        lines.append("| Category | Modality | N | Accuracy | 95% CI |")
        lines.append("|---|---|---:|---:|---|")
        for r in rows:
            if r.accuracy is None:
                acc_str = "—"
                ci_str = "—"
            else:
                acc_str = f"{r.accuracy * 100:.1f}%"
                if r.accuracy_ci_lower is not None and r.accuracy_ci_upper is not None:
                    ci_str = f"[{r.accuracy_ci_lower * 100:.1f}%, {r.accuracy_ci_upper * 100:.1f}%]"
                else:
                    ci_str = "—"
            lines.append(f"| {r.category} | {r.modality} | {r.n} | {acc_str} | {ci_str} |")

    if has_vqa:
        if lines:
            lines.append("")
        lines.append("| Category | Modality | N | Mean Likert | Median | 95% CI |")
        lines.append("|---|---|---:|---:|---:|---|")
        for r in rows:
            if r.mean_likert is None:
                mean_str = "—"
                med_str = "—"
                ci_str = "—"
            else:
                mean_str = f"{r.mean_likert:.2f}"
                med_str = f"{r.median_likert:.1f}" if r.median_likert is not None else "—"
                if r.likert_ci_lower is not None and r.likert_ci_upper is not None:
                    ci_str = f"[{r.likert_ci_lower:.2f}, {r.likert_ci_upper:.2f}]"
                else:
                    ci_str = "—"
            lines.append(f"| {r.category} | {r.modality} | {r.n} | {mean_str} | {med_str} | {ci_str} |")

    return "\n".join(lines)


def save_category_stats(
    stats: dict[str, CategoryStats],
    out_path: str,
) -> None:
    """Serialise per-category statistics to a JSON file.

    Args:
        stats: Output of :func:`compute_per_category_stats`.
        out_path: Destination file path (will be created / overwritten).
    """
    serialisable = {key: asdict(val) for key, val in stats.items()}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
