#!/usr/bin/env python3
"""
MARCUS Statistical Analysis Script.

Reproduces all statistical tests reported in the MARCUS paper
(O'Sullivan et al., 2026):

  - McNemar's test (paired MCQ comparison)
  - Mann-Whitney U test (VQA Likert score comparison)
  - Bootstrap 95% confidence intervals (accuracy and mean Likert)
  - Per-category breakdowns

Usage
-----
    python scripts/compute_statistics.py \\
        --predictions predictions_marcus.json \\
        --baseline predictions_gpt5.json \\
        --task mcq \\
        --out-dir stats_output/ \\
        [--seed 42] \\
        [--n-bootstrap 5000]

Input JSON format
-----------------
List of objects, each with:
  - ``task``: "mcq" or "vqa"
  - ``correct``: bool (MCQ only) — True if model answer matches ground truth
  - ``likert_score``: int 1-5 (VQA only)
  - ``category``: str (optional, e.g. "arrhythmia")
  - ``modality``: str (optional, e.g. "ecg")

Both ``--predictions`` and ``--baseline`` must contain the same number of
items in the same order (paired evaluation).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import scipy; fall back to pure-numpy implementations
# ---------------------------------------------------------------------------
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    log.warning("scipy not found; falling back to pure-numpy statistical tests")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StatResult:
    """Result of a single statistical comparison."""
    metric: str          # "accuracy" | "mean_likert"
    model: str
    n: int
    value: float
    ci_lower: float
    ci_upper: float
    p_value_vs_baseline: Optional[float] = None
    test_name: Optional[str] = None


@dataclass
class CategoryResult:
    """Per-category statistics."""
    category: str
    modality: str
    n: int
    model_value: float
    baseline_value: float
    model_ci_lower: float
    model_ci_upper: float
    p_value: Optional[float] = None


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    statistic=np.mean,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    values : list[float]
        Sample values.
    statistic : callable
        Statistic to compute (default np.mean).
    n_resamples : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (default 0.95 → 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    point = float(statistic(arr))
    boot_stats = np.array([
        statistic(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_resamples)
    ])
    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# McNemar's test (paired MCQ)
# ---------------------------------------------------------------------------

def mcnemar_test(
    model_correct: list[bool],
    baseline_correct: list[bool],
) -> tuple[float, str]:
    """
    McNemar's test for paired binary outcomes.

    Uses the exact binomial test when the number of discordant pairs < 25,
    otherwise the continuity-corrected chi-squared version.

    Returns
    -------
    (p_value, test_name)
    """
    if len(model_correct) != len(baseline_correct):
        raise ValueError("Paired predictions must have the same length.")

    # Discordant counts
    b = sum(1 for m, bl in zip(model_correct, baseline_correct) if m and not bl)
    c = sum(1 for m, bl in zip(model_correct, baseline_correct) if not m and bl)
    n_discordant = b + c

    if n_discordant == 0:
        return 1.0, "McNemar (exact, n_discordant=0)"

    if n_discordant < 25:
        # Exact binomial (Pr(X <= min(b,c)) * 2, two-sided)
        if _HAS_SCIPY:
            result = _scipy_stats.binom_test(min(b, c), n=n_discordant, p=0.5)
            return float(result), "McNemar (exact binomial)"
        else:
            # Pure-numpy exact test
            from math import comb
            half_p = sum(comb(n_discordant, k) * (0.5 ** n_discordant) for k in range(min(b, c) + 1))
            return min(1.0, 2 * half_p), "McNemar (exact binomial, numpy)"
    else:
        # Continuity-corrected chi-squared
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        if _HAS_SCIPY:
            p = float(_scipy_stats.chi2.sf(chi2, df=1))
        else:
            # Approximate p-value using normal approximation of chi2(1)
            z = chi2 ** 0.5
            p = 2 * (1 - _normal_cdf(z))
        return p, "McNemar (continuity-corrected chi-squared)"


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using the error function."""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Mann-Whitney U test (VQA Likert)
# ---------------------------------------------------------------------------

def mannwhitney_u_test(
    model_scores: list[float],
    baseline_scores: list[float],
) -> tuple[float, str]:
    """
    Two-sided Mann-Whitney U test for ordinal Likert scores.

    Returns
    -------
    (p_value, test_name)
    """
    if _HAS_SCIPY:
        stat, p = _scipy_stats.mannwhitneyu(
            model_scores, baseline_scores, alternative="two-sided"
        )
        return float(p), "Mann-Whitney U (two-sided)"
    else:
        # Approximate using normal approximation
        n1, n2 = len(model_scores), len(baseline_scores)
        all_vals = sorted(model_scores + baseline_scores)
        ranks = {v: [] for v in set(all_vals)}
        for i, v in enumerate(all_vals, 1):
            ranks[v].append(i)
        avg_ranks = {v: sum(r) / len(r) for v, r in ranks.items()}
        R1 = sum(avg_ranks[v] for v in model_scores)
        U1 = R1 - n1 * (n1 + 1) / 2
        mean_U = n1 * n2 / 2
        std_U = ((n1 * n2 * (n1 + n2 + 1)) / 12) ** 0.5
        z = (U1 - mean_U) / (std_U + 1e-9)
        p = min(1.0, 2 * (1 - _normal_cdf(abs(z))))
        return p, "Mann-Whitney U (normal approximation)"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_mcq_analysis(
    model_preds: list[dict],
    baseline_preds: list[dict],
    n_bootstrap: int,
    seed: int,
) -> tuple[list[StatResult], list[CategoryResult]]:
    """MCQ analysis: accuracy + McNemar + bootstrap CI + per-category."""
    model_correct = [bool(p.get("correct", False)) for p in model_preds]
    base_correct = [bool(p.get("correct", False)) for p in baseline_preds]

    m_acc, m_ci_lo, m_ci_hi = bootstrap_ci(
        [float(v) for v in model_correct], np.mean, n_bootstrap, seed=seed
    )
    b_acc, b_ci_lo, b_ci_hi = bootstrap_ci(
        [float(v) for v in base_correct], np.mean, n_bootstrap, seed=seed
    )

    p_val, test_name = mcnemar_test(model_correct, base_correct)

    results = [
        StatResult("accuracy", "model", len(model_correct), m_acc, m_ci_lo, m_ci_hi, p_val, test_name),
        StatResult("accuracy", "baseline", len(base_correct), b_acc, b_ci_lo, b_ci_hi),
    ]

    # Per-category
    cat_results = _per_category_mcq(model_preds, baseline_preds, n_bootstrap, seed)
    return results, cat_results


def run_vqa_analysis(
    model_preds: list[dict],
    baseline_preds: list[dict],
    n_bootstrap: int,
    seed: int,
) -> tuple[list[StatResult], list[CategoryResult]]:
    """VQA analysis: mean Likert + Mann-Whitney + bootstrap CI + per-category."""
    m_scores = [float(p.get("likert_score", 3)) for p in model_preds]
    b_scores = [float(p.get("likert_score", 3)) for p in baseline_preds]

    m_mean, m_ci_lo, m_ci_hi = bootstrap_ci(m_scores, np.mean, n_bootstrap, seed=seed)
    b_mean, b_ci_lo, b_ci_hi = bootstrap_ci(b_scores, np.mean, n_bootstrap, seed=seed)

    p_val, test_name = mannwhitney_u_test(m_scores, b_scores)

    results = [
        StatResult("mean_likert", "model", len(m_scores), m_mean, m_ci_lo, m_ci_hi, p_val, test_name),
        StatResult("mean_likert", "baseline", len(b_scores), b_mean, b_ci_lo, b_ci_hi),
    ]

    cat_results = _per_category_vqa(model_preds, baseline_preds, n_bootstrap, seed)
    return results, cat_results


def _per_category_mcq(model_preds, baseline_preds, n_bootstrap, seed):
    categories = {}
    for mp, bp in zip(model_preds, baseline_preds):
        cat = mp.get("category", "unknown")
        mod = mp.get("modality", "unknown")
        key = (cat, mod)
        if key not in categories:
            categories[key] = {"m_correct": [], "b_correct": []}
        categories[key]["m_correct"].append(float(mp.get("correct", False)))
        categories[key]["b_correct"].append(float(bp.get("correct", False)))

    results = []
    for (cat, mod), data in categories.items():
        mc, bc = data["m_correct"], data["b_correct"]
        m_acc, m_lo, m_hi = bootstrap_ci(mc, np.mean, n_bootstrap, seed=seed)
        b_acc, _, _ = bootstrap_ci(bc, np.mean, n_bootstrap, seed=seed)
        p = None
        if len(mc) >= 5:
            try:
                p, _ = mcnemar_test([bool(v) for v in mc], [bool(v) for v in bc])
            except Exception:
                pass
        results.append(CategoryResult(cat, mod, len(mc), m_acc, b_acc, m_lo, m_hi, p))
    return results


def _per_category_vqa(model_preds, baseline_preds, n_bootstrap, seed):
    categories = {}
    for mp, bp in zip(model_preds, baseline_preds):
        cat = mp.get("category", "unknown")
        mod = mp.get("modality", "unknown")
        key = (cat, mod)
        if key not in categories:
            categories[key] = {"m_scores": [], "b_scores": []}
        categories[key]["m_scores"].append(float(mp.get("likert_score", 3)))
        categories[key]["b_scores"].append(float(bp.get("likert_score", 3)))

    results = []
    for (cat, mod), data in categories.items():
        ms, bs = data["m_scores"], data["b_scores"]
        m_mean, m_lo, m_hi = bootstrap_ci(ms, np.mean, n_bootstrap, seed=seed)
        b_mean, _, _ = bootstrap_ci(bs, np.mean, n_bootstrap, seed=seed)
        p = None
        if len(ms) >= 5 and len(bs) >= 5:
            try:
                p, _ = mannwhitney_u_test(ms, bs)
            except Exception:
                pass
        results.append(CategoryResult(cat, mod, len(ms), m_mean, b_mean, m_lo, m_hi, p))
    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_table(stats: list[StatResult], task: str) -> str:
    metric_label = "Accuracy (%)" if task == "mcq" else "Mean Likert"
    lines = [
        f"\n{'='*70}",
        f"  MARCUS Statistical Analysis  |  Task: {task.upper()}",
        f"{'='*70}",
        f"{'Model':<15} {'N':>6}  {metric_label:>14}  {'95% CI':>20}  {'p-value':>12}  {'Test'}",
        f"{'-'*70}",
    ]
    for s in stats:
        val = f"{s.value*100:.1f}" if task == "mcq" else f"{s.value:.2f}"
        ci = f"({s.ci_lower*100 if task=='mcq' else s.ci_lower:.1f}–{s.ci_upper*100 if task=='mcq' else s.ci_upper:.1f})"
        p = f"p={s.p_value_vs_baseline:.4f}" if s.p_value_vs_baseline is not None else "—"
        test = s.test_name or "—"
        lines.append(f"{s.model:<15} {s.n:>6}  {val:>14}  {ci:>20}  {p:>12}  {test}")
    lines.append("=" * 70)
    return "\n".join(lines)


def _format_category_table(cat_results: list[CategoryResult], task: str) -> str:
    lines = [
        f"\n{'='*80}",
        "  Per-Category Breakdown",
        f"{'='*80}",
        f"{'Category':<25} {'Mod':<6} {'N':>5}  {'Model':>8}  {'Baseline':>8}  {'p-value':>10}",
        "-" * 80,
    ]
    for r in sorted(cat_results, key=lambda x: x.category):
        mv = f"{r.model_value*100:.1f}%" if task == "mcq" else f"{r.model_value:.2f}"
        bv = f"{r.baseline_value*100:.1f}%" if task == "mcq" else f"{r.baseline_value:.2f}"
        p = f"p={r.p_value:.4f}" if r.p_value is not None else "—"
        lines.append(
            f"{r.category:<25} {r.modality:<6} {r.n:>5}  {mv:>8}  {bv:>8}  {p:>10}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MARCUS statistical analysis — reproduce paper tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--predictions", required=True, help="JSON file with model predictions.")
    p.add_argument("--baseline", required=True, help="JSON file with baseline predictions.")
    p.add_argument("--task", choices=["mcq", "vqa"], required=True, help="Evaluation task type.")
    p.add_argument("--out-dir", default="stats_output", help="Output directory (default: stats_output).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--n-bootstrap", type=int, default=5000, help="Bootstrap resamples (default: 5000).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Loading predictions from %s", args.predictions)
    model_preds = json.loads(Path(args.predictions).read_text())
    log.info("Loading baseline from %s", args.baseline)
    baseline_preds = json.loads(Path(args.baseline).read_text())

    if len(model_preds) != len(baseline_preds):
        raise ValueError(
            f"Prediction files have different lengths: "
            f"{len(model_preds)} vs {len(baseline_preds)}. "
            "Predictions must be paired (same order, same items)."
        )

    log.info("Running %s analysis (n=%d, n_bootstrap=%d, seed=%d)",
             args.task, len(model_preds), args.n_bootstrap, args.seed)

    if args.task == "mcq":
        stats, cat_results = run_mcq_analysis(
            model_preds, baseline_preds, args.n_bootstrap, args.seed
        )
    else:
        stats, cat_results = run_vqa_analysis(
            model_preds, baseline_preds, args.n_bootstrap, args.seed
        )

    # Print tables
    table = _format_table(stats, args.task)
    cat_table = _format_category_table(cat_results, args.task)
    print(table)
    print(cat_table)

    # Save output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "task": args.task,
        "n": len(model_preds),
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "stats": [asdict(s) for s in stats],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_category.json").write_text(
        json.dumps([asdict(r) for r in cat_results], indent=2)
    )
    (out_dir / "summary.txt").write_text(table + "\n" + cat_table)

    log.info("Results saved to %s/", out_dir)


if __name__ == "__main__":
    main()
