#!/usr/bin/env python3
"""
B-Clean Dataset Filtering.

Implements the B-Clean protocol from the MARCUS paper (O'Sullivan et al.,
2026) to remove questions that can be answered correctly without
any visual input — a form of dataset contamination where text priors alone
are sufficient for correct classification.

Protocol
--------
For each question in the dataset:

1. Send the question text **only** (no image/video) to the model API.
2. For MCQ: flag if the text-only answer matches the ground truth.
3. Questions answerable by **any** evaluated model without visual input are
   excluded from the primary performance comparison.

In the MARCUS paper, this filtering retained 60% of the original question
set and confirmed that reported accuracy differences reflect visual
understanding rather than prior knowledge or benchmark-specific patterns.

Usage
-----
    python scripts/b_clean_filter.py \\
        --input dataset.json \\
        --api-url http://localhost:8775 \\
        --out filtered_dataset.json \\
        [--threshold 0.7] \\
        [--task mcq|vqa] \\
        [--delay 0.5] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# API helper (text-only query)
# ---------------------------------------------------------------------------

async def _text_only_query(
    client: httpx.AsyncClient,
    api_url: str,
    question: str,
    choices: Optional[list[str]] = None,
    model: str = "default",
    max_tokens: int = 64,
) -> str:
    """Query the model with text only (no image/video attached)."""
    full_question = question
    if choices:
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        full_question = (
            f"{question}\n\nOptions:\n{opts}\n\n"
            "Answer with the letter only (A, B, C, D, or E)."
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": full_question}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = await client.post(f"{api_url}/v1/chat/completions", json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# MCQ answer extraction
# ---------------------------------------------------------------------------

def _extract_mcq_letter(text: str) -> Optional[str]:
    """Extract MCQ answer letter from model output."""
    patterns = [
        r"^([A-E])[.)\s]",
        r"(?:answer is|answer:?)\s*([A-E])\b",
        r"\b([A-E])\s+is correct",
        r"^\s*([A-E])\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # Last resort: find any standalone letter A-E
    m = re.search(r"\b([A-E])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _mcq_matches_gt(prediction: str, gt: str) -> bool:
    """Check if extracted MCQ letter matches ground truth."""
    pred_letter = _extract_mcq_letter(prediction)
    gt_letter = _extract_mcq_letter(gt) or gt.strip().upper()
    if pred_letter and gt_letter:
        return pred_letter == gt_letter
    # Fall back to case-insensitive substring match
    return prediction.strip().lower().startswith(gt.strip().lower()[:3])


# ---------------------------------------------------------------------------
# VQA similarity (simple token overlap for text-only check)
# ---------------------------------------------------------------------------

def _vqa_similarity(prediction: str, gt: str) -> float:
    """Token-level Jaccard similarity between prediction and ground truth."""
    def tok(s: str) -> set:
        return set(re.findall(r"\b[a-z]+\b", s.lower()))
    t1, t2 = tok(prediction), tok(gt)
    if not t1 and not t2:
        return 1.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-item check
# ---------------------------------------------------------------------------

async def check_item(
    item: dict,
    client: httpx.AsyncClient,
    api_url: str,
    task: str,
    threshold: float,
    model: str,
) -> dict:
    """Test if an item is answerable text-only. Returns item augmented with flags."""
    result = dict(item)
    question = item["question"]
    gt = item.get("gt", "")
    choices = item.get("choices") if task == "mcq" else None

    try:
        text_only_pred = await _text_only_query(
            client, api_url, question, choices=choices, model=model
        )
        result["b_clean_text_only_prediction"] = text_only_pred

        if task == "mcq":
            answered_correctly = _mcq_matches_gt(text_only_pred, gt)
        else:
            similarity = _vqa_similarity(text_only_pred, gt)
            answered_correctly = similarity >= threshold
            result["b_clean_vqa_similarity"] = similarity

        result["b_clean_excluded"] = answered_correctly

    except Exception as exc:
        log.warning("B-Clean check failed for question %r: %s", question[:60], exc)
        result["b_clean_excluded"] = False
        result["b_clean_error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

async def run_b_clean(
    dataset: list[dict],
    api_url: str,
    task: str,
    threshold: float,
    delay: float,
    model: str,
) -> list[dict]:
    """Run B-Clean filtering over the full dataset."""
    results = []
    iterator = tqdm(dataset, desc="B-Clean", unit="item") if _HAS_TQDM else dataset

    async with httpx.AsyncClient(timeout=60.0) as client:
        for item in iterator:
            result = await check_item(item, client, api_url, task, threshold, model)
            results.append(result)
            if delay > 0:
                await asyncio.sleep(delay)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B-Clean dataset filtering for MARCUS evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, help="Input dataset JSON file.")
    p.add_argument("--api-url", required=True,
                   help="Model API base URL (e.g. http://localhost:8775).")
    p.add_argument("--out", required=True, help="Output filtered dataset JSON file.")
    p.add_argument("--task", choices=["mcq", "vqa"], default="mcq",
                   help="Task type (default: mcq).")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="VQA similarity threshold for exclusion (default: 0.7).")
    p.add_argument("--delay", type=float, default=0.2,
                   help="Delay between API calls in seconds (default: 0.2).")
    p.add_argument("--model", default="default", help="Model name for API.")
    p.add_argument("--dry-run", action="store_true",
                   help="Process only the first 10 items (for testing).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Loading dataset from %s", args.input)
    dataset = json.loads(Path(args.input).read_text())
    if args.dry_run:
        dataset = dataset[:10]
        log.info("Dry-run mode: using first 10 items only.")

    log.info("Running B-Clean filtering on %d items (task=%s)", len(dataset), args.task)
    t0 = time.time()

    results = asyncio.run(
        run_b_clean(
            dataset,
            api_url=args.api_url,
            task=args.task,
            threshold=args.threshold,
            delay=args.delay,
            model=args.model,
        )
    )

    elapsed = time.time() - t0

    # Summary statistics
    n_total = len(results)
    n_excluded = sum(1 for r in results if r.get("b_clean_excluded", False))
    n_retained = n_total - n_excluded
    pct_retained = 100 * n_retained / n_total if n_total > 0 else 0

    log.info(
        "B-Clean complete in %.1fs | Total: %d | Excluded: %d | Retained: %d (%.1f%%)",
        elapsed, n_total, n_excluded, n_retained, pct_retained,
    )

    # Per-category breakdown
    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "excluded": 0}
        by_category[cat]["total"] += 1
        if r.get("b_clean_excluded", False):
            by_category[cat]["excluded"] += 1

    if by_category:
        log.info("Per-category exclusion rates:")
        for cat, counts in sorted(by_category.items()):
            pct = 100 * counts["excluded"] / counts["total"] if counts["total"] else 0
            log.info("  %-25s  %3d/%3d excluded (%.0f%%)",
                     cat, counts["excluded"], counts["total"], pct)

    # Save output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Full annotated dataset saved to %s", out_path)

    # Also save only the retained items
    retained_path = out_path.with_name(out_path.stem + "_retained" + out_path.suffix)
    retained = [r for r in results if not r.get("b_clean_excluded", False)]
    retained_path.write_text(json.dumps(retained, indent=2))
    log.info("Retained items (%d) saved to %s", len(retained), retained_path)

    print(f"\nB-Clean summary: {n_retained}/{n_total} items retained ({pct_retained:.1f}%)")


if __name__ == "__main__":
    main()
