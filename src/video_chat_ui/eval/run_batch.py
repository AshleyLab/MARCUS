"""Load JSON batch, score each row, write output + summary."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from video_chat_ui.eval.judge import score_mcq, score_vqa


def run_batch(
    rows: list[dict[str, Any]],
    task: Literal["vqa", "mcq"],
    *,
    question_key: str = "question",
    gt_key: str = "ground_truth",
    pred_key: str = "model_answer",
    judge_model: str = "gpt-4o-mini",
    delay_s: float = 0.0,
    max_retries: int = 2,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    augmented: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="Judging"):
        q = row.get(question_key)
        gt = row.get(gt_key)
        pred = row.get(pred_key)
        if q is None or gt is None or pred is None:
            raise KeyError(
                f"Missing {question_key!r}, {gt_key!r}, or {pred_key!r} in row: {list(row.keys())}"
            )
        out_row = dict(row)
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if task == "vqa":
                    r = score_vqa(str(q), str(gt), str(pred), model=judge_model)
                    out_row["likert_score"] = r["answer"]
                    out_row["likert_explanation"] = r["explanation"]
                else:
                    r = score_mcq(str(q), str(gt), str(pred), model=judge_model)
                    out_row["eval_label"] = r["answer"]
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
        if last_err is not None:
            out_row["eval_error"] = str(last_err)
        augmented.append(out_row)
        if delay_s > 0:
            time.sleep(delay_s)

    out_path = out_dir / "output.json"
    out_path.write_text(json.dumps(augmented, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {"task": task, "total": len(augmented)}
    if task == "vqa":
        scored = [r for r in augmented if "likert_score" in r]
        if scored:
            s = sum(r["likert_score"] for r in scored)
            summary["mean_likert"] = s / len(scored)
            summary["scored_count"] = len(scored)
        errs = sum(1 for r in augmented if "eval_error" in r)
        summary["errors"] = errs
    else:
        correct = sum(1 for r in augmented if r.get("eval_label") == "Correct")
        excluded = sum(1 for r in augmented if r.get("eval_label") == "Excluded")
        incorrect = sum(1 for r in augmented if r.get("eval_label") == "Incorrect")
        denom = len(augmented) - excluded - sum(1 for r in augmented if "eval_error" in r)
        summary["correct"] = correct
        summary["incorrect"] = incorrect
        summary["excluded"] = excluded
        summary["errors"] = sum(1 for r in augmented if "eval_error" in r)
        if denom > 0:
            summary["accuracy_excluding_excluded"] = correct / denom
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
