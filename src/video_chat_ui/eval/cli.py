"""CLI: video-chat-eval."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_chat_ui.eval.run_batch import run_batch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="video-chat-eval",
        description="Batch VQA (Likert) or MCQ judge via OpenAI gpt-4o-mini.",
    )
    p.add_argument("--input", required=True, type=Path, help="JSON list of result objects")
    p.add_argument("--task", choices=("vqa", "mcq"), required=True)
    p.add_argument("--gt-key", default="ground_truth", help="Ground-truth field name")
    p.add_argument("--pred-key", default="model_answer", help="Model prediction field name")
    p.add_argument("--question-key", default="question", help="Question field name")
    p.add_argument("--out-dir", type=Path, default=Path("./eval_out"))
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI judge model")
    p.add_argument("--delay", type=float, default=0.0, help="Seconds between API calls")
    p.add_argument("--retries", type=int, default=2, help="Retries per row on failure")
    args = p.parse_args(argv)

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("Input must be a JSON array.", file=sys.stderr)
        return 1
    run_batch(
        data,
        args.task,
        question_key=args.question_key,
        gt_key=args.gt_key,
        pred_key=args.pred_key,
        judge_model=args.model,
        delay_s=args.delay,
        max_retries=args.retries,
        out_dir=args.out_dir,
    )
    print(f"Wrote {args.out_dir / 'output.json'} and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
