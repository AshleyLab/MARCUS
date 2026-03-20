"""
MARCUS Dataset Builder.

Generates visual Q&A pairs from physician reports using GPT-4o and
modality-specific templates. Produces datasets compatible with
LLaMA-Factory's ShareGPT format.

Output format (per example)::

    {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": "/abs/path/to/image_or_video.png"},
                {"type": "text",  "text": "<question>"}
            ]},
            {"role": "assistant", "content": "<answer>"}
        ]
    }

For GRPO (MCQ) datasets, a ``gt`` field is appended with the correct letter::

    {"messages": [...], "gt": "B"}

Usage::

    python scripts/build_dataset.py \\
        --reports-dir /data/ecg_reports \\
        --media-dir   /data/ecg_images \\
        --templates   data/templates/ecg_templates.json \\
        --modality    ecg \\
        --out-dir     data/generated/ \\
        --openai-model gpt-4o \\
        --stage       sft          # or pretrain / grpo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client (lazy import so the rest of the module works without it)
# ---------------------------------------------------------------------------

def _get_openai_client(model: str):
    try:
        from openai import AsyncOpenAI
        return AsyncOpenAI(), model
    except ImportError:
        raise SystemExit(
            "openai package required: pip install openai"
        )


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

def load_templates(path: str) -> dict[str, Any]:
    """Load Q&A generation templates from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Report → Q&A generation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a clinical cardiologist generating structured educational Q&A pairs \
from cardiac imaging reports for training an AI diagnostic system.

Given a physician's report, generate {n_questions} question-answer pairs that:
1. Require interpreting the {modality} image to answer correctly.
2. Cover diverse aspects: findings, measurements, diagnoses, severity grading.
3. Are unambiguous and have a single correct answer.
4. Are phrased naturally as a clinician would ask.

For MCQ format, provide exactly 4 choices (A–D) with one correct answer.

Respond with a JSON array of objects. Each object must have:
  - "question": the clinical question (string)
  - "answer": the correct answer (string)
  - "mcq_choices": list of 4 strings [A_text, B_text, C_text, D_text] (MCQ only)
  - "correct_letter": "A", "B", "C", or "D" (MCQ only)
  - "category": one of {categories}

Return only valid JSON, no markdown fences."""


async def _generate_qa(
    client,
    model: str,
    report_text: str,
    modality: str,
    templates: dict,
    n_questions: int,
    stage: str,
) -> list[dict]:
    """Call GPT-4o to generate Q&A pairs from a single physician report."""
    categories = templates.get("categories", ["diagnosis", "measurement", "finding"])
    system = _SYSTEM_PROMPT.format(
        n_questions=n_questions,
        modality=modality.upper(),
        categories=json.dumps(categories),
    )

    # Add modality-specific guidance from templates
    if "generation_guidance" in templates:
        system += "\n\nModality-specific guidance:\n" + templates["generation_guidance"]

    user_msg = f"Physician report:\n\n{report_text}"
    if stage == "grpo":
        user_msg += "\n\nGenerate MCQ format questions with choices."

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "questions" in parsed:
            return parsed["questions"]
        if isinstance(parsed, list):
            return parsed
        logger.warning("Unexpected JSON structure from LLM: %s", raw[:200])
        return []
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON: %s", raw[:200])
        return []


# ---------------------------------------------------------------------------
# Media file matching
# ---------------------------------------------------------------------------

_MEDIA_EXTENSIONS = {
    "ecg":  [".png", ".jpg", ".npy"],
    "echo": [".mp4", ".avi", ".dcm"],
    "cmr":  [".mp4", ".avi", ".dcm", ".nii", ".nii.gz"],
}


def _find_media_file(
    report_stem: str,
    media_dir: Path,
    modality: str,
) -> Optional[Path]:
    """Find the media file corresponding to a report by stem name."""
    for ext in _MEDIA_EXTENSIONS.get(modality, [".png"]):
        candidate = media_dir / f"{report_stem}{ext}"
        if candidate.exists():
            return candidate
    # Fuzzy: look for files containing the report stem
    for f in media_dir.iterdir():
        if report_stem in f.stem:
            return f
    return None


# ---------------------------------------------------------------------------
# Dataset record construction
# ---------------------------------------------------------------------------

def _build_record(
    qa: dict,
    media_path: Path,
    stage: str,
) -> dict:
    """Build a single LLaMA-Factory ShareGPT record from a Q&A pair."""
    media_type = "video" if media_path.suffix.lower() in (".mp4", ".avi", ".mkv") else "image"

    content = [
        {"type": media_type, media_type: str(media_path.resolve())},
        {"type": "text", "text": qa["question"]},
    ]

    record: dict = {
        "messages": [
            {"role": "user",      "content": content},
            {"role": "assistant", "content": qa["answer"]},
        ]
    }

    if qa.get("category"):
        record["category"] = qa["category"]

    # GRPO: add ground truth letter
    if stage == "grpo" and qa.get("correct_letter"):
        record["gt"] = qa["correct_letter"].upper()
        # Reformat answer to include all choices
        choices = qa.get("mcq_choices", [])
        if choices:
            formatted = "\n".join(
                f"{letter}. {text}"
                for letter, text in zip("ABCD", choices)
            )
            record["messages"][0]["content"][-1]["text"] += f"\n\n{formatted}"

    return record


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def _build_async(args: argparse.Namespace) -> None:
    client, model = _get_openai_client(args.openai_model)
    templates = load_templates(args.templates)

    reports_dir = Path(args.reports_dir)
    media_dir   = Path(args.media_dir) if args.media_dir else reports_dir
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_files = sorted(reports_dir.glob("*.txt")) + sorted(reports_dir.glob("*.json"))
    if not report_files:
        raise SystemExit(f"No report files found in {reports_dir}")

    logger.info("Found %d report files in %s", len(report_files), reports_dir)

    records: list[dict] = []
    skipped = 0

    for i, report_path in enumerate(report_files):
        # Load report text
        if report_path.suffix == ".json":
            data = json.loads(report_path.read_text())
            report_text = data.get("report", data.get("text", str(data)))
        else:
            report_text = report_path.read_text(encoding="utf-8", errors="replace")

        if len(report_text.strip()) < 50:
            skipped += 1
            continue

        # Find corresponding media file
        media_path = _find_media_file(report_path.stem, media_dir, args.modality)
        if media_path is None:
            logger.debug("No media file found for %s — skipping", report_path.name)
            skipped += 1
            continue

        # Generate Q&A pairs
        try:
            qa_list = await _generate_qa(
                client, model, report_text,
                args.modality, templates,
                n_questions=args.n_questions,
                stage=args.stage,
            )
        except Exception as exc:
            logger.warning("LLM call failed for %s: %s", report_path.name, exc)
            skipped += 1
            continue

        for qa in qa_list:
            if not qa.get("question") or not qa.get("answer"):
                continue
            record = _build_record(qa, media_path, args.stage)
            records.append(record)

        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d reports — %d records so far",
                        i + 1, len(report_files), len(records))

        # Polite rate limiting
        if args.delay > 0:
            await asyncio.sleep(args.delay)

    # Shuffle and split
    random.seed(42)
    random.shuffle(records)

    stage_name = f"{args.modality}_{args.stage}"
    out_path = out_dir / f"{stage_name}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    logger.info(
        "Done. %d records written to %s (%d reports skipped)",
        len(records), out_path, skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MARCUS training datasets from physician reports."
    )
    parser.add_argument("--reports-dir", required=True,
                        help="Directory containing physician report files (.txt or .json)")
    parser.add_argument("--media-dir",
                        help="Directory containing media files (default: same as --reports-dir)")
    parser.add_argument("--templates", required=True,
                        help="Path to modality-specific template JSON (e.g. data/templates/ecg_templates.json)")
    parser.add_argument("--modality", required=True, choices=["ecg", "echo", "cmr"],
                        help="Cardiac imaging modality")
    parser.add_argument("--out-dir", default="data/generated/",
                        help="Output directory for generated dataset JSON (default: data/generated/)")
    parser.add_argument("--stage", default="sft", choices=["pretrain", "sft", "grpo"],
                        help="Training stage — controls output format (default: sft)")
    parser.add_argument("--openai-model", default="gpt-4o",
                        help="OpenAI model to use for Q&A generation (default: gpt-4o)")
    parser.add_argument("--n-questions", type=int, default=5,
                        help="Q&A pairs to generate per report (default: 5)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Seconds to sleep between API calls (default: 0.1)")
    args = parser.parse_args()

    asyncio.run(_build_async(args))


if __name__ == "__main__":
    main()
