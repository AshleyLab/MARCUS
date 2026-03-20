#!/usr/bin/env python3
"""
MARCUS Batch Inference Script.

Generates model predictions for a dataset of clinical questions paired with
cardiac imaging media. Output is directly consumable by ``video-chat-eval``
(``marcus-eval``) for downstream scoring.

Usage
-----
    python scripts/run_inference_batch.py \\
        --input data/test_ecg.json \\
        --modality ecg \\
        --api-url http://localhost:8775 \\
        --out predictions_ecg.json \\
        [--mirage-probe] \\
        [--batch-size 1] \\
        [--delay 0.5] \\
        [--retries 3]

Input JSON format
-----------------
List of objects, each with:
  - ``question``: str
  - ``gt``: str (ground truth answer)
  - ``image_path`` OR ``video_path``: str (path to local media file)
  - ``task``: "vqa" | "mcq"
  - ``choices``: list[str] (MCQ only — the A/B/C/D options)
  - ``category``: str (optional, used for per-category analysis)
  - ``modality``: str (optional, e.g. "ecg")

Output JSON format
------------------
Same as input, each object augmented with:
  - ``prediction``: str (raw model response)
  - ``mirage_probe``: dict (if --mirage-probe enabled)
  - ``inference_error``: str (if the API call failed)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
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
# API helpers
# ---------------------------------------------------------------------------

def _build_content(question: str, media_path: Optional[str], media_kind: str) -> list[dict]:
    """Build the content list for an OpenAI-compatible message."""
    content: list[dict] = [{"type": "text", "text": question}]
    if media_path:
        path = Path(media_path)
        if not path.exists():
            log.warning("Media file not found: %s", media_path)
        else:
            # Encode as data URI for self-contained requests
            suffix = path.suffix.lower().lstrip(".")
            mime = {
                "mp4": "video/mp4",
                "avi": "video/avi",
                "mkv": "video/x-matroska",
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
            }.get(suffix, "application/octet-stream")
            data = base64.b64encode(path.read_bytes()).decode()
            data_url = f"data:{mime};base64,{data}"

            if media_kind == "video":
                content.append({"type": "video_url", "video_url": {"url": data_url}})
            else:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


async def _call_api(
    client: httpx.AsyncClient,
    api_url: str,
    question: str,
    media_path: Optional[str],
    media_kind: str,
    model: str = "default",
    temperature: float = 0.0,
    max_tokens: int = 512,
    choices: Optional[list[str]] = None,
) -> str:
    """
    Single inference call to the OpenAI-compatible expert API.

    For MCQ questions, appends the choices to the question text.
    """
    full_question = question
    if choices:
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        full_question = f"{question}\n\nOptions:\n{opts}\n\nAnswer with the letter only."

    content = _build_content(full_question, media_path, media_kind)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = await client.post(
        f"{api_url}/v1/chat/completions",
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Mirage probe (simplified for batch use)
# ---------------------------------------------------------------------------

async def _mirage_probe(
    client: httpx.AsyncClient,
    api_url: str,
    question: str,
    media_path: str,
    media_kind: str,
    **api_kwargs,
) -> dict:
    """Run a quick mirage probe: one with-image and one without-image call."""
    with_image = await _call_api(client, api_url, question, media_path, media_kind, **api_kwargs)
    without_image = await _call_api(client, api_url, question, None, media_kind, **api_kwargs)

    # Simple token-overlap similarity
    def tokenize(t: str) -> set:
        import re
        return set(re.findall(r"\b[a-z]+\b", t.lower()))

    t1, t2 = tokenize(with_image), tokenize(without_image)
    inter = len(t1 & t2)
    union = len(t1 | t2)
    similarity = inter / union if union > 0 else 1.0
    divergence = 1.0 - similarity

    return {
        "with_image_response": with_image,
        "without_image_response": without_image,
        "divergence_score": divergence,
        "mirage_flag": similarity > 0.85,
    }


# ---------------------------------------------------------------------------
# Per-item inference
# ---------------------------------------------------------------------------

async def infer_item(
    item: dict,
    client: httpx.AsyncClient,
    api_url: str,
    modality: str,
    retries: int = 3,
    delay: float = 0.5,
    run_mirage: bool = False,
    model: str = "default",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> dict:
    """Run inference for a single dataset item with retry logic."""
    media_kind = "image" if modality == "ecg" else "video"
    media_path = item.get("image_path") or item.get("video_path")
    question = item["question"]
    choices = item.get("choices")

    result = dict(item)

    for attempt in range(1, retries + 1):
        try:
            prediction = await _call_api(
                client, api_url, question, media_path, media_kind,
                model=model, temperature=temperature, max_tokens=max_tokens,
                choices=choices,
            )
            result["prediction"] = prediction

            if run_mirage and media_path:
                probe = await _mirage_probe(
                    client, api_url, question, media_path, media_kind,
                    model=model, temperature=temperature, max_tokens=max_tokens,
                )
                result["mirage_probe"] = probe

            result.pop("inference_error", None)
            return result

        except Exception as exc:
            log.warning("Item inference attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                await asyncio.sleep(delay * (2 ** (attempt - 1)))

    result["prediction"] = ""
    result["inference_error"] = "Max retries exceeded"
    return result


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

async def run_batch(
    dataset: list[dict],
    api_url: str,
    modality: str,
    retries: int,
    delay: float,
    run_mirage: bool,
    model: str,
    temperature: float,
    max_tokens: int,
) -> list[dict]:
    """Run inference over the full dataset sequentially with progress bar."""
    results = []
    iterator = tqdm(dataset, desc="Inference", unit="item") if _HAS_TQDM else dataset

    async with httpx.AsyncClient(timeout=120.0) as client:
        for item in iterator:
            result = await infer_item(
                item, client, api_url, modality,
                retries=retries, delay=delay, run_mirage=run_mirage,
                model=model, temperature=temperature, max_tokens=max_tokens,
            )
            results.append(result)
            if delay > 0:
                await asyncio.sleep(delay)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MARCUS batch inference — generate model predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, help="Input dataset JSON file.")
    p.add_argument("--modality", required=True, choices=["ecg", "echo", "cmr"],
                   help="Expert modality.")
    p.add_argument("--api-url", required=True,
                   help="Base URL of the expert API (e.g. http://localhost:8775).")
    p.add_argument("--out", required=True, help="Output predictions JSON file.")
    p.add_argument("--mirage-probe", action="store_true",
                   help="Run counterfactual mirage probing for each item.")
    p.add_argument("--retries", type=int, default=3,
                   help="Number of API call retries (default: 3).")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Delay between items in seconds (default: 0.5).")
    p.add_argument("--model", default="default", help="Model name for API (default: default).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0 = greedy).")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Maximum response tokens (default: 512).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Loading dataset from %s", args.input)
    dataset = json.loads(Path(args.input).read_text())
    log.info("Dataset size: %d items", len(dataset))

    t0 = time.time()
    results = asyncio.run(
        run_batch(
            dataset,
            api_url=args.api_url,
            modality=args.modality,
            retries=args.retries,
            delay=args.delay,
            run_mirage=args.mirage_probe,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )
    elapsed = time.time() - t0

    # Summary
    n_ok = sum(1 for r in results if "prediction" in r and not r.get("inference_error"))
    n_fail = sum(1 for r in results if r.get("inference_error"))
    n_mirage = sum(1 for r in results if r.get("mirage_probe", {}).get("mirage_flag", False))

    log.info(
        "Done in %.1fs | Total: %d | Succeeded: %d | Failed: %d%s",
        elapsed, len(results), n_ok, n_fail,
        f" | Mirage flags: {n_mirage}" if args.mirage_probe else "",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Predictions saved to %s", out_path)


if __name__ == "__main__":
    main()
