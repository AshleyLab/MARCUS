"""
MARCUS Checkpoint Downloader.

Downloads pre-trained MARCUS expert model checkpoints from HuggingFace Hub.
The checkpoints correspond to the Stage 3 GRPO deployment models described
in the MARCUS paper (O'Sullivan et al., 2026).

Usage::

    # Download all three expert models
    python scripts/download_checkpoints.py --model all

    # Download a specific modality
    python scripts/download_checkpoints.py --model ecg

    # Download to a custom directory
    python scripts/download_checkpoints.py --model all --out-dir /data/marcus/

Checkpoints are saved to::

    saves/Qwen2.5-VL-3B-Instruct/full/
    ├── ecg_grpo/    # ECG expert (Stage 3 deployment checkpoint)
    ├── echo_grpo/   # Echo expert (Stage 3 deployment checkpoint)
    └── cmr_grpo/    # CMR expert (Stage 3 deployment checkpoint)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace repository details
# ---------------------------------------------------------------------------

REPO_ID = "stanford-cardiac-ai/MARCUS"

CHECKPOINT_MAP = {
    "ecg":  "ecg_grpo",
    "echo": "echo_grpo",
    "cmr":  "cmr_grpo",
}

DEFAULT_OUT_DIR = Path("saves/Qwen2.5-VL-3B-Instruct/full")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _check_hf_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise SystemExit(
            "huggingface_hub package required:\n"
            "  pip install huggingface_hub\n"
            "or install all dependencies:\n"
            "  pip install -e '.[all]'"
        )


def _download_checkpoint(
    modality: str,
    subfolder: str,
    out_dir: Path,
    token: str | None,
) -> Path:
    """Download a single expert checkpoint from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    local_dir = out_dir / subfolder
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading %s expert checkpoint from %s/%s ...",
        modality.upper(), REPO_ID, subfolder,
    )

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        allow_patterns=[f"{subfolder}/**"],
        local_dir=str(out_dir),
        token=token,
    )

    logger.info("Checkpoint saved to: %s", local_dir)
    return local_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MARCUS expert model checkpoints from HuggingFace."
    )
    parser.add_argument(
        "--model", required=True,
        choices=["ecg", "echo", "cmr", "all"],
        help="Which expert model(s) to download",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Directory to save checkpoints (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision/branch to download (default: main)",
    )
    args = parser.parse_args()

    _check_hf_hub()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modalities = list(CHECKPOINT_MAP.keys()) if args.model == "all" else [args.model]

    failed = []
    for modality in modalities:
        subfolder = CHECKPOINT_MAP[modality]
        try:
            _download_checkpoint(modality, subfolder, out_dir, args.token)
        except Exception as exc:
            logger.error("Failed to download %s checkpoint: %s", modality, exc)
            failed.append(modality)

    if failed:
        logger.error("Download failed for: %s", ", ".join(failed))
        sys.exit(1)

    logger.info("")
    logger.info("All checkpoints downloaded successfully.")
    logger.info("You can now start the expert servers:")
    for modality in modalities:
        logger.info("  marcus-%s", modality)


if __name__ == "__main__":
    main()
