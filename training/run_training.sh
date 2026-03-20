#!/usr/bin/env bash
# MARCUS — Full three-stage training pipeline for a single modality.
#
# Stages:
#   1. Vision encoder pre-training   (LLaMA-Factory SFT, LLM frozen)
#   2. Supervised fine-tuning (SFT)  (LLaMA-Factory SFT, all params)
#   3. GRPO reinforcement learning   (verl, MCQ reward via mathruler)
#
# Usage:
#   ./training/run_training.sh ecg       # train ECG expert
#   ./training/run_training.sh echo      # train Echo expert
#   ./training/run_training.sh cmr       # train CMR expert
#
# Prerequisites:
#   pip install llamafactory        # Stages 1 & 2
#   pip install verl mathruler sglang  # Stage 3
#
# Environment variables:
#   NGPU          Number of GPUs (default: auto-detected via nvidia-smi)
#   WANDB_PROJECT W&B project name (default: marcus-{modality})
#   EXPORT_DIR    Where to export the SFT checkpoint for verl (default: LLaMA-Factory/export_{modality}_sft)

set -euo pipefail

MODALITY="${1:-}"
if [[ -z "$MODALITY" ]]; then
    echo "Usage: $0 <ecg|echo|cmr>" >&2
    exit 1
fi

case "$MODALITY" in
    ecg|echo|cmr) ;;
    *)
        echo "Unknown modality: $MODALITY. Must be ecg, echo, or cmr." >&2
        exit 1
        ;;
esac

# ── GPU detection ────────────────────────────────────────────────────────────
if [[ -z "${NGPU:-}" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        NGPU=1
    fi
fi
echo "[MARCUS] Modality: $MODALITY | GPUs: $NGPU"

export WANDB_PROJECT="${WANDB_PROJECT:-marcus-${MODALITY}}"
EXPORT_DIR="${EXPORT_DIR:-LLaMA-Factory/export_${MODALITY}_sft}"

# ── Helper: run an SFT stage ─────────────────────────────────────────────────
run_sft_stage() {
    local stage_name="$1"
    local config="$2"
    echo ""
    echo "========================================================"
    echo "  MARCUS Stage: $stage_name ($MODALITY)"
    echo "  Config: $config"
    echo "========================================================"

    if [[ "$NGPU" -gt 1 ]]; then
        torchrun \
            --nproc_per_node="$NGPU" \
            --master_port=29500 \
            "$(which llamafactory-cli)" train "$config"
    else
        llamafactory-cli train "$config"
    fi

    echo "[MARCUS] $stage_name complete."
}

START_TIME=$(date +%s)

# ── Stage 1: Vision Encoder Pre-training (LLaMA-Factory) ─────────────────────
run_sft_stage "Stage 1 — Vision Encoder Pre-training" \
    "training/stage1_pretrain/${MODALITY}_pretrain.yaml"

# ── Stage 2: Supervised Fine-Tuning (LLaMA-Factory) ──────────────────────────
run_sft_stage "Stage 2 — Supervised Fine-Tuning" \
    "training/stage2_sft/${MODALITY}_sft.yaml"

# ── Export SFT checkpoint for verl ───────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Exporting SFT checkpoint for verl GRPO..."
echo "  Export dir: $EXPORT_DIR"
echo "========================================================"
llamafactory-cli export "training/stage2_sft/${MODALITY}_sft.yaml" \
    --export_dir "$EXPORT_DIR"

# ── Prepare GRPO parquet data ─────────────────────────────────────────────────
echo ""
echo "  Preparing GRPO parquet data for verl..."
case "$MODALITY" in
    ecg)
        python verl/examples/data_preprocess/ecg_simple.py \
            --local_dir "data/ecg_simple"
        ;;
    echo)
        python verl/examples/data_preprocess/echo_mcq.py \
            --local_dir "data/echo_mcq"
        ;;
    cmr)
        python verl/examples/data_preprocess/cmr_mcq.py \
            --local_dir "data/cmr_mcq"
        ;;
esac

# ── Stage 3: GRPO Reinforcement Learning (verl) ───────────────────────────────
echo ""
echo "========================================================"
echo "  MARCUS Stage: Stage 3 — GRPO Reinforcement Learning ($MODALITY)"
echo "  Framework: verl (https://github.com/volcengine/verl)"
echo "========================================================"
MODEL_PATH="$EXPORT_DIR" \
    bash "training/stage3_grpo/run_${MODALITY}_grpo.sh"

echo "[MARCUS] Stage 3 GRPO complete."

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "========================================================"
echo "  MARCUS training complete!"
echo "  Modality: $MODALITY | Total time: ${HOURS}h ${MINS}m"
echo "  Stage 3 checkpoint: verl/checkpoints/marcus_${MODALITY}_grpo/"
echo "========================================================"
