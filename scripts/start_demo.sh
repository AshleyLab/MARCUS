#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# MARCUS Demo — start all expert APIs, FastAPI UI servers, and Gradio UI
# Usage:  bash scripts/start_demo.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="${LLAMA_FACTORY_DIR:-$HOME/LLaMA-Factory}"
SIF="${LLAMA_FACTORY_SIF:-$HOME/llamafactory_latest.sif}"
YAML="examples/inference/qwen2_5vl.yaml"

# GPU assignment — one model per GPU
ECG_GPU="${ECG_GPU:-0}"
ECHO_GPU="${ECHO_GPU:-1}"
CMR_GPU="${CMR_GPU:-2}"

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down MARCUS demo..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

start_api() {
    local name="$1" gpu="$2" port="$3" model="$4"
    echo "Starting $name expert API on GPU $gpu, port $port..."

    if [ -f "$SIF" ]; then
        cd "$LLAMA_DIR"
        CUDA_VISIBLE_DEVICES="$gpu" singularity exec --nv \
            --env "API_HOST=0.0.0.0,API_PORT=$port,MPLCONFIGDIR=/tmp/mpl" \
            "$SIF" \
            llamafactory-cli api "$YAML" "model_name_or_path=$model" \
            > "/tmp/marcus_${name}_api.log" 2>&1 &
        cd "$PROJECT_DIR"
    else
        cd "$LLAMA_DIR"
        API_HOST=0.0.0.0 API_PORT="$port" CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR=/tmp/mpl \
            llamafactory-cli api "$YAML" "model_name_or_path=$model" \
            > "/tmp/marcus_${name}_api.log" 2>&1 &
        cd "$PROJECT_DIR"
    fi
    PIDS+=($!)
}

wait_api() {
    local name="$1" port="$2" timeout="${3:-300}"
    local deadline=$((SECONDS + timeout))
    while [ $SECONDS -lt $deadline ]; do
        if curl -sf "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; then
            echo "  $name API ready on port $port"
            return 0
        fi
        sleep 3
    done
    echo "  WARNING: $name API did not become ready within ${timeout}s"
    return 1
}

start_ui() {
    local name="$1" api_port="$2" ui_port="$3"
    echo "Starting $name UI server on port $ui_port..."
    LLAMA_API_URL="http://127.0.0.1:${api_port}" \
    PORT="$ui_port" \
    UPLOAD_DIR="/tmp/marcus_uploads" \
    EXPERT_DOC_TITLE="${name} expert model" \
    EXPERT_PAGE_HEADING="${name} expert model" \
    EXPERT_TAGLINE="Upload and chat with the ${name} expert model." \
        python -m uvicorn video_chat_ui.app:app --host 0.0.0.0 --port "$ui_port" \
        > /tmp/marcus_${name}_ui.log 2>&1 &
    PIDS+=($!)
}

# ── Step 1: Start LLaMA-Factory API servers ──────────────────────────
echo "============================================"
echo "  MARCUS Demo Startup"
echo "============================================"
echo ""

start_api "ECG"  "$ECG_GPU"  8020 "saves/Qwen2.5-VL-3B-Instruct/full/ecg_sft"
start_api "Echo" "$ECHO_GPU" 8010 "saves/Qwen2.5-VL-3B-Instruct/full/echo_grpo"
start_api "CMR"  "$CMR_GPU"  8000 "saves/Qwen2.5-VL-3B-Instruct/full/cmr_grpo"

# ── Step 2: Wait for APIs ────────────────────────────────────────────
echo ""
echo "Waiting for expert APIs to load models..."
wait_api "ECG"  8020 &
wait_api "Echo" 8010 &
wait_api "CMR"  8000 &
wait

# ── Step 3: Start FastAPI UI servers ─────────────────────────────────
echo ""
cd "$PROJECT_DIR"
start_ui "ECG"  8020 8775
start_ui "Echo" 8010 8770
start_ui "CMR"  8000 8765

sleep 2

# ── Step 4: Start Gradio demo ────────────────────────────────────────
echo ""
echo "Starting MARCUS Gradio demo on port 7860..."
cd "$PROJECT_DIR"
UPLOAD_DIR="/tmp/marcus_uploads" python -m video_chat_ui.demo > /tmp/marcus_demo.log 2>&1 &
PIDS+=($!)

sleep 3

echo ""
echo "============================================"
echo "  MARCUS Demo running"
echo "  Open http://localhost:7860"
echo ""
echo "  Logs: /tmp/marcus_*.log"
echo "  Press Ctrl+C to stop all services"
echo "============================================"

# Keep alive
wait
