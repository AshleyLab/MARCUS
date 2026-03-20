# MARCUS Training Guide

This directory contains all training configurations for the MARCUS multimodal cardiac AI system.

- **Stages 1 & 2** (vision encoder pre-training and SFT) use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
- **Stage 3** (GRPO reinforcement learning) uses [verl](https://github.com/volcengine/verl) with the sglang rollout backend.

Base model: Qwen2.5-VL-3B-Instruct.

## Hardware Requirements

| Stage | GPUs | VRAM/GPU | Duration (approx.) |
|-------|------|----------|---------------------|
| Stage 1 (ECG pretrain) | 4× H100 80GB | 75 GB | ~12 h |
| Stage 1 (Echo pretrain) | 8× H100 80GB | 78 GB | ~48 h |
| Stage 1 (CMR pretrain) | 8× H100 80GB | 79 GB | ~72 h |
| Stage 2 SFT (each modality) | 8× H100 80GB | 75 GB | ~24 h |
| Stage 3 GRPO (each modality) | 8× H100 80GB | 78 GB | ~36 h |

All experiments reported in the paper were run on a DGX H100 cluster.

## Setup

### 1. Install LLaMA-Factory

```bash
pip install llamafactory
# Or from source for the latest version:
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]"
```

### 2. Install MARCUS dependencies

```bash
pip install -e ".[all]"
```

### 3. Prepare the dataset

Generate Q&A pairs from physician reports using the dataset builder:

```bash
python scripts/build_dataset.py \
    --reports-dir /path/to/physician/reports \
    --templates data/templates/ecg_templates.json \
    --modality ecg \
    --out-dir data/generated/ \
    --openai-model gpt-4o
```

Repeat for `--modality echo` and `--modality cmr`.

Register the generated datasets with LLaMA-Factory by copying `training/dataset_info.json` to the LLaMA-Factory data directory:

```bash
cp training/dataset_info.json /path/to/LLaMA-Factory/data/dataset_info.json
# Or set: export LLAMAFACTORY_DATA_DIR=data/
```

## Three-Stage Training Pipeline

MARCUS training follows a three-stage curriculum for each modality:

```
Stage 1: Vision encoder pre-training (LLM frozen)
         ↓
Stage 2: Supervised fine-tuning (all parameters)
         ↓
Stage 3: GRPO reinforcement learning on MCQ
```

### Stage 1 — Vision Encoder Pre-training

Trains the modality-specific visual encoder while keeping the Qwen2 language model frozen. This aligns the medical image representation with the LLM's token space.

```bash
# ECG (SigLIP encoder → 2-layer MLP → Qwen2)
llamafactory-cli train training/stage1_pretrain/ecg_pretrain.yaml

# Echo (Multi-view visual encoder → temporal aggregation → Qwen2)
llamafactory-cli train training/stage1_pretrain/echo_pretrain.yaml

# CMR (similar to Echo + metadata-driven sequence selection)
llamafactory-cli train training/stage1_pretrain/cmr_pretrain.yaml
```

Multi-GPU (8 GPUs):
```bash
torchrun --nproc_per_node=8 \
    $(which llamafactory-cli) train training/stage1_pretrain/ecg_pretrain.yaml
```

### Stage 2 — Supervised Fine-tuning (SFT)

Jointly fine-tunes the vision encoder and language model on visual Q&A pairs with physician-verified ground truths.

| Modality | Training pairs | Epochs |
|----------|---------------|--------|
| ECG | 460,000 | 5 |
| Echo | 155,000 | 5 |
| CMR | 126,000 | 5 |

```bash
llamafactory-cli train training/stage2_sft/ecg_sft.yaml
llamafactory-cli train training/stage2_sft/echo_sft.yaml
llamafactory-cli train training/stage2_sft/cmr_sft.yaml
```

### Stage 3 — GRPO Reinforcement Learning (verl)

Optimises model MCQ accuracy using Group Relative Policy Optimization via [verl](https://github.com/volcengine/verl). The SFT checkpoint is first exported from LLaMA-Factory, then GRPO training runs with the sglang rollout backend and a rule-based MCQ reward via `mathruler.grader.grade_answer`.

Key hyperparameters:
- Rollout group size (`n`): 4 responses per prompt
- KL loss coefficient: 0.01 (`low_var_kl` type)
- Actor learning rate: 1e-6
- FSDP + sglang backend, 8× H100 80GB

```bash
# First export the SFT checkpoint
llamafactory-cli export training/stage2_sft/echo_sft.yaml --export_dir LLaMA-Factory/export_echo_sft

# Prepare parquet data for verl
python verl/examples/data_preprocess/echo_mcq.py --local_dir data/echo_mcq

# Run GRPO
bash training/stage3_grpo/run_echo_grpo.sh
bash training/stage3_grpo/run_cmr_grpo.sh
bash training/stage3_grpo/run_ecg_grpo.sh
```

### Run All Stages for One Modality

```bash
./training/run_training.sh ecg   # or echo / cmr
```

## Checkpoint Structure

Stage 1 & 2 checkpoints (LLaMA-Factory) are saved to `saves/Qwen2.5-VL-3B-Instruct/full/`:

```
LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/
├── ecg_sft/          # Stage 2 ECG (also used as Stage 1 base for ECG)
├── echo_sft/         # Stage 2 Echo
└── cmr_sft/          # Stage 2 CMR
```

Stage 3 GRPO checkpoints (verl) are saved to `verl/checkpoints/`:

```
verl/checkpoints/
├── marcus_ecg_grpo/qwen2_5_vl_3b_ecg_mcq/    # ECG deployment checkpoint
├── marcus_echo_grpo/qwen2_5_vl_3b_echo_mcq/  # Echo deployment checkpoint
└── marcus_cmr_grpo/qwen2_5_vl_3b_cmr_mcq/    # CMR deployment checkpoint
```

The verl GRPO checkpoints are what the CLI entry points
(`marcus-ecg`, `marcus-echo`, `marcus-cmr`) load by default.

## GRPO Reward Function

The reward function is in `training/stage3_grpo/reward.py`. It wraps verl's
`compute_score` interface and uses `mathruler.grader.grade_answer` for MCQ
answer matching. Rewards are binary: 1.0 (correct) or 0.0 (incorrect).

The function is registered in verl's reward score module via the `data_source`
field in the training parquet files (see `verl/verl/utils/reward_score/`).

To smoke-test the reward function:

```bash
python training/stage3_grpo/reward.py
```

## Monitoring

MARCUS training uses Weights & Biases for experiment tracking (when `wandb` is installed):

```bash
pip install wandb
wandb login
# Training will automatically log to your W&B project
```

## Reproducing Paper Results

To reproduce the MCQ accuracy and VQA Likert scores reported in the paper:

```bash
# 1. Download model weights
python scripts/download_checkpoints.py --model all

# 2. Start an expert server
marcus-ecg

# 3. Generate predictions on the benchmark
python scripts/run_inference_batch.py \
    --input data/benchmark/ecg_test.json \
    --modality ecg \
    --api-url http://localhost:8775 \
    --out predictions/ecg_marcus.json

# 4. Score with the judge
marcus-eval \
    --input predictions/ecg_marcus.json \
    --task mcq \
    --gt-key gt \
    --pred-key prediction \
    --out-dir eval_results/

# 5. Compute statistics
python scripts/compute_statistics.py \
    --predictions predictions/ecg_marcus.json \
    --baseline predictions/ecg_gpt5.json \
    --task mcq \
    --out-dir stats/ecg/
```
