#!/usr/bin/env bash
# MARCUS — Stage 3: CMR GRPO via verl
#
# Runs Group Relative Policy Optimization on the CMR expert model.
# Starts from the Stage 2 SFT checkpoint exported from LLaMA-Factory.
#
# Export the SFT checkpoint before running:
#   llamafactory-cli export training/stage2_sft/cmr_sft.yaml \
#       --export_dir LLaMA-Factory/export_cmr_sft
#
# Data preparation:
#   python verl/examples/data_preprocess/cmr_mcq.py \
#       --local_dir data/cmr_mcq
#
# Usage:
#   bash training/stage3_grpo/run_cmr_grpo.sh
#
# See: O'Sullivan et al., 2026 — MARCUS preprint.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-LLaMA-Factory/export_cmr_sft}"
TRAIN_FILES="${TRAIN_FILES:-data/cmr_mcq/train.parquet}"
VAL_FILES="${VAL_FILES:-data/cmr_mcq/test.parquet}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length=5000 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.video_key=videos \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.mode=sync \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=marcus_cmr_grpo \
    trainer.experiment_name=qwen2_5_vl_3b_cmr_mcq \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=250 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    +trainer.rollout_dump_freq=2 \
    +trainer.rollout_data_dir=./cmr_mcq_rollout_data \
    +trainer.validation_data_dir=./cmr_mcq_validation_data \
    "$@"
