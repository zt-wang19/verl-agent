#!/bin/bash
set -x
ENGINE=vllm
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export SWANLAB_MODE=local
export RAY_USAGE_STATS_ENABLED=0

cd /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent
export CUDA_VISIBLE_DEVICES=0,1

export SWANLAB_PROJECT_NAME=scienceworld_grpo
TIMESTAMP=$(date +%Y%m%d_%H%M%S)


if [ -n "$1" ]; then
    model_path="$1"
else
    model_path=/mnt/shared-storage-gpfs2/sfteval/lsz/spid/models/qwen2.5-7b-instruct
fi
if [[ "$model_path" == *"checkpoints"* ]]; then
    CHECKPOINT_NAME=$(echo "$model_path" | sed 's|.*checkpoints/||' | tr '/' '_')
else
    CHECKPOINT_NAME=$(echo "$model_path" | sed 's|.*models/||' | tr '/' '_')
fi

# ScienceWorld task configuration
# Available tasks: boil, melt, freeze, change-the-state-of-matter-of, use-thermometer, 
#                  measure-melting-point-known-substance, measure-melting-point-unknown-substance,
#                  power-component, power-component-renewable-vs-nonrenewable-energy, test-conductivity,
#                  test-conductivity-of-unknown-substances, find-living-thing, find-non-living-thing,
#                  find-plant, find-animal, grow-plant, grow-fruit, chemistry-mix, 
#                  chemistry-mix-paint-secondary-color, chemistry-mix-paint-tertiary-color,
#                  lifespan-longest-lived, lifespan-shortest-lived, lifespan-longest-lived-then-shortest-lived,
#                  identify-life-stages-1, identify-life-stages-2, inclined-plane-determine-angle,
#                  inclined-plane-friction-named-surfaces, inclined-plane-friction-unnamed-surfaces,
#                  mendelian-genetics-known-plant, mendelian-genetics-unknown-plant
#
# Special value: "all" - Randomly select task at each reset, weighted by variation counts.
#                        Training uses train split, validation uses test split.
SCIENCEWORLD_TASK=${SCIENCEWORLD_TASK:-"all"}
export SWANLAB_EXP_NAME=scienceworld_grpo_${SCIENCEWORLD_TASK}_${CHECKPOINT_NAME}_$TIMESTAMP

TOTAL_EPOCHS=150
ppo_micro_batch_size_per_gpu=8
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
ppo_mini_batch_size=$((NUM_GPUS * ppo_micro_batch_size_per_gpu))

output_dir=checkpoints/${SWANLAB_EXP_NAME}
mkdir -p $output_dir

num_cpus_per_env_worker=0.1  # ScienceWorld needs more CPU due to JVM
train_data_size=2
val_data_size=64
group_size=8
data_path=data/scienceworld

# Activate environment - you may need to install scienceworld in your venv
# pip install scienceworld
. .venv-scw/bin/activate

ray stop; ray start --head --include-dashboard=false

# Data preparation
python -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    --local_dir $data_path

# Training
python -m exp.main \
    algorithm.adv_estimator=grpo \
    data.train_files=${data_path}/text/train.parquet \
    data.val_files=${data_path}/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=scienceworld \
    env.seed=0 \
    env.max_steps=100 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.history_length=2 \
    env.scienceworld.task_name=$SCIENCEWORLD_TASK \
    env.scienceworld.simplification_str=easy \
    env.scienceworld.env_step_limit=100 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$SWANLAB_PROJECT_NAME \
    trainer.experiment_name=$SWANLAB_EXP_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.validation_data_dir=$output_dir \
    trainer.default_local_dir=$output_dir \
    trainer.val_before_train=False \
    +trainer.num_val_iterations=2 \
    2>&1 | tee $output_dir/train.log

# actor_path=$output_dir/global_step_${TOTAL_EPOCHS}/actor
# sh exp/merge_model.sh ${actor_path}

