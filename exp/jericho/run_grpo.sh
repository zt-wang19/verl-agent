#!/bin/bash
set -x
ENGINE=vllm
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export SWANLAB_MODE=local
export RAY_USAGE_STATS_ENABLED=0

cd /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent
export CUDA_VISIBLE_DEVICES=0,1

export SWANLAB_PROJECT_NAME=jericho_grpo
TIMESTAMP=$(date +%Y%m%d_%H%M%S)


TOTAL_EPOCHS=150
ppo_micro_batch_size_per_gpu=8

if [ -n "$1" ]; then
    model_path="$1"
else
    model_path=/mnt/shared-storage-gpfs2/sfteval/lsz/spid/models/qwen2.5-1.5b-instruct
fi
if [[ "$model_path" == *"checkpoints"* ]]; then
    CHECKPOINT_NAME=$(echo "$model_path" | sed 's|.*checkpoints/||' | tr '/' '_')
else
    CHECKPOINT_NAME=$(echo "$model_path" | sed 's|.*models/||' | tr '/' '_')
fi

JERICHO_GAME=${JERICHO_GAME:-"all"}
# Path to ROM files directory (should contain .z5/.z8 files)
JERICHO_ROM_PATH=${JERICHO_ROM_PATH:-"/mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/data/jericho/z-machine-games-master/jericho-game-suite"}
sp_coef="${SP_COEF:-0.1}"
id_coef="${ID_COEF:-0.1}"
export SWANLAB_EXP_NAME=jericho_grpo_${JERICHO_GAME}_${CHECKPOINT_NAME}_sp_${sp_coef}_id_${id_coef}_$TIMESTAMP

# NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
NUM_GPUS="${NUM_GPUS:-8}"
ppo_mini_batch_size=$((NUM_GPUS * ppo_micro_batch_size_per_gpu))

output_dir=checkpoints/${SWANLAB_EXP_NAME}
mkdir -p $output_dir

num_cpus_per_env_worker=0.1
train_data_size=2
val_data_size=64
group_size=8
data_path=data/jericho
max_prompt_length=4096
max_response_length=512

# Activate environment - you need to install jericho in your venv
# pip install jericho
# python -m spacy download en_core_web_sm
. .venv-alf/bin/activate

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
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
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
    env.env_name=jericho \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.history_length=2 \
    +env.jericho.game_name=$JERICHO_GAME \
    +env.jericho.rom_path=$JERICHO_ROM_PATH \
    +env.jericho.env_step_limit=30 \
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
    +actor_rollout_ref.actor.sp_coef=$sp_coef \
    +actor_rollout_ref.actor.id_coef=$id_coef \
    +actor_rollout_ref.actor.debug_dir=$output_dir \
    +actor_rollout_ref.actor.aux_max_length=$((max_prompt_length + max_response_length)) \
    +actor_rollout_ref.actor.aux_history_length=2 \
    +actor_rollout_ref.actor.aux_micro_batch_size=8 \
    2>&1 | tee $output_dir/train.log

# actor_path=$output_dir/global_step_${TOTAL_EPOCHS}/actor
# sh exp/merge_model.sh ${actor_path}
# NUM_GPUS=${NUM_GPUS} bash exp/jericho/test_repeat.sh ${actor_path}/huggingface


