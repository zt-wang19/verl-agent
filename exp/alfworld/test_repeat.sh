#!/bin/bash
# 在测试集上多次运行 validate 并计算平均 success_rate
# 使用方法: bash test_repeat.sh [checkpoint_path] [num_iterations]
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export SWANLAB_MODE=local
export ALFWORLD_DATA=/mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/data/alfworld
export RAY_USAGE_STATS_ENABLED=0
export SWANLAB_PROJECT_NAME=verl_agent_test
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent
DEFAULT_CHECKPOINT_PATH=/mnt/shared-storage-gpfs2/sfteval/lsz/spid/models/qwen2.5-1.5b-instruct
# NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
NUM_GPUS="${NUM_GPUS:-8}"
N_GPUS_PER_NODE=$NUM_GPUS
PPO_MICRO_BATCH_SIZE_PER_GPU=32
PPO_MINI_BATCH_SIZE=$((NUM_GPUS * PPO_MICRO_BATCH_SIZE_PER_GPU))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 参数1: checkpoint路径
if [ -n "$1" ]; then
    CHECKPOINT_PATH="$1"
else
    CHECKPOINT_PATH=/mnt/shared-storage-gpfs2/sfteval/lsz/spid/models/qwen2.5-1.5b-instruct
fi

# 参数2: validate迭代次数（默认5次）
if [ -n "$2" ]; then
    NUM_ITERATIONS="$2"
else
    NUM_ITERATIONS=3
fi

if [[ "$CHECKPOINT_PATH" == *"checkpoints"* ]]; then
    CHECKPOINT_NAME=$(echo "$CHECKPOINT_PATH" | sed 's|.*checkpoints/||' | tr '/' '_')
else
    CHECKPOINT_NAME=$(echo "$CHECKPOINT_PATH" | sed 's|.*models/||' | tr '/' '_')
fi
echo "Testing checkpoint: ${CHECKPOINT_PATH}"
echo "Checkpoint name: ${CHECKPOINT_NAME}"
echo "Number of iterations: ${NUM_ITERATIONS}"

export SWANLAB_EXP_NAME=alfworld_repeat_${CHECKPOINT_NAME}_test
output_dir=checkpoints/test_repeat/${CHECKPOINT_NAME}
mkdir -p $output_dir
num_cpus_per_env_worker=0.1
data_path=data/alfworld

# 测试集配置
# 140 games in split=eval_in_distribution
train_data_size=8  # 最小训练数据量（不会实际使用）
val_data_size=140  # 测试集大小
group_size=1

. .venv-alf/bin/activate
# 准备数据
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    --local_dir $data_path

# 使用自定义的 ALFWorld main，支持 repeated validation
python3 -m exp.main \
    algorithm.adv_estimator=gigpo \
    data.train_files=${data_path}/text/train.parquet \
    data.val_files=${data_path}/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$CHECKPOINT_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.gamma=0.95 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.history_length=2 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$SWANLAB_PROJECT_NAME \
    trainer.experiment_name=$SWANLAB_EXP_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.validation_data_dir=$output_dir \
    trainer.default_local_dir=$output_dir \
    +trainer.num_val_iterations=$NUM_ITERATIONS \
    2>&1 | tee $output_dir/test_repeat.log

echo ""
echo "========================================"
echo "Repeated testing completed!"
echo "Output directory: $output_dir"
echo "Log file: $output_dir/test_repeat.log"
echo "========================================"

