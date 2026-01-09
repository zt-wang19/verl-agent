#!/bin/bash
set -x
ENGINE=vllm
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export SWANLAB_MODE=local
export RAY_USAGE_STATS_ENABLED=0

cd /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export SWANLAB_PROJECT_NAME=search_grpo
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
export SWANLAB_EXP_NAME=search_grpo_${CHECKPOINT_NAME}_$TIMESTAMP

TOTAL_EPOCHS=1
ppo_micro_batch_size_per_gpu=32
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
ppo_mini_batch_size=$((NUM_GPUS * ppo_micro_batch_size_per_gpu))

output_dir=checkpoints/${SWANLAB_EXP_NAME}
mkdir -p $output_dir

# setup retriever
. .venv-retriever/bin/activate
bash examples/search/retriever/retrieval_launch.sh > $output_dir/retriever.log 2>&1 &
RETRIEVER_PID=$!

# 设置 trap 确保脚本退出时清理 retriever 进程
cleanup() {
    echo "Stopping retriever service (PID: $RETRIEVER_PID)..."
    kill $RETRIEVER_PID 2>/dev/null
    wait $RETRIEVER_PID 2>/dev/null
    echo "Retriever service stopped."
}
trap cleanup EXIT

# 等待 uvicorn 服务启动完成
echo "Waiting for retriever service to start..."
while ! grep -q "Uvicorn running" $output_dir/retriever.log 2>/dev/null; do
    if ! kill -0 $RETRIEVER_PID 2>/dev/null; then
        echo "Retriever process died unexpectedly!"
        cat $output_dir/retriever.log
        exit 1
    fi
    sleep 15
done
echo "Retriever service started successfully!"

train_data_size=512
val_data_size=512
group_size=5

TRAIN_DATA="data/searchR1_processed_direct/train.parquet"
VAL_DATA="data/searchR1_processed_direct/test.parquet"

. .venv-sea/bin/activate
ray start --head --include-dashboard=false

python -m exp.main \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url='http://127.0.0.1:8000/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$SWANLAB_PROJECT_NAME \
    trainer.experiment_name=$SWANLAB_EXP_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.validation_data_dir=$output_dir \
    trainer.default_local_dir=$output_dir \
    trainer.val_before_train=False \
    2>&1 | tee $output_dir/train.log

