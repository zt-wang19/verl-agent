#!/bin/bash
set -x

# 观察预测模型训练脚本
# 使用 LLaMA-Factory 进行全参数微调

# 设置环境变量
export DISABLE_VERSION_CHECK=1
export WANDB_DISABLED=true
export SWANLAB_MODE=disabled

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据实际 GPU 数量调整
# export CUDA_VISIBLE_DEVICES=1,3,4,6
# export PYTHONPATH=/home/test/test06/wzt/code/verl-agent:$PYTHONPATH

cd /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent
. .venv-lf/bin/activate

data=$1
MODEL_PATH=$2
exp_name=$3

echo "当前数据集: $data"
echo $SWANLAB_EXP_NAME

export SWANLAB_PROJ_NAME=spid_llamafactory
# export SWANLAB_API_KEY=gProQuNeK5qKjjDKq87sa
export SWANLAB_EXP_NAME=${exp_name}

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
GRADIENT_ACCUMULATION_STEPS=$((32 / $NUM_GPUS))
lr=5e-7

# 训练参数
# MODEL_PATH="/home/test/test06/wzt/models/qwen2.5-1.5b-instruct"
DATASET_DIR="data/exp"
OUTPUT_DIR="checkpoints/$SWANLAB_EXP_NAME"
DEEPSPEED_CONFIG="exp/ds_z3_config.json"
echo $OUTPUT_DIR

# 使用激活后的环境调用
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path ${MODEL_PATH} \
    --dataset ${data} \
    --dataset_dir ${DATASET_DIR} \
    --template qwen \
    --finetuning_type full \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --overwrite_cache \
    --cutoff_len 4608 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $lr \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --save_steps 500 \
    --plot_loss \
    --val_size 0.1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 500 \
    --use_swanlab \
    --logging_steps 10 \
    --bf16 \
    --ddp_timeout 180000000




