#!/bin/bash
# 批量评测所有 webshop_grpo 模型
# 使用方法: bash eval_webshop_grpo_all.sh [num_iterations]
# 参数: num_iterations - 每个模型的验证迭代次数（默认3次）

set -e

cd /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent
. .venv-web/bin/activate

CHECKPOINTS_DIR=/mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/checkpoints
TEST_SCRIPT=/mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/exp/webshop/test_repeat.sh

# 验证迭代次数
NUM_ITERATIONS=${1:-3}

echo "========================================"
echo "Batch Evaluation for WebShop GRPO Models"
echo "========================================"
echo "Checkpoints directory: $CHECKPOINTS_DIR"
echo "Number of iterations per model: $NUM_ITERATIONS"
echo ""

# 查找所有包含 webshop_grpo 的目录
WEBSHOP_GRPO_DIRS=$(find "$CHECKPOINTS_DIR" -maxdepth 1 -type d -name "*webshop_grpo*" | sort)

if [ -z "$WEBSHOP_GRPO_DIRS" ]; then
    echo "No webshop_grpo directories found in $CHECKPOINTS_DIR"
    exit 1
fi

# 统计总数
TOTAL_MODELS=$(echo "$WEBSHOP_GRPO_DIRS" | wc -l)
echo "Found $TOTAL_MODELS webshop_grpo directories"
echo ""

# 收集所有要评测的 checkpoint 路径
declare -a CHECKPOINT_PATHS
declare -a CHECKPOINT_NAMES

for DIR in $WEBSHOP_GRPO_DIRS; do
    DIR_NAME=$(basename "$DIR")
    
    # 检查是否有 latest_checkpointed_iteration.txt
    LATEST_FILE="$DIR/latest_checkpointed_iteration.txt"
    
    if [ -f "$LATEST_FILE" ]; then
        # 读取最新的 global_step
        LATEST_STEP=$(cat "$LATEST_FILE")
        ACTOR_PATH="$DIR/global_step_${LATEST_STEP}/actor"
        
        if [ -d "$ACTOR_PATH" ]; then
            CHECKPOINT_PATHS+=("$ACTOR_PATH")
            CHECKPOINT_NAMES+=("${DIR_NAME}_step_${LATEST_STEP}")
            echo "  ✓ $DIR_NAME -> global_step_${LATEST_STEP}"
        else
            echo "  ✗ $DIR_NAME: actor directory not found at $ACTOR_PATH"
        fi
    else
        # 如果没有 latest_checkpointed_iteration.txt，找最大的 global_step
        LATEST_GLOBAL_STEP=$(find "$DIR" -maxdepth 1 -type d -name "global_step_*" | \
            sed 's|.*global_step_||' | sort -n | tail -1)
        
        if [ -n "$LATEST_GLOBAL_STEP" ]; then
            ACTOR_PATH="$DIR/global_step_${LATEST_GLOBAL_STEP}/actor"
            
            if [ -d "$ACTOR_PATH" ]; then
                CHECKPOINT_PATHS+=("$ACTOR_PATH")
                CHECKPOINT_NAMES+=("${DIR_NAME}_step_${LATEST_GLOBAL_STEP}")
                echo "  ✓ $DIR_NAME -> global_step_${LATEST_GLOBAL_STEP}"
            else
                echo "  ✗ $DIR_NAME: actor directory not found at $ACTOR_PATH"
            fi
        else
            echo "  ✗ $DIR_NAME: no global_step directories found"
        fi
    fi
done

echo ""
echo "========================================"
echo "Total checkpoints to evaluate: ${#CHECKPOINT_PATHS[@]}"
echo "========================================"
echo ""

# 列出所有要评测的模型
echo "Checkpoints to evaluate:"
for i in "${!CHECKPOINT_PATHS[@]}"; do
    echo "  [$((i+1))] ${CHECKPOINT_NAMES[$i]}"
    echo "      Path: ${CHECKPOINT_PATHS[$i]}"
done
echo ""

# 确认是否继续
# read -p "Do you want to start evaluation? (y/n): " -n 1 -r
# echo ""
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     echo "Evaluation cancelled."
#     exit 0
# fi

# 开始评测
echo ""
echo "========================================"
echo "Starting evaluation..."
echo "========================================"

SUCCESS_COUNT=0
FAIL_COUNT=0
declare -a FAILED_MODELS

for i in "${!CHECKPOINT_PATHS[@]}"; do
    CHECKPOINT_PATH="${CHECKPOINT_PATHS[$i]}"
    CHECKPOINT_NAME="${CHECKPOINT_NAMES[$i]}"
    
    echo ""
    echo "========================================"
    echo "[$((i+1))/${#CHECKPOINT_PATHS[@]}] Evaluating: $CHECKPOINT_NAME"
    echo "========================================"
    echo "Checkpoint: $CHECKPOINT_PATH"
    echo ""

    HF_PATH=$CHECKPOINT_PATH/huggingface

    echo "HF_PATH: $HF_PATH"

    bash /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/exp/merge_model.sh $CHECKPOINT_PATH
    
    bash "$TEST_SCRIPT" "$HF_PATH" "$NUM_ITERATIONS"
    # 运行评测脚本
    # if ; then
    #     echo ""
    #     echo "✓ Successfully evaluated: $CHECKPOINT_NAME"
    #     ((SUCCESS_COUNT++))
    # else
    #     echo ""
    #     echo "✗ Failed to evaluate: $CHECKPOINT_NAME"
    #     ((FAIL_COUNT++))
    #     FAILED_MODELS+=("$CHECKPOINT_NAME")
    # fi
    # exit
    
    echo ""
done

# 打印汇总
echo ""
echo "========================================"
echo "Batch Evaluation Completed!"
echo "========================================"
echo "Total models: ${#CHECKPOINT_PATHS[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  - $model"
    done
fi

echo ""
echo "Results saved to: $CHECKPOINTS_DIR/test_repeat_webshop/"
echo "========================================"

