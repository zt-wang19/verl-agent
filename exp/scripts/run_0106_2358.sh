#!/bin/bash

dataset="alfworld_spid_mixed_100steps"
model_path="/mnt/shared-storage-gpfs2/sfteval/lsz/spid/models/qwen2.5-7b-instruct"
exp_name="qwen7b_data-${dataset}"
checkpoints_dir="/mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/checkpoints/${exp_name}"

# 训练模型
sh /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/exp/scripts/llamafactory.sh \
    "${dataset}" \
    "${model_path}" \
    "${exp_name}"


# 运行 grpo
sh /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/exp/alfworld/run_grpo.sh "${checkpoints_dir}"

