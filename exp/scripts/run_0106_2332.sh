#!/bin/bash

cd /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent
actor_paths=(
    "/mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/checkpoints/alfworld_grpo_checkpoints_qwen7b_data-alfworld_spid_inverse_dynamics_only_100steps_20251226_091854/global_step_150/actor/huggingface"
    "/mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/checkpoints/alfworld_grpo_checkpoints_qwen7b_data-alfworld_spid_state_prediction_only_100steps_20251225_175328/global_step_150/actor/huggingface"
)

for actor_path in "${actor_paths[@]}"; do
    sh exp/alfworld/test_repeat.sh "$actor_path"
done

