
actor_paths=(
    "/mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/checkpoints/alfworld_grpo_qwen2.5-7b-instruct_aux_coef_0.1_20260106_190513/global_step_150/actor"
    "/mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/checkpoints/alfworld_grpo_qwen2.5-7b-instruct_aux_coef_0.5_20260106_191759/global_step_150/actor"
)
cd /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent
. .venv-alf/bin/activate

for actor_path in "${actor_paths[@]}"; do
    sh exp/merge_model.sh ${actor_path}
    sh exp/alfworld/test_repeat.sh ${actor_path}/huggingface
done