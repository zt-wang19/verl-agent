# /mnt/shared-storage-gpfs2/sfteval/lsz/spid/models/qwen2.5-7b-instruct
for dataset in webshop_spid_state_prediction_only_100steps webshop_spid_inverse_dynamics_only_100steps; do
    sh /home/test/test06/wzt/code/verl-agent/exp/scripts/llamafactory.sh \
    $dataset \
    /home/test/test06/wzt/models/qwen2.5-7b-instruct \
    qwen7b_data-${dataset}

    sh /home/test/test06/wzt/code/verl-agent/exp/alfworld/run_grpo_7b.sh /home/test/test06/wzt/code/verl-agent/checkpoints/qwen7b_data-${dataset}
done