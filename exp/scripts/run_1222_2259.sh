# /home/test/test06/wzt/models/qwen2.5-7b-instruct
cd /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent
for dataset in webshop_spid_inverse_dynamics_only_100steps webshop_spid_state_prediction_only_100steps; do
    sh exp/scripts/llamafactory.sh \
    $dataset \
    /mnt/shared-storage-gpfs2/sfteval/lsz/spid/models/qwen2.5-1.5b-instruct \
    qwen1.5b_data-${dataset}

    sh exp/webshop/run_grpo.sh \
        checkpoints/qwen1.5b_data-${dataset}
done