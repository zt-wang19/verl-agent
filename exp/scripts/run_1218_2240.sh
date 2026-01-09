# /home/test/test06/wzt/models/qwen2.5-7b-instruct
for dataset in alfworld_spid_mixed_1000steps; do
    # sh /home/test/test06/wzt/code/verl-agent/exp/scripts/llamafactory.sh \
    # $dataset \
    # /home/test/test06/wzt/models/qwen2.5-1.5b-instruct \
    # qwen1.5b_data-${dataset}

    sh /home/test/test06/wzt/code/verl-agent/exp/alfworld/run_grpo.sh /home/test/test06/wzt/code/verl-agent/checkpoints/qwen1.5b_data-${dataset}
done