actor_path=$1
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir $actor_path \
    --target_dir $actor_path/huggingface