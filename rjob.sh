# /mnt/shared-storage-gpfs2/sfteval/lsz/spid/models/qwen2.5-1.5b-instruct
# /mnt/shared-storage-gpfs2/sfteval/lsz/spid/models/qwen2.5-7b-instruct
rjob submit \
    --name=spid-webshop-7b-sp0id02 \
    --gpu=8 --memory=1280000 --cpu=128 \
    --private-machine=group --charged-group=ptdata_gpu \
    --image=registry.h.pjlab.org.cn/ailab-ptdata-ptdata_gpu/lsz:lsz-ninja-20260104143512 \
    --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
    --mount=gpfs://gpfs2/sfteval/lsz/spid:/mnt/shared-storage-gpfs2/sfteval/lsz/spid \
    --custom-resources brainpp.cn/fuse=1 \
    --negative-tags node/gpu-lg-cmc-h-h200-2287.host.h.pjlab.org.cn \
    -- bash -c 'NUM_GPUS=8 SP_COEF=0 ID_COEF=0.2 bash /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/exp/webshop/run_grpo.sh /mnt/shared-storage-gpfs2/sfteval/lsz/spid/models/qwen2.5-7b-instruct'



rjob submit \
    --name=spid-webshop \
    --gpu=2 --memory=400000 --cpu=40 \
    --private-machine=group --charged-group=ptdata_gpu \
    --image=registry.h.pjlab.org.cn/ailab-ptdata-ptdata_gpu/lsz:lsz-tmpdir-jdk-20251222192224 \
    --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
    --mount=gpfs://gpfs2/sfteval/lsz/spid:/mnt/shared-storage-gpfs2/sfteval/lsz/spid \
    --custom-resources brainpp.cn/fuse=1 \
    -- bash -c 'NUM_GPUS=2 SP_COEF=0 ID_COEF=0 bash /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/exp/webshop/run_grpo.sh'

rjob submit \
    --name=spid-alfworld-sp02id0 \
    --gpu=4 --memory=800000 --cpu=80 \
    --private-machine=group --charged-group=ptdata_gpu \
    --image=registry.h.pjlab.org.cn/ailab-ptdata-ptdata_gpu/lsz:lsz-tmpdir-jdk-20251222192224 \
    --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
    --mount=gpfs://gpfs2/sfteval/lsz/spid:/mnt/shared-storage-gpfs2/sfteval/lsz/spid \
    --custom-resources brainpp.cn/fuse=1 \
    -- bash -c 'NUM_GPUS=4 SP_COEF=0.2 ID_COEF=0 bash /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/exp/alfworld/run_grpo.sh'

rjob submit \
    --name=spid-webshop \
    --gpu=8 --memory=1000000 --cpu=160 \
    --private-machine=group --charged-group=ptdata_gpu \
    --image=registry.h.pjlab.org.cn/ailab-ptdata-ptdata_gpu/lsz:lsz-tmpdir-jdk-20251222192224 \
    --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
    --mount=gpfs://gpfs2/sfteval/lsz/spid:/mnt/shared-storage-gpfs2/sfteval/lsz/spid \
    --custom-resources brainpp.cn/fuse=1 \
    -- bash -c 'sh /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/exp/alfworld/test_repeat.sh /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/checkpoints/alfworld_grpo_checkpoints_qwen7b_data-alfworld_spid_inverse_dynamics_only_100steps_20251226_091854/global_step_150/actor/huggingface
    sh /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/exp/alfworld/test_repeat.sh /mnt/shared-storage-gpfs2/sfteval/lsz/spid/verl-agent/checkpoints/alfworld_grpo_checkpoints_qwen7b_data-alfworld_spid_state_prediction_only_100steps_20251225_175328/global_step_150/actor/huggingface'