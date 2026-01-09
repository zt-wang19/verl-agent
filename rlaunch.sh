
# rlaunch --gpu=8 --memory=1600000 --cpu=140 \
#     --charged-group=ptdata_gpu \
#     --private-machine=yes \
#     --image registry.h.pjlab.org.cn/ailab-puyuvlm-puyuvlm_gpu/xtuner:glx-b0cd52bf8c711b71e66573c3cb0307a2a737e757-20251027 \
#     --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
#     --mount=gpfs://gpfs1/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-8B-Instruct:/mnt/shared-storage-user/models--Qwen--Qwen3-VL-8B-Instruct \
#     --mount=gpfs://gpfs2/sfteval/datasets/MMPR:/mnt/shared-storage-user/MMPR \
#     --custom-resources brainpp.cn/fuse=1 \
#     -- bash
rlaunch --gpu=8 --memory=1280000 --cpu=128 \
    --charged-group=ptdata_gpu \
    --private-machine=yes \
    --image=registry.h.pjlab.org.cn/ailab-ptdata-ptdata_gpu/lsz:lsz-ninja-20260104143512 \
    --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
    --mount=gpfs://gpfs2/sfteval/lsz/spid:/mnt/shared-storage-gpfs2/sfteval/lsz/spid \
    --custom-resources brainpp.cn/fuse=1 \
    --i-know-i-am-wasting-resource=true \
    -- bash

rlaunch --gpu=2 --memory=400000 --cpu=35 \
    --charged-group=ptdata_gpu \
    --private-machine=yes \
    --image=registry.h.pjlab.org.cn/ailab-ptdata-ptdata_gpu/lsz:lsz-ninja-20260104143512 \
    --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
    --mount=gpfs://gpfs2/sfteval/lsz/spid:/mnt/shared-storage-gpfs2/sfteval/lsz/spid \
    --custom-resources brainpp.cn/fuse=1 \
    --negative-tags node/gpu-lg-cmc-h-h200-2287.host.h.pjlab.org.cn \
    --i-know-i-am-wasting-resource=true \
    -- bash

# 别人的卡
rjob submit \
    --name=spid-test-mix \
    --gpu=8 --memory=1000000 --cpu=160 \
    --task-type=idle \
    --private-machine=no \
    --image=registry.h.pjlab.org.cn/ailab-ptdata-ptdata_gpu/lsz:lsz-20251221195550 \
    --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
    --mount=gpfs://gpfs2/sfteval/lsz/spid:/mnt/shared-storage-gpfs2/sfteval/lsz/spid \
    --custom-resources brainpp.cn/fuse=1 \
    -- bash -c 'sleep 360000'

#8卡
rjob submit \
    --name=spid-test \
    --gpu=8 --memory=1000000 --cpu=160 \
    --private-machine=group --charged-group=ptdata_gpu \
    --image=registry.h.pjlab.org.cn/ailab-ptdata-ptdata_gpu/lsz:lsz-tmpdir-jdk-20251222192224 \
    --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
    --mount=gpfs://gpfs2/sfteval/lsz/spid:/mnt/shared-storage-gpfs2/sfteval/lsz/spid \
    --custom-resources brainpp.cn/fuse=1 \
    -- bash -c 'sleep 360000'

# 2卡
rjob submit \
    --name=spid-test-num2 \
    --gpu=2 --memory=1000000 --cpu=160 \
    --private-machine=group --charged-group=ptdata_gpu \
    --image=registry.h.pjlab.org.cn/ailab-ptdata-ptdata_gpu/lsz:lsz-20251221195550 \
    --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
    --mount=gpfs://gpfs2/sfteval/lsz/spid:/mnt/shared-storage-gpfs2/sfteval/lsz/spid \
    --custom-resources brainpp.cn/fuse=1 \
    -- bash -c 'sleep 360000'

# rlaunch --gpu=8 --memory=1600000 --cpu=140 \
#     --charged-group=ptdata_gpu \
#     --private-machine=yes \
#     --image registry.h.pjlab.org.cn/ailab-puyuvlm-puyuvlm_gpu/xtuner:glx-b0cd52bf8c711b71e66573c3cb0307a2a737e757-20251027 \
#     --mount=gpfs://gpfs1/intern-multi-modal-delivery:/mnt/shared-storage-user/intern-multi-modal-delivery/ \
#     --mount=gpfs://gpfs1/puyullmgpu-shared:/mnt/shared-storage-user/puyullmgpu-shared \
#     --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared \
#     --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
#     --mount=gpfs://gpfs1/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-8B-Instruct:/mnt/shared-storage-user/models--Qwen--Qwen3-VL-8B-Instruct \
#     --mount=gpfs://gpfs2/sfteval/datasets/MMPR:/mnt/shared-storage-user/MMPR \
#     --custom-resources brainpp.cn/fuse=1 \
#     -- bash

# export AWS_ACCESS_KEY_ID=FB7QKWTWP279SQMLBX4H
# export AWS_SECRET_ACCESS_KEY=dN6ph2f9cQcVhnOCngiGKwPUjMqpM9o4oiKM67mb
# ./s3mount public-dataset-p2 ~/data --endpoint-url http://p-ceph-norm-inside.pjlab.org.cn --force-path-style --allow-delete --allow-overwrite

# export AWS_ACCESS_KEY_ID=HES1ITGIVHJ36GHYMTV4
# export AWS_SECRET_ACCESS_KEY=G2FKmfJxKCTkfyICOD4GsT08x8wUNCbsa17OmNQM
# ./s3mount VideoMME /mnt/shared-storage-user/lisongze/mount/videomme/ --endpoint-url http://p-ceph-norm-inside.pjlab.org.cn --force-path-style --allow-delete --allow-overwrite

# export AWS_ACCESS_KEY_ID=o0asnixdrdvctn7pdpjb
# export AWS_SECRET_ACCESS_KEY=ul34evliiq6929jbvby3x4w1xxb6fp8xh0gqo3pg
# ./s3mount heyinan /mnt/shared-storage-user/lisongze/mount/heyinan/ --endpoint-url http://hdd1.h.pjlab.org.cn:8060 --force-path-style --allow-delete --allow-overwrite

export AWS_ACCESS_KEY_ID=o0asnixdrdvctn7pdpjb
export AWS_SECRET_ACCESS_KEY=ul34evliiq6929jbvby3x4w1xxb6fp8xh0gqo3pg
proxy_off
cd /mnt/shared-storage-user/lisongze/
./s3mount lisongze /mnt/shared-storage-user/lisongze/cache/ --prefix cache/ --endpoint-url http://hdd1.h.pjlab.org.cn:8060 --force-path-style --allow-delete --allow-overwrite

# umount -l -f {your-mountpoint} #取消挂载

# aws-hceph s3 ls s3://lisongze/cache/ --recursive --human-readable --summarize 

# # rclone mount pnorm:VideoMME  \
# #   --vfs-cache-mode off \
# #   --buffer-size 1G \
# #   --allow-other \
# #   --umask 000 \
# #   --daemon

# rjob submit \
#         --name=sft-internvl35-8b-tiny \
#         --gpu=8 --memory=1200000 --charged-group=ptdata_gpu --cpu=128 \
#         --private-machine=group \
#         -P 4 \
#         --image registry.h.pjlab.org.cn/ailab-puyuvlm-puyuvlm_gpu/xtuner:glx-b0cd52bf8c711b71e66573c3cb0307a2a737e757-20251027 \
#         --mount=gpfs://gpfs1/intern-multi-modal-delivery:/mnt/shared-storage-user/intern-multi-modal-delivery/ \
#         --mount=gpfs://gpfs1/puyullmgpu-shared:/mnt/shared-storage-user/puyullmgpu-shared \
#         --mount=gpfs://gpfs1/lisongze:/mnt/shared-storage-user/lisongze \
#         --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared \
#         --host-network=true \
#         --gang-start=true \
#         --custom-resources rdma/mlnx_shared=8  \
#         --custom-resources mellanox.com/mlnx_rdma=1 \
#         -e DISTRIBUTED_JOB=true \
#         -- bash -c '
#         cd /workspace/xtuner
#         bash scripts/sft_intern_s1_vl_entrypoint.sh path_to_sft_internvl3.5_8B_config_tiny.py'