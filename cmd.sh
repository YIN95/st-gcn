#!/usr/bin/env bash

nvidia-docker run --shm-size=1g --rm -it -u $(id -u):$(id -g) \
--mount type=bind,source=/Midgard/home/wyin/.local,target=/.local \
--mount type=bind,source=/Midgard/home/wyin/,target=/wyin \
--mount type=bind,source=/Midgard/home/wyin/repo/st-gcn/,target=/st-gcn \
nvcr.io/nvidia/wyin_pytorch:19.01-100

# srun --gres=gpu:4 run-docker --cinstrain=belegost --shm-size=32g \
# --mount "src=/Midgard/home/wyin/.local,dst=/.local" \
# "src=/Midgard/home/wyin/,dst=/wyin" \
# "src=/Midgard/home/wyin/repo/st-gcn/,dst=/st-gcn" \
# "src=/Midgard/home/wyin/repo/st-gcn/config/,dst=/config" \
# "src=/Midgard/home/wyin/repo/st-gcn/data/,dst=/data" \
# "src=/Midgard/home/wyin/repo/st-gcn/feeder/,dst=/feeder" \
# "src=/Midgard/home/wyin/repo/st-gcn/models/,dst=/models" \
# "src=/Midgard/home/wyin/repo/st-gcn/net/,dst=/net" \
# "src=/Midgard/home/wyin/repo/st-gcn/processor/,dst=/processor" \
# "src=/Midgard/home/wyin/repo/st-gcn/resource/,dst=/resource" \
# "src=/Midgard/home/wyin/repo/st-gcn/tools/,dst=/tools" \
# "src=/Midgard/home/wyin/repo/st-gcn/work_dir/,dst=/work_dir" \
# --image_name nvcr.io/nvidia/wyin_pytorch:19.01-100 \
# --execute "sh /st-gcn/fivefold.sh" &

