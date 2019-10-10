#!/usr/bin/env bash

nvidia-docker run --shm-size=1g --rm -it -u $(id -u):$(id -g) \
--mount type=bind,source=/Midgard/home/wyin/.local,target=/.local \
--mount type=bind,source=/Midgard/home/wyin/,target=/wyin \
--mount type=bind,source=/Midgard/home/wyin/repo/st-gcn/,target=/st-gcn \
nvcr.io/nvidia/pytorch:19.08
