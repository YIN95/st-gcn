srun \
-o nohup_multistage_2.out \
--gres=gpu:4 \
--shm-size=1g \
--constrain=belegost run-docker \
--mount "src=/Midgard/home/wyin/.local,dst=/.local/" \
"src=/Midgard/home/wyin/,dst=/wyin/" \
"src=/Midgard/home/wyin/repo/st-gcn/,dst=/workspace" \
--image_name nvcr.io/nvidia/wyin_pytorch:19.01 \
--execute "python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_2.yaml"