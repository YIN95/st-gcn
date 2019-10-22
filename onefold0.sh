srun \
-o nohup_disori_0.out \
--gres=gpu:4 \
--constrain=belegost run-docker \
--mount "src=/Midgard/home/wyin/.local,dst=/.local/" \
"src=/Midgard/home/wyin/,dst=/wyin/" \
"src=/Midgard/home/wyin/repo/st-gcn/,dst=/workspace" \
--image_name nvcr.io/nvidia/wyin_pytorch:19.01 \
--execute "python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_0.yaml"