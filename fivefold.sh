
# nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_0.yaml > nohup0.out

# nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_1.yaml > nohup1.out

nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_2.yaml > nohup2.out

nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_3.yaml > nohup3.out

nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_4.yaml > nohup4.out

python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_test.yaml

nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_2.yaml > nohup_group2.out
nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_4.yaml > nohup_group4.out


nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_0.yaml > nohup_multistage_0.out
nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_1.yaml > nohup_multistage_1.out
nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_2.yaml > nohup_multistage_2.out
nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_3.yaml > nohup_multistage_3.out
nohup python main.py recognition -c config/st_gcn.twostream/congreg8-marker/train_4.yaml > nohup_multistage_4.out
