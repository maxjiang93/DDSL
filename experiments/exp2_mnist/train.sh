export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python3 train.py -l log > training.log &
