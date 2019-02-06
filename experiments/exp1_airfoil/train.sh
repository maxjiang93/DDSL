export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python3 train_AFNet.py --batch-size 110 --lr 0.01 --log-interval 50 --epochs 500 --decay --bottleneck 1000 --log-dir log > training.log &
