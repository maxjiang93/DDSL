#!/bin/bash
source activate
export CUDA_VISIBLE_DEVICES=2

python train.py \
--batch_size 48 \
--val_batch_size 48 \
--loss_type l1 \
--epochs 200 \
--lr 1e-3 \
--seed 1 \
--data_folder processed_data \
--log_interval 20 \
--log_dir logs/net2_new_new \
--timestamp
