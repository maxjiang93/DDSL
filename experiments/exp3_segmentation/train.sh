#!/bin/bash
source activate pytorch1
export CUDA_VISIBLE_DEVICES=2

python train.py \
--batch_size 48 \
--val_batch_size 48 \
--loss_type l1 \
--nlevels 4 \
--epochs 300 \
--lr 1e-3 \
--seed 1 \
--data_folder mres_processed_data \
--log_interval 50 \
--timestamp \
--feat 256 \
--dropout \
--smooth_loss 1 \
--multires \
--log_dir logs/run1
