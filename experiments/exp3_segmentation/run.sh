#!/bin/bash
source activate
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
--batch_size 256 \
--val_batch_size 256 \
--loss_type l1 \
--nlevels 5 \
--epochs 200 \
--lr 1e-3 \
--seed 1 \
--data_folder mres_processed_data \
--log_interval 1 \
--timestamp \
--feat 256 \
--dropout \
--transpose \
--smooth_loss 1.0 \
--multires \
--log_dir logs/mres_sm1_l5_f256_dp_4gpu