#!/bin/bash
source activate
export CUDA_VISIBLE_DEVICES=2

python train.py \
--batch_size 48 \
--val_batch_size 48 \
--loss_type l1 \
--nlevels 5 \
--epochs 200 \
--lr 1e-3 \
--seed 1 \
--data_folder mres_processed_data \
--log_interval 50 \
--timestamp \
--feat 256 \
--dropout \
--transpose \
--smooth_loss 1 \
--multires \
--network 2 \
--log_dir logs/CLIP_net2_mres_sm1_l5_f256_dp