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
--smooth_loss 0.1 \
--multires \
--network 3 \
--uniform_loss \
--resume logs/RESUME_net3_mres_sm1_l4_f256_dp_2019_01_21_10_30_57/checkpoint_polygonnet_199.pth.tar \
--log_dir logs/RESUME_uni_ssl_net3_mres_sm1_l4_f256_dp_gpu2
