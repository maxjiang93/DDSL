#!/bin/bash

python train.py \
--batch_size=20 \
--loss_type="l1" \
--nlevels=3 \
--feat=256 \
--epochs=100 \
--lr=5e-3 \
--alpha_raster=1.0 \
--alpha_chamfer=1.0 \
--data_folder="data" \
--mesh_folder="mesh_files" \
--log_interval=1 \
--log_dir="logs/m2r1c1" \
--timestamp \
--workers=12 \
--n_tgt_pts=2048 \
--n_gen_pts=2048 \
--model2
