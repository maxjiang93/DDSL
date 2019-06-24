#!/bin/bash

python train.py \
--batch_size=20 \
--loss_type="l1" \
--nlevels=3 \
--feat=256 \
--epochs=100 \
--lr=1.2e-3 \
--decay \
--decay_step=10 \
--decay_rate=0.9 \
--alpha_raster=1.0 \
--alpha_chamfer=1.0 \
--alpha_laplacian=0.1 \
--alpha_edge=0.1 \
--data_folder="data" \
--mesh_folder="mesh_files" \
--log_interval=10 \
--log_dir="logs/m2r1c1l.1e.1_l4_decay1.2" \
--timestamp \
--workers=12 \
--n_tgt_pts=2048 \
--n_gen_pts=2048 \
--model2 \
--maxedge
