#!/bin/bash

# create directory to store checkpoint
mkdir -p checkpoint

# download pretrained model
wget island.me.berkeley.edu/ddsl/checkpoint_polygonnet_best.pth.tar -P checkpoint
