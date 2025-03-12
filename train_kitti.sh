#!/bin/bash
mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=0,1 python train_flow.py --name raft-kitti \
                                              --stage kitti \
                                              --validation kitti \
                                              --outpath checkpoints_kitti \
                                              --restore_ckpt checkpoints_sintel/raft-sintel.pth \
                                              --num_steps 30000 \
                                              --batch_size 6 \
                                              --lr 0.0001 \
                                              --image_size 288 960 \
                                              --wdecay 0.00001 \
                                              --gamma=0.85 \
                                              --mixed_precision
