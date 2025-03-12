#!/bin/bash
mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=0,1 python train_flow.py --name raft-sintel \
                                              --stage sintel \
                                              --validation sintel \
                                              --outpath checkpoints_sintel \
                                              --restore_ckpt checkpoints_things/raft-things.pth \
                                              --num_steps 120000 \
                                              --batch_size 6 \
                                              --lr 0.000125 \
                                              --image_size 368 768 \
                                              --wdecay 0.00001 \
                                              --gamma=0.85 \
                                              --mixed_precision
