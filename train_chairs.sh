#!/bin/bash
mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=0,1 python train_flow.py --name raft-chairs \
                                              --stage chairs \
                                              --validation chairs \
                                              --num_steps 100000 \
                                              --batch_size 10 \
                                              --lr 0.00025 \
                                              --image_size 368 496 \
                                              --wdecay 0.0001 \
                                              --outpath checkpoints_chairs \
                                              --mixed_precision
                                       