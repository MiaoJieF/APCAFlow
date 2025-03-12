#!/bin/bash
mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=0,1 python train_flow.py --name raft-things \
									                            --stage things \
									                            --validation sintel \
									                            --outpath checkpoints_things \
									                            --restore_ckpt checkpoints_chairs/raft-chairs.pth \
									                            --num_steps 150000 \
									                            --batch_size 6 \
									                            --lr 0.000175 \
									                            --image_size 400 720 \
									                            --wdecay 0.0001 \
									                            --mixed_precision
									   