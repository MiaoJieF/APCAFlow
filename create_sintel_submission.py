from __future__ import print_function, division
import sys
sys.path.append('core')
import numpy as np
import torch
import torch.nn as nn
# import models
import cv2
import time
from core.apcaflow import APCAFlow
# from core.raft_agg import RAFT
# from core.raft_small import RAFT
# from core.raft_small_v3 import RAFT
import argparse
import evaluate


# CUDA_VISIBLE_DEVICES=0 python create_sintel_submission.py


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true', help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    args = parser.parse_args()

    model = nn.DataParallel(APCAFlow(args))
    checkpoints_path = 'checkpoints_sintel/raft-lc.pth'
    if checkpoints_path is not None:
        model.load_state_dict(torch.load(checkpoints_path), strict=True)
    model.cuda()
    model.eval()

    evaluate.create_sintel_submission(model.module)


