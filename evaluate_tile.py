import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import torch.nn as nn

from core import datasets
from core.utils import flow_viz
from core.utils import frame_utils

# from raft import RAFT
from core.apcaflow import APCAFlow
from core.utils.utils import InputPadder, forward_interpolate

# @torch.no_grad()
# def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
#     """ Create submission for the Sintel leaderboard """
#     model.eval()
#     test_dataset = datasets.KITTI(split='testing', aug_params=None)
#
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#
#     for test_id in range(len(test_dataset)):
#         image1, image2, (frame_id,) = test_dataset[test_id]
#         padder = InputPadder(image1.shape, mode='kitti')
#         image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
#
#         _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
#         flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
#
#         output_filename = os.path.join(output_path, frame_id)
#         frame_utils.writeFlowKITTI(output_filename, flow)


# class InputPadder:
#     """ Pads images such that dimensions are divisible by 8 """
#
#     def __init__(self, dims, mode='sintel'):
#         self.ht, self.wd = dims[-2:]
#         pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
#         pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
#         if mode == 'sintel':
#             self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
#         elif mode == 'kitti432':
#             self._pad = [0, 0, 0, 432 - self.ht]
#         elif mode == 'kitti400':
#             self._pad = [0, 0, 0, 400 - self.ht]
#         elif mode == 'kitti376':
#             self._pad = [0, 0, 0, 376 - self.ht]
#         else:
#             self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]
#
#     def pad(self, *inputs):
#         return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]
#
#     def unpad(self, x):
#         ht, wd = x.shape[-2:]
#         c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
#         return x[..., c[0]:c[1], c[2]:c[3]]


def compute_grid_indices(image_shape, patch_size=[0, 0], min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size=[0, 0], sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * weights_hw ** 2)
    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h + patch_size[0], w:w + patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx + 1, h:h + patch_size[0], w:w + patch_size[1]])
    return patch_weights


@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission', sigma=0.05, TRAIN_SIZE=[288, 960]):
    """ Create submission for the KITTI leaderboard """
    IMAGE_SIZE = [0, 0]
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')  # padding the image to height of 432
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        IMAGE_SIZE[0] = image1.shape[-2]
        IMAGE_SIZE[1] = image1.shape[-1]
        hws = compute_grid_indices(IMAGE_SIZE, patch_size=TRAIN_SIZE)
        weights = compute_weight(hws, IMAGE_SIZE, patch_size=TRAIN_SIZE, sigma=sigma)
        flows = 0
        flow_count = 0
        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
            _, flow_pre = model(image1_tile, image2_tile, iters=24, test_mode=True)
            # flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)
        flow_pre = flows / flow_count
        flow = padder.unpad(flow_pre[0]).permute(1, 2, 0).cpu().numpy()
        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_kitti(model, iters=24, sigma=0.05, TRAIN_SIZE=[320, 720]):
    """ Peform validation using the KITTI-2015 (train) split """
    IMAGE_SIZE = [0, 0]
    model.eval()
    val_dataset = datasets.KITTI(split='training')
    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        IMAGE_SIZE[0] = image1.shape[-2]
        IMAGE_SIZE[1] = image1.shape[-1]
        hws = compute_grid_indices(IMAGE_SIZE, patch_size=TRAIN_SIZE)
        weights = compute_weight(hws, IMAGE_SIZE, patch_size=TRAIN_SIZE, sigma=sigma)
        flows = 0
        flow_count = 0
        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
            _, flow_pre = model(image1_tile, image2_tile, iters=iters, test_mode=True)
            # flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)
        flow_pre = flows / flow_count

        flow = padder.unpad(flow_pre[0]).cpu()
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5
        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_sintel(model, iters=32, sigma=0.05, TRAIN_SIZE=[400, 720]):
    """ Peform validation using the Sintel (train) split """
    IMAGE_SIZE = [0, 0]
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            IMAGE_SIZE[0] = image1.shape[-2]
            IMAGE_SIZE[1] = image1.shape[-1]
            hws = compute_grid_indices(IMAGE_SIZE, patch_size=TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, patch_size=TRAIN_SIZE, sigma=sigma)
            flows = 0
            flow_count = 0
            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h + TRAIN_SIZE[0], w:w + TRAIN_SIZE[1]]
                _, flow_pre = model(image1_tile, image2_tile, iters=iters, test_mode=True)
                padding = (w, IMAGE_SIZE[1] - w - TRAIN_SIZE[1], h, IMAGE_SIZE[0] - h - TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)
            flow_pre = flows / flow_count

            flow = padder.unpad(flow_pre[0]).cpu()
            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)
        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)
    return results


# CUDA_VISIBLE_DEVICES=0 python evaluate_tile.py

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
    checkpoints_path = 'checkpoints_things_v1/raft-things.pth'
    if checkpoints_path is not None:
        model.load_state_dict(torch.load(checkpoints_path), strict=True)
    model.cuda()
    model.eval()

    # validate_sintel(model.module)
    validate_kitti(model.module)


