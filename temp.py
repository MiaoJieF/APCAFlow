from core import apcaflow
import argparse
import torch
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true', help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    flow_model = nn.DataParallel(apcaflow.APCAFlow(args))
    print("Parameter Count: %d" % count_parameters(flow_model))
    flow_model.cuda()
    flow_model.eval()
    img_1 = torch.rand((1, 3, 256, 512)).cuda()
    img_2 = torch.rand((1, 3, 256, 512)).cuda()
    with torch.no_grad():
        flow_predictions = flow_model(img_1, img_2)
    print(len(flow_predictions))
    flow = flow_predictions[-1]
    print(flow.shape)
