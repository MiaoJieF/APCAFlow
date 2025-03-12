import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler, coords_grid
from einops.layers.torch import Rearrange
import time
from torch import einsum

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, cost_volume_8, cost_volume_64, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = cost_volume_8.unsqueeze(dim=3)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        # init_cost_volume = init_cost_volume
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 2):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)
        corr_64 = cost_volume_64.unsqueeze(dim=3)
        batch, h1, w1, dim, h2, w2 = corr_64.shape
        corr_64 = corr_64.reshape(batch * h1 * w1, dim, h2, w2)
        self.corr_pyramid.append(corr_64)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            # print(corr.shape, coords_lvl.shape)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


'''
class CorrBlock:
    def __init__(self, cost_volume, init_cost_volume, num_levels=4, radius=4):
        # print(cost_volume.shape, init_cost_volume.shape)
        # torch.Size([1, 32, 64, 32, 64]) torch.Size([1, 32, 64, 32, 64])
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation
        corr = cost_volume.unsqueeze(dim=3)
        init_corr = init_cost_volume.unsqueeze(dim=3)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        init_corr = init_corr.reshape(batch * h1 * w1, dim, h2, w2)
        # init_cost_volume = init_cost_volume
        self.corr_pyramid.append(corr)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            init_corr = F.avg_pool2d(init_corr, 2, stride=2)
            self.corr_pyramid.append(corr)
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            init_corr = self.init_corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            # print(corr.shape, coords_lvl.shape)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            init_corr = bilinear_sampler(init_corr, coords_lvl)
            init_corr = init_corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
            out_pyramid.append(init_corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    # @staticmethod
    # def corr(fmap1, fmap2):
    #     batch, dim, ht, wd = fmap1.shape
    #     fmap1 = fmap1.view(batch, dim, ht * wd)
    #     fmap2 = fmap2.view(batch, dim, ht * wd)
    #
    #     corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    #     corr = corr.view(batch, ht, wd, 1, ht, wd)
    #     return corr / torch.sqrt(torch.tensor(dim).float())
'''


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2 ** i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvLA(nn.Module):
    def __init__(self, dim, use_norm=False):
        super().__init__()
        self.use_norm = use_norm
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.v = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.attn = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.use_norm:
            x = self.norm(x)
        v = self.v(x)
        x = v * self.attn(x)
        x = self.proj(x)
        return x


class Aggregation_Block(nn.Module):
    def __init__(self, in_channels=64, mid_channels=128):
        super(Aggregation_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            ConvLA(mid_channels),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

    def forward(self, cost_volume):
        cost_volume = self.conv1(cost_volume)
        return cost_volume


class Aggregation_Block_global(nn.Module):
    def __init__(self, in_channels=64, a_in_chanels=1, a_mid_channels=4):
        super(Aggregation_Block_global, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1, stride=1, padding=0),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            BasicConv(a_in_chanels, a_mid_channels, is_3d=True, bn=False, relu=True, kernel_size=3, padding=1, stride=1,
                      dilation=1),
            BasicConv(a_mid_channels, a_mid_channels, is_3d=True, bn=False, relu=True, kernel_size=3, padding=1, stride=1,
                      dilation=1),
            BasicConv(a_mid_channels, a_mid_channels // 2, is_3d=True, bn=False, relu=True, kernel_size=3, padding=1,
                      stride=1, dilation=1),
            BasicConv(a_mid_channels // 2, a_in_chanels, is_3d=True, bn=False, relu=True, kernel_size=3, padding=1,
                      stride=1, dilation=1)
        )

    def forward(self, cost_volume, batch, hnum, wnum):
        cost = self.conv1(cost_volume)
        _, _, h, w = cost.shape
        cost_volume_64 = cost.reshape(batch, 1, hnum*wnum, h, w)
        cost_volume_64 = self.conv2(cost_volume_64).squeeze(dim=1).reshape(batch, hnum, wnum, h, w)
        return cost_volume_64


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        self.f_h = self.ht + pad_ht
        self.f_w = self.wd + pad_wd
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='constant', value=0) for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class Gcorr_agg(nn.Module):
    def __init__(self, patch_hight=8, patch_width=8):
        super(Gcorr_agg, self).__init__()
        self.patch_hight = patch_hight
        self.patch_width = patch_width
        self.corr_to_patch = nn.Sequential(
            Rearrange('b h1 w1 (h_num p1) (w_num p2) -> b (h_num w_num) (p1 p2) h1 w1', p1=patch_hight, p2=patch_width)
        )
        self.agg_block = Aggregation_Block(in_channels=self.patch_hight * self.patch_width)
        self.agg_block_global = Aggregation_Block_global(in_channels=self.patch_hight * self.patch_width)

    def forward(self, corr):
        b, h1, w1, h2, w2 = corr.shape
        corr = corr.reshape((b, h1 * w1, h2, w2))
        input_padder = InputPadder(corr.shape)
        corr = input_padder.pad(corr)[0]
        corr = corr.reshape(b, h1, w1, input_padder.f_h, input_padder.f_w)
        h_num = corr.shape[-2] // self.patch_hight
        w_num = corr.shape[-1] // self.patch_width
        corr = self.corr_to_patch(corr)
        corr = corr.reshape(b * h_num * w_num, self.patch_hight * self.patch_width, h1, w1)
        corr = self.agg_block(corr)
        corr_64 = self.agg_block_global(corr, b, h_num, w_num)
        corr_64 = corr_64.permute(0, 3, 4, 1, 2)
        corr = corr.reshape(b, h_num * w_num, self.patch_hight * self.patch_width, h1, w1)
        corr = corr.reshape(b, h_num, w_num, self.patch_hight, self.patch_width, h1, w1)
        corr = corr.permute(0, 5, 6, 1, 3, 2, 4)
        corr = corr.reshape(b, h1, w1, h_num * self.patch_hight, w_num * self.patch_width)
        corr_8 = input_padder.unpad(corr)
        return corr_8, corr_64


if __name__ == '__main__':
    # corr = torch.rand((1, 128, 128, 128, 128)).cuda()
    corr = torch.rand((1, 64, 128, 64, 128)).cuda()
    # corr = torch.rand((1, 32, 64, 32, 64)).cuda()
    # model = Gm_corr_agg().cuda()
    with torch.no_grad():
        model = Gcorr_agg().cuda()
        model(corr)
