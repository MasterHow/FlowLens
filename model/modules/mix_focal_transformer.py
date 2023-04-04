"""
    This code is based on:
    [1] FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting, ICCV 2021
        https://github.com/ruiliu-ai/FuseFormer
    [2] Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet, ICCV 2021
        https://github.com/yitu-opensource/T2T-ViT
    [3] Focal Self-attention for Local-Global Interactions in Vision Transformers, NeurIPS 2021
        https://github.com/microsoft/Focal-Transformer
    [4] Self-slimmed Vision Transformer, ECCV 2022
        https://github.com/Sense-X/SiT
"""

import math
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.feat_prop import flow_warp


class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding,
                 t2t_param):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

        self.t2t_param = t2t_param

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.t2t_param['padding'][0] -
                   (self.t2t_param['kernel_size'][0] - 1) - 1) /
                  self.t2t_param['stride'][0] + 1)      # token在竖直方向的个数
        f_w = int((output_size[1] + 2 * self.t2t_param['padding'][1] -
                   (self.t2t_param['kernel_size'][1] - 1) - 1) /
                  self.t2t_param['stride'][1] + 1)      # token在水平方向的个数

        feat = self.t2t(x)      # 把特征图划分为token(不含有可学习参数)，[B*t, C*token_h*token_w, f_h*f_w]
        feat = feat.permute(0, 2, 1)    # [B*t, Num_token, Length_token]
        # feat shape [b*t, num_vec, ks*ks*c]
        feat = self.embedding(feat)     # [B*t, Num_token, hidden] 含参数
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))     # [B, t, f_h, f_w, hidden]
        return feat


class Mlp(nn.Module):
    '''
    MLP with support to use group linear operator
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SoftComp(nn.Module):
    r"""Revised by Hao:
        Add token fusion for video inpainting support.
        Transfer x to contiguous before view operation"""
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel,
                                   channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.contiguous().view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        feat = self.bias_conv(feat)
        return feat


class MixConv2d(nn.Module):
    """MixConv2d from HRViT."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv_3 = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups // 2,
            bias=bias,
        )
        self.conv_5 = nn.Conv2d(
            in_channels - in_channels // 2,
            out_channels - out_channels // 2,
            kernel_size + 2,
            stride=stride,
            padding=padding + 1,
            dilation=dilation,
            groups=groups - groups // 2,
            bias=bias,
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        x = torch.cat([x1, x2], dim=1)
        return x


class MixFusionFeedForward(nn.Module):
    """Mix F3N for transformer, by Hao."""
    def __init__(self, d_model, n_vecs=None, t2t_params=None):
        super(MixFusionFeedForward, self).__init__()
        # We set d_ff as a default to 1960
        hd = 1960   # hidden dim
        self.conv1 = nn.Sequential(nn.Linear(d_model, hd))

        # MixConv
        self.mix_conv = MixConv2d(
            in_channels=hd,
            out_channels=hd,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hd,
            dilation=1,
            bias=True,
        )

        self.conv2 = nn.Sequential(nn.GELU(), nn.Linear(hd, d_model))
        assert t2t_params is not None and n_vecs is not None
        self.t2t_params = t2t_params

    def forward(self, x, output_size, T, H, W):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params['kernel_size']):
            n_vecs *= int((output_size[i] + 2 * self.t2t_params['padding'][i] -
                           (d - 1) - 1) / self.t2t_params['stride'][i] + 1)

        x = self.conv1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, 49).view(-1, n_vecs, 49).permute(0, 2, 1)
        normalizer = F.fold(normalizer,
                            output_size=output_size,
                            kernel_size=self.t2t_params['kernel_size'],
                            padding=self.t2t_params['padding'],
                            stride=self.t2t_params['stride'])

        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.t2t_params['kernel_size'],
                   padding=self.t2t_params['padding'],
                   stride=self.t2t_params['stride'])

        x = F.unfold(x / normalizer,
                     kernel_size=self.t2t_params['kernel_size'],
                     padding=self.t2t_params['padding'],
                     stride=self.t2t_params['stride']).permute(
                         0, 2, 1).contiguous().view(b, n, c)

        x = x.reshape(b*T, H, W, c).permute(0, 3, 1, 2).contiguous()    # B*T, C, H, W
        x = self.mix_conv(x).permute(0, 2, 3, 1).contiguous().reshape(b, n, c)  # B, T*H*W, C

        x = self.conv2(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, T*window_size*window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1],
               window_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(
        -1, T * window_size[0] * window_size[1], C)
    return windows


def window_partition_noreshape(x, window_size):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, T, window_size, window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1],
               window_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
    return windows


def window_reverse(windows, window_size, T, H, W):
    """
    Args:
        windows: shape is (num_windows*B, T, window_size, window_size, C)
        window_size (tuple[int]): Window size
        T (int): Temporal length of video
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, T, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], T,
                     window_size[0], window_size[1], -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
    return x


class TemporalLePEAttention(nn.Module):
    """by Hao:
            1. Able to compute attention with non-square input.
            2. Extend ddca to Temporal-ddca.
            3. Enhance ddca with global window attention with pooling and focal.
            temporal (bool): It True, extend ddca to Temporal ddca
            """
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None, temporal=False):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution[0], self.resolution[1]
        elif idx == 0:
            H_sp, W_sp = self.resolution[0], self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution[1], self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.idx = idx

        self.attn_drop = nn.Dropout(attn_drop)
        self.temporal = temporal

        # 现在默认是池化到和条带宽度一样的
        # 池化到宽度为条带的宽度
        self.pool_layers = nn.ModuleList()
        window_size_glo = [self.H_sp, self.W_sp]
        self.pool_layers.append(
            nn.Linear(window_size_glo[0] * window_size_glo[1], self.split_size))
        self.pool_layers[-1].weight.data.fill_(
            self.split_size / (window_size_glo[0] * window_size_glo[1]))
        self.pool_layers[-1].bias.data.fill_(0)

        # 展开函数
        self.unfolds = nn.ModuleList()

        # ddca
        # 使用与池化完特征方向相同的滑窗，感受野扩展到非局部
        # 现在默认是池化到和条带宽度一样的
        # 宽度上不需要padding因为会池化到和条带宽度相同
        self.focal_window = [self.W_sp, self.H_sp]      # 刚好和原来的窗口相反
        kernel_size = self.focal_window
        if idx == 0:
            # H_sp等于纵向分辨率时，考虑最后一个窗口需要pad H_sp-1, 注意padding是两边的
            padding = [0, self.H_sp//2]
            stride = [1, 1]
        elif idx == 1:
            # 反之当横向的窗口大小等于横向分辨率时，考虑最后一个窗口需要pad W_sp-1, 注意padding是两边的
            padding = [self.W_sp//2, 0]
            stride = [1, 1]

        # define unfolding operations
        # 保证unfold后的kv尺寸和原来一样 变向等于是展开了kv
        self.unfolds += [
            nn.Unfold(kernel_size=kernel_size,
                      stride=stride,
                      padding=padding)
        ]

    def im2ddca(self, x, H_sp=None, W_sp=None):
        """
        H_sp: height of strip window size.
        W_sp: Width of strip window size.
        """
        B, N, C = x.shape

        # H = W = int(np.sqrt(N))
        H = self.resolution[0]
        W = self.resolution[1]

        if H_sp is None:
            # default manner
            H_sp = self.H_sp
            W_sp = self.W_sp

        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.img2windows(x, H_sp, W_sp)
        x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # x: [-1, head, H_sp*W_sp, C/head]
        return x

    def img2windows(self, img, H_sp, W_sp):
        """
        img: B C H W
        """
        B, C, H, W = img.shape
        img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
        return img_perm

    def get_lepe(self, x, func):
        B, N, C = x.shape

        # H = W = int(np.sqrt(N))
        H = self.resolution[0]
        W = self.resolution[1]

        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def windows2img(self, img_splits_hw, H_sp, W_sp, H, W):
        """
        img_splits_hw: B' H W C
        """
        B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

        img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
        img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return img

    def im2ddca_temporal(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()   # B C T H W
        x = self.img2windows_temporal(x, self.H_sp, self.W_sp)  # B*H/H_sp*W/W_sp T*H_sp*W_sp C
        x = x.reshape(-1, T * self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def img2windows_temporal(self, img, H_sp, W_sp):
        """
        img: B C T H W
        """
        B, C, T, H, W = img.shape
        img_reshape = img.view(B, C, T, H // H_sp, H_sp, W // W_sp, W_sp)
        img_perm = img_reshape.permute(0, 3, 5, 2, 4, 6, 1).contiguous().reshape(-1, T * H_sp * W_sp, C)
        return img_perm

    def get_lepe_temporal(self, x, func):
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().reshape(B * T, C, H, W)   # B*T C H W

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, T, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 3, 5, 1, 2, 4, 6).contiguous().reshape(-1, C, H_sp, W_sp)  # B'*T, C, H', W'

        lepe = func(x)  ### B'*T, C, H', W'
        lepe = lepe.reshape(B * H // H_sp * W // W_sp, T, self.num_heads, C // self.num_heads, H_sp * W_sp)\
            .permute(0, 2, 1, 4, 3).contiguous().reshape(-1, self.num_heads, T * H_sp * W_sp, C // self.num_heads)
        # lepe: B' head T*H_sp*W_sp C/head

        x = x.reshape(B * H // H_sp * W // W_sp, T, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp)\
            .permute(0, 2, 1, 4, 3).contiguous().reshape(-1, self.num_heads, T * H_sp * W_sp, C // self.num_heads)
        # x: B' head T*Hsp*Wsp C/head
        return x, lepe

    def windows2img_temporal(self, img_splits_hw, T, H_sp, W_sp, H, W):
        """
        img_splits_hw: B' THW C
        img: B T H W C
        """
        B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

        img = img_splits_hw.view(B, H // H_sp, W // W_sp, T, H_sp, W_sp, -1)
        img = img.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
        return img

    def forward(self, qkv):
        """
        q,k,v: B, T, H, W, C
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, T, H, W, C = q.shape

        # 池化kv用于获得global att
        nWh = H // self.H_sp    # 窗口数量
        nWw = W // self.W_sp

        # 改变kv形状->B, nWh, nWw, T, C, window_size_h*window_size_w
        k = k.reshape(B, T, nWh, self.H_sp, nWw, self.W_sp, C).permute(0, 2, 4, 1, 6, 3, 5).contiguous() \
            .reshape(B, nWh, nWw, T, C, self.H_sp * self.W_sp)
        v = v.reshape(B, T, nWh, self.H_sp, nWw, self.W_sp, C).permute(0, 2, 4, 1, 6, 3, 5).contiguous() \
            .reshape(B, nWh, nWw, T, C, self.H_sp * self.W_sp)

        # 池化kv
        k_pooled = self.pool_layers[0](k).flatten(-2)  # B, nWh, nWw, T, C
        v_pooled = self.pool_layers[0](v).flatten(-2)  # B, nWh, nWw, T, C

        # 转化池化后的kv到需要的shape，默认池化到条带宽度，上面的focal v1逻辑已经被抛弃
        # 条带宽度不为1时池化的形状也不同
        if self.idx == 0:
            # 纵向条纹，池化完是横向的
            k_pooled = k_pooled.permute(0, 3, 4, 1, 2).contiguous().reshape(B * T, C, nWh * self.split_size, nWw)  # B*T, C, nWh, nWw
            v_pooled = v_pooled.permute(0, 3, 4, 1, 2).contiguous().reshape(B * T, C, nWh * self.split_size, nWw)  # B*T, C, nWh, nWw
        else:
            # 横向条纹，池化完是纵向的
            k_pooled = k_pooled.permute(0, 3, 4, 1, 2).contiguous().reshape(B * T, C, nWh, nWw * self.split_size)  # B*T, C, nWh, nWw
            v_pooled = v_pooled.permute(0, 3, 4, 1, 2).contiguous().reshape(B * T, C, nWh, nWw * self.split_size)  # B*T, C, nWh, nWw

        # 恢复kv形状->B, T, H, W, C
        k = k.reshape(B, nWh, nWw, T, C, self.H_sp, self.W_sp).permute(0, 3, 1, 5, 2, 6, 4)\
            .contiguous().reshape(B, T, H, W, C)
        v = v.reshape(B, nWh, nWw, T, C, self.H_sp, self.W_sp).permute(0, 3, 1, 5, 2, 6, 4)\
            .contiguous().reshape(B, T, H, W, C)

        ### Img2Window
        if self.temporal:
            # 3D temporal ddca att
            q = self.im2ddca_temporal(q)
            k = self.im2ddca_temporal(k)
            v, lepe = self.get_lepe_temporal(v, self.get_v)
        else:
            # 2D ddca att
            # reshape qkv to [B*T H*W C]
            q = q.reshape(B * T, H * W, C)
            k = k.reshape(B * T, H * W, C)
            v = v.reshape(B * T, H * W, C)

            # 利用不同宽度的kv池化到当前宽度来增强kv，先获得不同宽度的滑窗捏

            q = self.im2ddca(q)
            k = self.im2ddca(k)
            # 其实这里相当于已经有一个CONV path了
            v, lepe = self.get_lepe(v, self.get_v)

        # 利用池化kv增强kv
        if self.temporal:
            # 时间也展开(其实一样因为时间窗口是1)
            (k_pooled, v_pooled) = map(
                lambda t: self.unfolds[0]
                (t).view(B, T, C, self.unfolds[0].kernel_size[0], self.
                         unfolds[0].kernel_size[1], -1)
                    .permute(0, 5, 1, 3, 4, 2).contiguous().view(
                    -1, T * self.unfolds[0].kernel_size[0] * self.unfolds[
                        0].kernel_size[1], self.num_heads, C // self.
                           num_heads).permute(0, 2, 1, 3).contiguous(),
                # (B x (nWh*nWw)) x nHeads x (T x unfold_wsize x unfold_wsize) x C/head
                (k_pooled, v_pooled))

            # 因为两侧对称的padding会导致unfold多一个滑窗
            if self.idx == 0:
                # 丢掉竖直方向上最后一个
                k_pooled = k_pooled.view(
                    B, nWh, -1, self.num_heads, T * self.unfolds[0].kernel_size[0]
                    * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :-1, :, :, :, :] \
                    .reshape(-1, self.num_heads, T * self.unfolds[0].kernel_size[0] * self.unfolds[0]
                             .kernel_size[1], C // self.num_heads)      # drop last
                v_pooled = v_pooled.view(
                    B, nWh, -1, self.num_heads, T * self.unfolds[0].kernel_size[0]
                    * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :-1, :, :, :, :] \
                    .reshape(-1, self.num_heads, T * self.unfolds[0].kernel_size[0] * self.unfolds[0]
                             .kernel_size[1], C // self.num_heads)      # drop last
            elif self.idx == 1:
                # 丢掉水平方向上最后一个
                k_pooled = k_pooled.view(
                    B, -1, nWw, self.num_heads, T * self.unfolds[0].kernel_size[0]
                    * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :-1, :, :, :, :, :] \
                    .reshape(-1, self.num_heads, T * self.unfolds[0].kernel_size[0] * self.unfolds[0]
                             .kernel_size[1], C // self.num_heads)     # drop last
                v_pooled = v_pooled.view(
                    B, -1, nWw, self.num_heads, T * self.unfolds[0].kernel_size[0]
                    * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :-1, :, :, :, :, :] \
                    .reshape(-1, self.num_heads, T *  self.unfolds[0].kernel_size[0] * self.unfolds[0]
                             .kernel_size[1], C // self.num_heads)  # drop last

        else:
            # 空间展开
            (k_pooled, v_pooled) = map(
                lambda t: self.unfolds[0]
                (t).view(B * T, C, self.unfolds[0].kernel_size[0], self.
                         unfolds[0].kernel_size[1], -1)
                    .permute(0, 4, 2, 3, 1).contiguous().view(
                    -1, self.unfolds[0].kernel_size[0] * self.unfolds[
                        0].kernel_size[1], self.num_heads, C // self.
                        num_heads).permute(0, 2, 1, 3).contiguous(),
                # (B x T x (nWh*nWw)) x nHeads x (unfold_wsize x unfold_wsize) x C/head
                (k_pooled, v_pooled))

            # 因为两侧对称的padding会导致unfold多一个滑窗
            if self.idx == 0:
                # 丢掉竖直方向上最后一个
                k_pooled = k_pooled.view(
                    B, T, nWh, -1, self.num_heads, self.unfolds[0].kernel_size[0]
                    * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :, :-1, :, :, :] \
                    .reshape(-1, self.num_heads, self.unfolds[0].kernel_size[0] * self.unfolds[0]
                             .kernel_size[1], C // self.num_heads)      # drop last
                v_pooled = v_pooled.view(
                    B, T, nWh, -1, self.num_heads, self.unfolds[0].kernel_size[0]
                    * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :, :-1, :, :, :] \
                    .reshape(-1, self.num_heads, self.unfolds[0].kernel_size[0] * self.unfolds[0]
                             .kernel_size[1], C // self.num_heads)      # drop last
            elif self.idx == 1:
                # 丢掉水平方向上最后一个
                k_pooled = k_pooled.view(
                    B, T, -1, nWw, self.num_heads, self.unfolds[0].kernel_size[0]
                    * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :-1, :, :, :, :] \
                    .reshape(-1, self.num_heads, self.unfolds[0].kernel_size[0] * self.unfolds[0]
                             .kernel_size[1], C // self.num_heads)      # drop last
                v_pooled = v_pooled.view(
                    B, T, -1, nWw, self.num_heads, self.unfolds[0].kernel_size[0]
                    * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :-1, :, :, :, :] \
                    .reshape(-1, self.num_heads, self.unfolds[0].kernel_size[0] * self.unfolds[0]
                             .kernel_size[1], C // self.num_heads)      # drop last

        # 增强kv
        k = torch.cat((k, k_pooled), dim=2)
        v = torch.cat((v, v_pooled), dim=2)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe

        ### Window2Img
        if self.temporal:
            # 3D temporal ddca att
            x = x.transpose(1, 2).reshape(-1, T * self.H_sp * self.W_sp, C)
            x = self.windows2img_temporal(x, T, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B T*H*W C
        else:
            # 2D ddca att
            # B*T*H/H_sp*W/W_sp head H_sp*W_sp C/head -> B*T*H/H_sp*W/W_sp H_sp*W_sp C
            x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
            x = self.windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B T*H*W C

        return x


class WindowAttention(nn.Module):
    """Temporal focal window attention
    """
    def __init__(self, dim, expand_size, window_size, focal_window,
                 focal_level, num_heads, qkv_bias, pool_method):

        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        if any(i > 0 for i in self.expand_size) and focal_level > 0:
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br),
                                      0).flatten(0)
            self.register_buffer("valid_ind_rolled",
                                 mask_rolled.nonzero(as_tuple=False).view(-1))

        if pool_method != "none" and focal_level > 1:
            self.unfolds = nn.ModuleList()

            # build relative position bias between local patch and pooled windows
            for k in range(focal_level - 1):
                stride = 2**k
                # 对于奇数和偶数的window size，展开用的kernel尺寸应该不一样。
                if (self.focal_window[0] % 2 != 0) and (self.focal_window[1] % 2 != 0):
                    kernel_size = tuple(2 * (i // 2) + 2**k + (2**k - 1)
                                        for i in self.focal_window)
                elif (self.focal_window[0] % 2 == 0) and (self.focal_window[1] % 2 == 0):
                    # kernel_size = tuple(2 * (i // 2)
                    #                     for i in self.focal_window)
                    raise Exception('Not support with focal_window % 2 == 0 for now')

                # define unfolding operations
                self.unfolds += [
                    nn.Unfold(kernel_size=kernel_size,
                              stride=stride,
                              padding=tuple(i // 2 for i in kernel_size))
                ]

                # define unfolding index for focal_level > 0
                if k > 0:
                    mask = torch.zeros(kernel_size)
                    mask[(2**k) - 1:, (2**k) - 1:] = 1
                    self.register_buffer(
                        "valid_ind_unfold_{}".format(k),
                        mask.flatten(0).nonzero(as_tuple=False).view(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_all, mask_all=None):
        """
        Args:
            x: input features with shape of (B, T, Wh, Ww, C)
            mask: (0/-inf) mask with shape of (num_windows, T*Wh*Ww, T*Wh*Ww) or None

            output: (nW*B, Wh*Ww, C)
        """
        x = x_all[0]

        B, T, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, T, nH, nW, 3,
                                  C).permute(4, 0, 1, 2, 3, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, T, nH, nW, C

        # partition q map
        (q_windows, k_windows, v_windows) = map(
            lambda t: window_partition(t, self.window_size).view(
                -1, T, self.window_size[0] * self.window_size[1], self.
                num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4).
            contiguous().view(-1, self.num_heads, T * self.window_size[
                0] * self.window_size[1], C // self.num_heads), (q, k, v))
        # q(k/v)_windows shape : [16, 4, 225, 128]

        if any(i > 0 for i in self.expand_size) and self.focal_level > 0:
            (k_tl, v_tl) = map(
                lambda t: torch.roll(t,
                                     shifts=(-self.expand_size[0], -self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_tr, v_tr) = map(
                lambda t: torch.roll(t,
                                     shifts=(-self.expand_size[0], self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_bl, v_bl) = map(
                lambda t: torch.roll(t,
                                     shifts=(self.expand_size[0], -self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_br, v_br) = map(
                lambda t: torch.roll(t,
                                     shifts=(self.expand_size[0], self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))

            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.
                    num_heads, C // self.num_heads), (k_tl, k_tr, k_bl, k_br))
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.
                    num_heads, C // self.num_heads), (v_tl, v_tr, v_bl, v_br))
            k_rolled = torch.cat(
                (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows),
                2).permute(0, 3, 1, 2, 4).contiguous()
            v_rolled = torch.cat(
                (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows),
                2).permute(0, 3, 1, 2, 4).contiguous()

            # mask out tokens in current window
            k_rolled = k_rolled[:, :, :, self.valid_ind_rolled]
            v_rolled = v_rolled[:, :, :, self.valid_ind_rolled]
            temp_N = k_rolled.shape[3]
            k_rolled = k_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            v_rolled = v_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            k_rolled = torch.cat((k_windows, k_rolled), 2)
            v_rolled = torch.cat((v_windows, v_rolled), 2)
        else:
            k_rolled = k_windows
            v_rolled = v_windows

        # q(k/v)_windows shape : [16, 4, 225, 128]
        # k_rolled.shape : [16, 4, 5, 165, 128]
        # ideal expanded window size 153 ((5+2*2)*(9+2*4))
        # k_windows=45 expand_window=108 overlap_window=12 (since expand_size < window_size / 2)

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            for k in range(self.focal_level - 1):
                stride = 2**k
                # B, T, nWh, nWw, C
                x_window_pooled = x_all[k + 1].permute(0, 3, 1, 2,
                                                       4).contiguous()

                nWh, nWw = x_window_pooled.shape[2:4]

                # generate mask for pooled windows
                mask = x_window_pooled.new(T, nWh, nWw).fill_(1)
                # unfold mask: [nWh*nWw//s//s, k*k, 1]
                unfolded_mask = self.unfolds[k](mask.unsqueeze(1)).view(
                    1, T, self.unfolds[k].kernel_size[0], self.unfolds[k].kernel_size[1], -1).permute(4, 1, 2, 3, 0).contiguous().\
                    view(nWh*nWw // stride // stride, -1, 1)

                if k > 0:
                    valid_ind_unfold_k = getattr(
                        self, "valid_ind_unfold_{}".format(k))
                    unfolded_mask = unfolded_mask[:, valid_ind_unfold_k]

                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                x_window_masks = x_window_masks.masked_fill(
                    x_window_masks == 0,
                    float(-100.0)).masked_fill(x_window_masks > 0, float(0.0))
                mask_all[k + 1] = x_window_masks

                # generate k and v for pooled windows
                qkv_pooled = self.qkv(x_window_pooled).reshape(
                    B, T, nWh, nWw, 3, C).permute(4, 0, 1, 5, 2,
                                                  3).view(3, -1, C, nWh,
                                                          nWw).contiguous()
                # B*T, C, nWh, nWw
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]
                # k_pooled_k shape: [5, 512, 4, 4], i.e. [B*T, C, nWh, nWw] 空间池化后的window, 最后两个通道是window数量
                # self.unfolds[k](k_pooled_k) shape: [5, 23040 (512 * 5 * 9 ), 16]

                (k_pooled_k, v_pooled_k) = map(
                    lambda t: self.unfolds[k]
                    (t).view(B, T, C, self.unfolds[k].kernel_size[0], self.
                             unfolds[k].kernel_size[1], -1)
                    .permute(0, 5, 1, 3, 4, 2).contiguous().view(
                        -1, T, self.unfolds[k].kernel_size[0] * self.unfolds[
                            k].kernel_size[1], self.num_heads, C // self.
                        num_heads).permute(0, 3, 1, 2, 4).contiguous(),
                    # (B x (nH*nW)) x nHeads x T x (unfold_wsize x unfold_wsize) x head_dim
                    (k_pooled_k, v_pooled_k))
                # k_pooled_k shape : [16, 4, 5, 45, 128],
                # i.e. [B * nWh * nWw, head, T, sh * sw, C // head], sh和sw是window的尺寸(5*9)

                # select valid unfolding index
                if k > 0:
                    (k_pooled_k, v_pooled_k) = map(
                        lambda t: t[:, :, :, valid_ind_unfold_k],
                        (k_pooled_k, v_pooled_k))

                k_pooled_k = k_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)
                v_pooled_k = v_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k]

            # k_all (v_all) shape : [16, 4, 5 * 210, 128], i.e. [B * nWh * nWw, head, k_rolled + k_pooled, C // head]
            # k_pooled : [B * nWh * nWw, head, T * sh * sw, C // head]
            k_all = torch.cat([k_rolled] + k_pooled, 2)
            v_all = torch.cat([v_rolled] + v_pooled, 2)
        else:
            k_all = k_rolled
            v_all = v_rolled

        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        # B*nW, nHead, T*window_size*window_size, T*focal_window_size*focal_window_size
        attn = (q_windows @ k_all.transpose(-2, -1))
        # T * 45
        window_area = T * self.window_size[0] * self.window_size[1]
        # T * 165
        window_area_rolled = k_rolled.shape[2]

        if self.pool_method != "none" and self.focal_level > 1:
            offset = window_area_rolled
            for k in range(self.focal_level - 1):
                # add attentional mask
                # mask_all[1] shape [1, 16, T * 45]

                bias = tuple((i + 2**k - 1) for i in self.focal_window)

                if mask_all[k + 1] is not None:
                    attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] = \
                        attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] + \
                        mask_all[k+1][:, :, None, None, :].repeat(
                            attn.shape[0] // mask_all[k+1].shape[1], 1, 1, 1, 1).view(-1, 1, 1, mask_all[k+1].shape[-1])

                offset += T * bias[0] * bias[1]

        if mask_all[0] is not None:
            nW = mask_all[0].shape[0]
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads,
                             window_area, N)
            attn[:, :, :, :, :
                 window_area] = attn[:, :, :, :, :window_area] + mask_all[0][
                     None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, window_area, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area,
                                                   C)
        x = self.proj(x)
        return x


class WindowAttentionClipRecurrent(nn.Module):
    """Temporal focal window attention with clip recurrent hub built in."""
    def __init__(self, dim, expand_size, window_size, focal_window,
                 focal_level, num_heads, qkv_bias, pool_method,
                 memory,
                 cs_win_strip):

        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        self.memory = memory

        if any(i > 0 for i in self.expand_size) and focal_level > 0:
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br),
                                      0).flatten(0)
            self.register_buffer("valid_ind_rolled",
                                 mask_rolled.nonzero(as_tuple=False).view(-1))

        if pool_method != "none" and focal_level > 1:
            self.unfolds = nn.ModuleList()

            # build relative position bias between local patch and pooled windows
            for k in range(focal_level - 1):
                stride = 2**k
                kernel_size = tuple(2 * (i // 2) + 2**k + (2**k - 1)
                                    for i in self.focal_window)
                # define unfolding operations
                self.unfolds += [
                    nn.Unfold(kernel_size=kernel_size,
                              stride=stride,
                              padding=tuple(i // 2 for i in kernel_size))
                ]

                # define unfolding index for focal_level > 0
                if k > 0:
                    mask = torch.zeros(kernel_size)
                    mask[(2**k) - 1:, (2**k) - 1:] = 1
                    self.register_buffer(
                        "valid_ind_unfold_{}".format(k),
                        mask.flatten(0).nonzero(as_tuple=False).view(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

        if self.memory:
            self.m_k = []  # 缓存的memory keys
            self.m_v = []  # 缓存的memory values
            self.max_len = 1  # 缓存memory的最大记忆长度
            self.compression_factor = 1  # 缓存memory的通道压缩因子

            # memory机制的含参数运算层-[基于通道的压缩]
            # 兼容局部非局部都存储和只存储局部帧的行为
            self.f_k = nn.Linear(dim, dim // self.compression_factor, bias=True)  # 用于更新上一时刻的k并压缩之前的记忆张量
            self.f_v = nn.Linear(dim, dim // self.compression_factor, bias=True)  # 用于更新上一时刻的v并压缩之前的记忆张量

            # 使用cross attention对齐记忆缓存和当前帧
            # 当记忆时间大于1时，需要先将缓存里的记忆压缩到和当前迭代同样尺度，才能做attention.
            # 使用线性层聚合不同时间的记忆，然后和当前做cross att
            if self.max_len > 1:
                self.lin_k = nn.Linear(
                    dim // self.compression_factor * self.max_len,
                    dim, bias=qkv_bias)  # 用于把记忆里的k和当前的k进行融合
                self.lin_v = nn.Linear(
                    dim // self.compression_factor * self.max_len,
                    dim, bias=qkv_bias)  # 用于把记忆里的v和当前的v进行融合

            # 将记忆查询输出和当前帧的输出融合
            self.fusion_proj = nn.Linear(2 * dim, dim)

            # 使用DDCA
            window_stride = 4  # 每个window在两个方向上占用了多少个token
            split_size = cs_win_strip  # 条形窗口的宽度
            num_heads_cs = num_heads//2

            patches_resolution = [self.window_size[0] * window_stride,
                                  self.window_size[1] * window_stride]     # token的纵向和横向的个数

            # DDCA (3d decoupled cross attention): 解耦时间和空间注意力
            self.cs_att = nn.ModuleList([
                TemporalLePEAttention(dim//2, resolution=patches_resolution, idx=i,
                                      split_size=split_size, num_heads=num_heads_cs, dim_out=dim//2,
                                      qk_scale=None, attn_drop=0., proj_drop=0.,
                                      temporal=False)
                for i in range(0, 2)])      # 两个，一个横向一个纵向
            # 时间attention的线性层
            self.cm_proj_t = nn.Linear(dim, dim)

            # ddca 的线性层
            self.cs_proj = nn.Linear(dim, dim)

    def forward(self, x_all, mask_all=None, l_t=5):
        """
        Args:
            x_all: input features with shape of (B, T, Wh, Ww, C)
            mask_all: (0/-inf) mask with shape of (num_windows, T*Wh*Ww, T*Wh*Ww) or None
            l_t: local frame nums

            output: (nW*B, Wh*Ww, C)
        """
        x = x_all[0]    # x_all[1]是用来生成池化的kv来做self attention focal的

        B, T, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, T, nH, nW, 3,
                                  C).permute(4, 0, 1, 2, 3, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, T, nH, nW, C

        # memory ability
        if self.memory:
            # 局部和随机的非局部帧都会被存储
            # 压缩上一个记忆缓存
            if len(self.m_k) != 0:
                cm_k = self.f_k(self.m_k[-1])
                cm_v = self.f_v(self.m_v[-1])
            else:
                # 第一帧时没有记忆张量，使用当前帧的k，v
                cm_k = self.f_k(k)
                cm_v = self.f_v(v)

            # 增强qkv
            # 使用cross attention聚合记忆
            # 聚合记忆缓存，用于后续和当前特征进行cross attention
            if self.max_len == 1:
                # 只记忆1次迭代时，不需要聚合
                att_num = 1
                mem_k = cm_k
                mem_v = cm_v
            else:
                # 记忆缓存时间大于1，需要聚合记忆再做attention
                # 使用线性层聚合记忆
                # 只需要最后聚合的记忆和当前特征做一次cross att
                att_num = 1
                if len(self.m_k) == self.max_len:
                    # 记忆缓存满了，直接用线性层聚合
                    # 因为缓存里的最后一帧还没被压缩的替换，所以不取最后一帧
                    mem_k = self.lin_k(torch.cat((
                        torch.cat(self.m_k[:-1], dim=4), cm_k), dim=4))
                    mem_v = self.lin_v(torch.cat((
                        torch.cat(self.m_v[:-1], dim=4), cm_v), dim=4))
                else:
                    # 记忆缓存没满，复制一下
                    repeat_k = self.max_len - len(self.m_k)
                    repeat_v = self.max_len - len(self.m_v)
                    if len(self.m_k) == 0:
                        # 缓存里面啥也没有，当前帧多复制几次
                        mem_k = self.lin_k(cm_k.repeat(1, 1, 1, 1, repeat_k))
                        mem_v = self.lin_v(cm_v.repeat(1, 1, 1, 1, repeat_k))
                    else:
                        # 尽量使用缓存中的帧
                        mem_k = self.lin_k(torch.cat((
                            torch.cat(self.m_k[:-1], dim=4), cm_k.repeat(1, 1, 1, 1, repeat_k + 1)), dim=4))
                        mem_v = self.lin_v(torch.cat((
                            torch.cat(self.m_v[:-1], dim=4), cm_v.repeat(1, 1, 1, 1, repeat_v + 1)), dim=4))

            for att_idx in range(0, att_num):
                # 记忆时间超过1并且不用线性层聚合才需要这些判断逻辑
                # 也就是说，只有需要做多次cross attention才需要这些逻辑
                # 各种不同的cross attention选择
                # 信息将额外在T维度流动
                # 基于3d decoupled attention聚合时空记忆和当前迭代
                # 解耦时间和空间聚合，时间聚合使用vanilla attention
                q = q.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                     self.num_heads) \
                    .permute(0, 3, 1, 2).contiguous()  # B*N, head, T, C//head
                mem_k = mem_k.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                             self.num_heads) \
                    .permute(0, 3, 1, 2).contiguous()
                mem_v = mem_v.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                             self.num_heads) \
                    .permute(0, 3, 1, 2).contiguous()
                cm_attn_t = (q @ mem_k.transpose(-2, -1)) * self.scale
                cm_attn_t = cm_attn_t.softmax(dim=-1)
                cm_x_t = (cm_attn_t @ mem_v).permute(0, 2, 3, 1).reshape(B, nH * nW, T, C) \
                    .permute(0, 2, 1, 3).reshape(B * T, nH * nW, C)
                cm_x_t = self.cm_proj_t(cm_x_t)
                # 恢复qkv的shape
                q = q.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2,
                                                                           4).contiguous()
                mem_k = mem_k.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2,
                                                                                   4).contiguous()
                mem_v = mem_v.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2,
                                                                                   4).contiguous()

                cm_x1 = self.cs_att[0](qkv=[q[:, :, :, :, :C // 2],
                                            mem_k[:, :, :, :, :C // 2],
                                            mem_v[:, :, :, :, :C // 2]])
                cm_x2 = self.cs_att[1](qkv=[q[:, :, :, :, C // 2:],
                                            mem_k[:, :, :, :, C // 2:],
                                            mem_v[:, :, :, :, C // 2:]])
                cm_x = torch.cat([cm_x1, cm_x2], dim=2)
                cm_x = self.cs_proj(cm_x)

                cm_x += cm_x_t.reshape(B, T * nH * nW, C)

                # cm_x_final用来存储不同时间记忆attention的结果
                if att_idx == 0:
                    # 第一次直接初始化cm_x_final
                    cm_x_final = cm_x
                else:
                    cm_x_final += cm_x

            k_temp = k  # debug


        # partition q map
        (q_windows, k_windows, v_windows) = map(
            lambda t: window_partition(t, self.window_size).view(
                -1, T, self.window_size[0] * self.window_size[1], self.
                num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4).
            contiguous().view(-1, self.num_heads, T * self.window_size[
                0] * self.window_size[1], C // self.num_heads), (q, k, v))
        # q(k/v)_windows shape : [16, 4, 225, 128]

        if any(i > 0 for i in self.expand_size) and self.focal_level > 0:
            (k_tl, v_tl) = map(
                lambda t: torch.roll(t,
                                     shifts=(-self.expand_size[0], -self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_tr, v_tr) = map(
                lambda t: torch.roll(t,
                                     shifts=(-self.expand_size[0], self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_bl, v_bl) = map(
                lambda t: torch.roll(t,
                                     shifts=(self.expand_size[0], -self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_br, v_br) = map(
                lambda t: torch.roll(t,
                                     shifts=(self.expand_size[0], self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))

            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.
                    num_heads, C // self.num_heads), (k_tl, k_tr, k_bl, k_br))
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.
                    num_heads, C // self.num_heads), (v_tl, v_tr, v_bl, v_br))
            k_rolled = torch.cat(
                (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows),
                2).permute(0, 3, 1, 2, 4).contiguous()
            v_rolled = torch.cat(
                (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows),
                2).permute(0, 3, 1, 2, 4).contiguous()

            # mask out tokens in current window
            k_rolled = k_rolled[:, :, :, self.valid_ind_rolled]
            v_rolled = v_rolled[:, :, :, self.valid_ind_rolled]
            temp_N = k_rolled.shape[3]
            k_rolled = k_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            v_rolled = v_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            k_rolled = torch.cat((k_windows, k_rolled), 2)
            v_rolled = torch.cat((v_windows, v_rolled), 2)
        else:
            k_rolled = k_windows
            v_rolled = v_windows

        # q(k/v)_windows shape : [16, 4, 225, 128]
        # k_rolled.shape : [16, 4, 5, 165, 128]
        # ideal expanded window size 153 ((5+2*2)*(9+2*4))
        # k_windows=45 expand_window=108 overlap_window=12 (since expand_size < window_size / 2)

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            for k in range(self.focal_level - 1):
                stride = 2**k
                # B, T, nWh, nWw, C
                x_window_pooled = x_all[k + 1].permute(0, 3, 1, 2,
                                                       4).contiguous()

                nWh, nWw = x_window_pooled.shape[2:4]

                # generate mask for pooled windows
                mask = x_window_pooled.new(T, nWh, nWw).fill_(1)
                # unfold mask: [nWh*nWw//s//s, k*k, 1]
                unfolded_mask = self.unfolds[k](mask.unsqueeze(1)).view(
                    1, T, self.unfolds[k].kernel_size[0], self.unfolds[k].kernel_size[1], -1).permute(4, 1, 2, 3, 0).contiguous().\
                    view(nWh*nWw // stride // stride, -1, 1)

                if k > 0:
                    valid_ind_unfold_k = getattr(
                        self, "valid_ind_unfold_{}".format(k))
                    unfolded_mask = unfolded_mask[:, valid_ind_unfold_k]

                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                x_window_masks = x_window_masks.masked_fill(
                    x_window_masks == 0,
                    float(-100.0)).masked_fill(x_window_masks > 0, float(0.0))
                mask_all[k + 1] = x_window_masks

                # generate k and v for pooled windows
                qkv_pooled = self.qkv(x_window_pooled).reshape(
                    B, T, nWh, nWw, 3, C).permute(4, 0, 1, 5, 2,
                                                  3).view(3, -1, C, nWh,
                                                          nWw).contiguous()
                # B*T, C, nWh, nWw
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]
                # k_pooled_k shape: [5, 512, 4, 4]
                # self.unfolds[k](k_pooled_k) shape: [5, 23040 (512 * 5 * 9 ), 16]

                (k_pooled_k, v_pooled_k) = map(
                    lambda t: self.unfolds[k]
                    (t).view(B, T, C, self.unfolds[k].kernel_size[0], self.
                             unfolds[k].kernel_size[1], -1)
                    .permute(0, 5, 1, 3, 4, 2).contiguous().view(
                        -1, T, self.unfolds[k].kernel_size[0] * self.unfolds[
                            k].kernel_size[1], self.num_heads, C // self.
                        num_heads).permute(0, 3, 1, 2, 4).contiguous(),
                    # (B x (nH*nW)) x nHeads x T x (unfold_wsize x unfold_wsize) x head_dim
                    (k_pooled_k, v_pooled_k))
                # k_pooled_k shape : [16, 4, 5, 45, 128]

                # select valid unfolding index
                if k > 0:
                    (k_pooled_k, v_pooled_k) = map(
                        lambda t: t[:, :, :, valid_ind_unfold_k],
                        (k_pooled_k, v_pooled_k))

                k_pooled_k = k_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)
                v_pooled_k = v_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k]

            # k_all (v_all) shape : [16, 4, 5 * 210, 128]
            k_all = torch.cat([k_rolled] + k_pooled, 2)
            v_all = torch.cat([v_rolled] + v_pooled, 2)
        else:
            k_all = k_rolled
            v_all = v_rolled

        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        # B*nW, nHead, T*window_size*window_size, T*focal_window_size*focal_window_size
        attn = (q_windows @ k_all.transpose(-2, -1))
        # T * 45
        window_area = T * self.window_size[0] * self.window_size[1]
        # T * 165
        window_area_rolled = k_rolled.shape[2]

        if self.pool_method != "none" and self.focal_level > 1:
            offset = window_area_rolled
            for k in range(self.focal_level - 1):
                # add attentional mask
                # mask_all[1] shape [1, 16, T * 45]

                bias = tuple((i + 2**k - 1) for i in self.focal_window)

                if mask_all[k + 1] is not None:
                    attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] = \
                        attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] + \
                        mask_all[k+1][:, :, None, None, :].repeat(
                            attn.shape[0] // mask_all[k+1].shape[1], 1, 1, 1, 1).view(-1, 1, 1, mask_all[k+1].shape[-1])

                offset += T * bias[0] * bias[1]

        if mask_all[0] is not None:
            nW = mask_all[0].shape[0]
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads,
                             window_area, N)
            attn[:, :, :, :, :
                 window_area] = attn[:, :, :, :, :window_area] + mask_all[0][
                     None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, window_area, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area,
                                                   C)
        x = self.proj(x)

        # memory ability
        if self.memory:
            # 将从记忆中查询到的特征与当前特征融合
            res_x = self.fusion_proj(torch.cat((x, cm_x_final.reshape(attn.shape[0], window_area,
                                               C)), dim=2))     # 这里cm_x_final的形状调整到和默认的x一致
            x = x + res_x

            # 缓存更新过的记忆张量
            # 存储没对齐或者token级别对齐的记忆
            try:
                self.m_k[-1] = cm_k.detach()
                self.m_v[-1] = cm_v.detach()
            except:
                # 第一帧的时候记忆张量list为空，需要保证list除了最后一个元素，其他元素都是压缩过的
                self.m_k.append(cm_k.detach())
                self.m_v.append(cm_v.detach())

            # 缓存当前时刻还没被压缩过的记忆张量，会在下一个时刻被压缩
            # 局部帧和非局部帧都会被缓存
            # 直接缓存当前的kv
            self.m_k.append(k_temp.detach())  # debug
            self.m_v.append(v.detach())

            # 保持记忆力的最大长度
            if len(self.m_k) > self.max_len:
                self.m_k.pop(0)
                self.m_v.pop(0)

        return x


class MixFocalTransformerBlock(nn.Module):
    r""" Mix Focal Transformer Block.
    Args:
        dim (int): Number of input channels. Equal to hidden dim.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int):  The number level of focal window.
        focal_window (int):  Window size of each focal window.
        n_vecs (int): Required for F3N.
        t2t_params (int): T2T parameters for F3N.
    Revised by Hao:
        Add token fusion support and memory ability.
        memory (bool): Required for memory ability. Using WindowAttentionClipRecurrent replace the original WindowAttention.
        cs_win_strip (int): ddca attention strip width. Default: 1.
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(5, 9),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 pool_method="fc",
                 focal_level=2,
                 focal_window=(5, 9),
                 norm_layer=nn.LayerNorm,
                 n_vecs=None,
                 t2t_params=None,
                 memory=False,
                 cs_win_strip=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.expand_size = tuple(i // 2 for i in window_size)  # 窗口大小除以2是拓展大小
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window

        self.memory = memory

        self.window_size_glo = self.window_size

        self.pool_layers = nn.ModuleList()
        if self.pool_method != "none":
            for k in range(self.focal_level - 1):
                window_size_glo = tuple(
                    math.floor(i / (2**k)) for i in self.window_size_glo)
                self.pool_layers.append(
                    nn.Linear(window_size_glo[0] * window_size_glo[1], 1))
                self.pool_layers[-1].weight.data.fill_(
                    1. / (window_size_glo[0] * window_size_glo[1]))
                self.pool_layers[-1].bias.data.fill_(0)

        self.norm1 = norm_layer(dim)

        if not self.memory:
            # 使用默认的window attention
            self.attn = WindowAttention(dim,
                                        expand_size=self.expand_size,
                                        window_size=self.window_size,
                                        focal_window=focal_window,
                                        focal_level=focal_level,
                                        num_heads=num_heads,
                                        qkv_bias=qkv_bias,
                                        pool_method=pool_method)
        else:
            # 使用ClipRecurrentHub增强的window attention
            self.attn = WindowAttentionClipRecurrent(dim,
                                                     expand_size=self.expand_size,
                                                     window_size=self.window_size,
                                                     focal_window=focal_window,
                                                     focal_level=focal_level,
                                                     num_heads=num_heads,
                                                     qkv_bias=qkv_bias,
                                                     pool_method=pool_method,
                                                     memory=self.memory,
                                                     cs_win_strip=cs_win_strip)

        self.norm2 = norm_layer(dim)
        # mixf3n, hidden dim的维度与F3N默认的维度保持一致(1960)
        self.mlp = MixFusionFeedForward(dim, n_vecs=n_vecs, t2t_params=t2t_params)

    def forward(self, x):

        l_t = x[2]

        output_size = x[1]
        x = x[0]

        B, T, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)

        shifted_x = x

        x_windows_all = [shifted_x]
        x_window_masks_all = [None]

        # partition windows tuple(i // 2 for i in window_size)
        if self.focal_level > 1 and self.pool_method != "none":
            # if we add coarser granularity and the pool method is not none
            for k in range(self.focal_level - 1):
                window_size_glo = tuple(
                    math.floor(i / (2**k)) for i in self.window_size_glo)
                pooled_h = math.ceil(H / window_size_glo[0]) * (2**k)
                pooled_w = math.ceil(W / window_size_glo[1]) * (2**k)
                H_pool = pooled_h * window_size_glo[0]
                W_pool = pooled_w * window_size_glo[1]

                x_level_k = shifted_x
                # trim or pad shifted_x depending on the required size
                if H > H_pool:
                    trim_t = (H - H_pool) // 2
                    trim_b = H - H_pool - trim_t
                    x_level_k = x_level_k[:, :, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = (H_pool - H) // 2
                    pad_b = H_pool - H - pad_t
                    x_level_k = F.pad(x_level_k, (0, 0, 0, 0, pad_t, pad_b))

                if W > W_pool:
                    trim_l = (W - W_pool) // 2
                    trim_r = W - W_pool - trim_l
                    x_level_k = x_level_k[:, :, :, trim_l:-trim_r]
                elif W < W_pool:
                    pad_l = (W_pool - W) // 2
                    pad_r = W_pool - W - pad_l
                    x_level_k = F.pad(x_level_k, (0, 0, pad_l, pad_r))

                x_windows_noreshape = window_partition_noreshape(
                    x_level_k.contiguous(), window_size_glo
                )  # B, nWh, nWw, T, window_size_h, window_size_w, C
                nWh, nWw = x_windows_noreshape.shape[1:3]
                x_windows_noreshape = x_windows_noreshape.view(
                    B, nWh, nWw, T, window_size_glo[0] * window_size_glo[1],
                    C).transpose(4, 5)  # B, nWh, nWw, T, C, window_size_h*window_size_w
                x_windows_pooled = self.pool_layers[k](
                    x_windows_noreshape).flatten(-2)  # B, nWh, nWw, T, C | window被池化聚合

                x_windows_all += [x_windows_pooled]
                x_window_masks_all += [None]

        # nW*B, T*window_size*window_size, C
        if not self.memory:
            # default
            attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)
        else:
            # memory build in, with l_t as input
            attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all, l_t=l_t)

        # merge windows
        attn_windows = attn_windows.view(-1, T, self.window_size[0],
                                         self.window_size[1], C)    # _, T, nWh, nWw, C
        shifted_x = window_reverse(attn_windows, self.window_size, T, H,
                                   W)  # B T H' W' C, 从window格式变回token格式

        # FFN
        x = shortcut + shifted_x
        y = self.norm2(x)

        # default manner
        # MixF3N需要额外传递H, W
        x = x + self.mlp(y.view(B, T * H * W, C), output_size, T, H, W).view(
            B, T, H, W, C)

        return x, output_size, l_t