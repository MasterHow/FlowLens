''' Towards An End-to-End Framework for Video Inpainting
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.flow_comp import SPyNet
from model.modules.maskflownets import MaskFlowNetS
from model.modules.feat_prop import BidirectionalPropagation, SecondOrderDeformableAlignment
from model.modules.mix_focal_transformer import MixFocalTransformerBlock, SoftSplit, SoftComp
from model.modules.spectral_norm import spectral_norm as _spectral_norm


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Encoder(nn.Module):
    def __init__(self, out_channel=128, reduction=1):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64//reduction, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64//reduction, 64//reduction, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64//reduction, 128//reduction, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128//reduction, 256//reduction, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256//reduction, 384//reduction, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640//reduction, 512//reduction, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768//reduction, 384//reduction, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640//reduction, 256//reduction, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512//reduction, out_channel, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, _, _ = x.size()
        # h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class InpaintGenerator(BaseNetwork):
    """
        window_size                   # 窗口的尺寸，相当于patch数量除以4
        output_size                   # 输出的尺寸，训练尺寸//4
        spy_net                       # 如果为True，使用spynet替换maskflownets计算光流
        freeze_dcn                    # 如果为True，冻结dcn参数
    """
    def __init__(self, init_weights=True, freeze_dcn=False, spy_net=False,
                 flow_res=0.25,
                 depths=9,
                 window_size=None, output_size=None, small_model=False):
        super(InpaintGenerator, self).__init__()

        if not small_model:
            # large model:
            channel = 256   # default
            hidden = 512    # default
            reduction = 1   # default
        else:
            # v2
            channel = 128
            hidden = 256
            reduction = 2

        # 设置transformer参数
        # 设置trans block的数量
        if depths is None:
            depths = 2  # 0.08s/frame, 0.07s/frame with hidden = 128,
        else:
            depths = depths

        # 只有一个stage
        # 设置不同层的head数量
        # 默认条件下每层都使用4个head，相当于宽度和高度各2个head
        num_heads = [4] * depths

        # 光流推理分辨率
        self.flow_res = flow_res

        # encoder
        self.encoder = Encoder(out_channel=channel // 2, reduction=reduction)

        # decoder-default
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        # feature propagation module
        self.feat_prop_module = BidirectionalPropagation(channel // 2, freeze_dcn=freeze_dcn)

        # soft split and soft composition
        kernel_size = (7, 7)    # 滑块的大小
        padding = (3, 3)    # 两个方向上隐式填0的数量
        stride = (3, 3)     # 滑块的步长
        if output_size is None:
            # 默认的输出尺寸
            output_size = (60, 108)
        else:
            output_size = (output_size[0], output_size[1])
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }

        self.ss = SoftSplit(channel // 2,
                            hidden,
                            kernel_size,
                            stride,
                            padding,
                            t2t_param=t2t_params)

        self.sc = SoftComp(channel // 2, hidden, kernel_size, stride, padding)

        n_vecs = 1  # 计算token的数量
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] -
                           (d - 1) - 1) / stride[i] + 1)

        blocks = []
        if window_size is None:
            window_size = [(5, 9)] * depths
            focal_windows = [(5, 9)] * depths
        else:
            window_size_h = window_size[0]
            window_size_w = window_size[1]
            window_size = [(window_size_h, window_size_w)] * depths
            focal_windows = [(window_size_h, window_size_w)] * depths
        focal_levels = [2] * depths
        pool_method = "fc"

        # default temporal focal transformer
        for i in range(depths):
            # 只有第一层有记忆力
            if (i + 1) == 1:
                # 第一层有记忆力
                blocks.append(
                    MixFocalTransformerBlock(dim=hidden,
                                             num_heads=num_heads[i],
                                             window_size=window_size[i],
                                             focal_level=focal_levels[i],
                                             focal_window=focal_windows[i],
                                             n_vecs=n_vecs,
                                             t2t_params=t2t_params,
                                             pool_method=pool_method,
                                             memory=True,
                                             cs_win_strip=1), )

            else:
                # 后面的层没有记忆
                blocks.append(
                    MixFocalTransformerBlock(dim=hidden,
                                             num_heads=num_heads[i],
                                             window_size=window_size[i],
                                             focal_level=focal_levels[i],
                                             focal_window=focal_windows[i],
                                             n_vecs=n_vecs,
                                             t2t_params=t2t_params,
                                             pool_method=pool_method,
                                             memory=False))

        self.transformer = nn.Sequential(*blocks)

        if init_weights:
            self.init_weights()
            # Need to initial the weights of MSDeformAttn specifically
            for m in self.modules():
                if isinstance(m, SecondOrderDeformableAlignment):
                    m.init_offset()

        # flow completion network
        if spy_net:
            # 使用SpyNet
            self.update_MFN = SPyNet()
        else:
            # 使用MaskFlowNetS
            self.update_MFN = MaskFlowNetS()

    def forward_bidirect_flow(self, masked_local_frames):
        b, l_t, c, h, w = masked_local_frames.size()
        scale_factor = int(1 / self.flow_res)   # 1/4 -> 4 用来恢复尺度

        # compute forward and backward flows of masked frames
        masked_local_frames = F.interpolate(masked_local_frames.view(
            -1, c, h, w),
                                            scale_factor=self.flow_res,     # 1/4 for default
                                            mode='bilinear',
                                            align_corners=True,
                                            recompute_scale_factor=True)
        masked_local_frames = masked_local_frames.view(b, l_t, c, h // scale_factor,
                                                       w // scale_factor)
        mlf_1 = masked_local_frames[:, :-1, :, :, :].reshape(
            -1, c, h // scale_factor, w // scale_factor)
        mlf_2 = masked_local_frames[:, 1:, :, :, :].reshape(
            -1, c, h // scale_factor, w // scale_factor)
        pred_flows_forward = self.update_MFN(mlf_1, mlf_2)
        pred_flows_backward = self.update_MFN(mlf_2, mlf_1)

        pred_flows_forward = pred_flows_forward.view(b, l_t - 1, 2, h // scale_factor,
                                                     w // scale_factor)
        pred_flows_backward = pred_flows_backward.view(b, l_t - 1, 2, h // scale_factor,
                                                       w // scale_factor)

        # 最后必须resize到1/4来做feature warping
        if scale_factor != 4:
            pred_flows_forward = F.interpolate(pred_flows_forward.view(-1, 2, h // scale_factor, w // scale_factor),
                                            scale_factor=scale_factor / 4,
                                            mode='bilinear',
                                            align_corners=True,
                                            recompute_scale_factor=True).view(b, l_t - 1, 2, h // 4,
                                                     w // 4)
            pred_flows_backward = F.interpolate(pred_flows_backward.view(-1, 2, h // scale_factor, w // scale_factor),
                                               scale_factor=scale_factor / 4,
                                               mode='bilinear',
                                               align_corners=True,
                                               recompute_scale_factor=True).view(b, l_t - 1, 2, h // 4,
                                                                                 w // 4)

        return pred_flows_forward, pred_flows_backward

    def forward(self, masked_frames, num_local_frames=5):
        l_t = num_local_frames
        b, t, ori_c, ori_h, ori_w = masked_frames.size()

        # normalization before feeding into the flow completion module
        masked_local_frames = (masked_frames[:, :l_t, ...] + 1) / 2
        pred_flows = self.forward_bidirect_flow(masked_local_frames)

        # extracting features and performing the feature propagation on local features
        enc_feat = self.encoder(masked_frames.view(b * t, ori_c, ori_h, ori_w))
        _, c, h, w = enc_feat.size()
        fold_output_size = (h, w)
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]

        local_feat = self.feat_prop_module(local_feat, pred_flows[1],
                                           pred_flows[0])

        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        # content hallucination through stacking multiple temporal focal transformer blocks
        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_output_size)   # [B, t, f_h, f_w, hidden]

        trans_feat = self.transformer([trans_feat, fold_output_size, l_t])  # 比默认行为多传一个lt

        # 软组合
        trans_feat = self.sc(trans_feat[0], t, fold_output_size)

        trans_feat = trans_feat.view(b, t, -1, h, w)
        enc_feat = enc_feat + trans_feat    # 残差链接

        # decode frames from features
        output = self.decoder(enc_feat.view(b * t, c, h, w))
        output = torch.tanh(output)
        return output, pred_flows


# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(3, 5, 5),
                      stride=(1, 2, 2),
                      padding=(1, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module