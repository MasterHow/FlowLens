# -*- coding: utf-8 -*-
import cv2
import numpy as np
import importlib
import os
import time
import json
import random
import argparse
from PIL import Image

import torch
from torch.utils.data import DataLoader

from core.dataset import TestDataset
from core.metrics import calc_psnr_and_ssim, calculate_i3d_activations, calculate_vfid, init_i3d_model

# global variables
# w h can be changed by args.output_size
w, h = 432, 240     # default acc. test setting in e2fgvi for davis dataset and KITTI-EXO
# w, h = 336, 336     # default acc. test setting for KITTI-EXI
ref_length = 10     # non-local frames的步幅间隔，此处为每10帧取1帧NLF


def read_cfg(args):
    """read flowlens cfg from config file"""
    # loading configs
    config = json.load(open(args.cfg_path))

    # # # # pass config to args # # # #
    args.dataset = config['train_data_loader']['name']
    args.data_root = config['train_data_loader']['data_root']
    args.output_size = [432, 240]
    args.output_size[0], args.output_size[1] = (config['train_data_loader']['w'], config['train_data_loader']['h'])
    args.model_win_size = config['model'].get('window_size', None)
    args.model_output_size = config['model'].get('output_size', None)
    args.neighbor_stride = config['train_data_loader'].get('num_local_frames', 10)

    # 是否使用spynet作为光流补全网络 (FlowLens-S)
    config['model']['spy_net'] = config['model'].get('spy_net', 0)
    if config['model']['spy_net'] != 0:
        # default for FlowLens-S
        args.spy_net = True
    else:
        # default for FlowLens
        args.spy_net = False

    if config['model']['net'] == 'flowlens':

        # 定义transformer的深度
        if config['model']['depths'] != 0:
            args.depths = config['model']['depths']
        else:
            # 使用网络默认的深度
            args.depths = None

        # 定义trans block的window个数(token除以window划分大小)
        config['model']['window_size'] = config['model'].get('window_size', 0)
        if config['model']['window_size'] != 0:
            args.window_size = config['model']['window_size']
        else:
            # 使用网络默认的window
            args.window_size = None

        # 定义是大模型还是小模型
        if config['model']['small_model'] != 0:
            args.small_model = True
        else:
            args.small_model = False

        # 是否冻结dcn参数
        config['model']['freeze_dcn'] = config['model'].get('freeze_dcn', 0)
        if config['model']['freeze_dcn'] != 0:
            args.freeze_dcn = True
        else:
            # default
            args.freeze_dcn = False

    # # # # pass config to args # # # #

    return args


# sample reference frames from the whole video with mem support
def get_ref_index_mem(length, neighbor_ids, same_id=False):
    """smae_id(bool): If True, allow same ref and local id as input."""
    ref_index = []
    for i in range(0, length, ref_length):
        if same_id:
            # 允许相同id
            ref_index.append(i)
        else:
            # 不允许相同的id，当出现相同id时找到最近的一个不同的i
            if i not in neighbor_ids:
                ref_index.append(i)
            else:
                lf_id_avg = sum(neighbor_ids)/len(neighbor_ids)     # 计算 local frame id 平均值
                for _iter in range(0, 100):
                    if i < (length - 1):
                        # 不能超过视频长度
                        if i == 0:
                            # 第0帧的时候重复，直接取到下一个 NLF + 5   +5是为了防止和下一个重复的 nlf id 改的id重复
                            i = ref_length + args.neighbor_stride
                            ref_index.append(i)
                            break
                        elif i < lf_id_avg:
                            # 往前找不重复的参考帧, 防止都往一个方向找而重复
                            i -= 1
                        else:
                            # 往后找不重复的参考帧
                            i += 1
                    else:
                        # 超过了直接用最后一帧，然后退出
                        ref_index.append(i)
                        break

                    if i not in neighbor_ids:
                        ref_index.append(i)
                        break

    return ref_index


# sample reference frames from the remain frames with random behavior like trainning
def get_ref_index_mem_random(neighbor_ids, video_length, num_ref_frame=3, before_nlf=False):
    if not before_nlf:
        # 从过去和未来采集非局部帧
        complete_idx_set = list(range(video_length))
    else:
        # 非局部帧只会从过去的视频帧中选取，不会使用未来的信息
        complete_idx_set = list(range(neighbor_ids[-1]))
    # complete_idx_set = list(range(video_length))

    remain_idx = list(set(complete_idx_set) - set(neighbor_ids))

    # 当只用过去的帧作为非局部帧时，可能会出现过去的帧数量少于非局部帧需求的问题，比如视频的一开始
    if before_nlf:
        if len(remain_idx) < num_ref_frame:
            # 则我们允许从局部帧中采样非局部帧 转换为set可以去除重复元素
            remain_idx = list(set(remain_idx + neighbor_ids))

    ref_index = sorted(random.sample(remain_idx, num_ref_frame))
    return ref_index


def main_worker(args):
    args = read_cfg(args=args)      # 读取网络的所有设置
    w = args.output_size[0]
    h = args.output_size[1]
    args.size = (w, h)

    # set up datasets and data loader
    # default result
    test_dataset = TestDataset(args)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)

    # set up models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)

    if args.model == 'flowlens':
        model = net.InpaintGenerator(freeze_dcn=args.freeze_dcn, spy_net=args.spy_net, depths=args.depths,
                                     window_size=args.model_win_size, output_size=args.model_output_size,
                                     small_model=args.small_model).to(device)
    else:
        # 加载一些尺寸窗口设置
        model = net.InpaintGenerator(window_size=args.model_win_size, output_size=args.model_output_size).to(device)

    if args.ckpt is not None:
        data = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(data)
        print(f'Loading from: {args.ckpt}')

    model.eval()

    total_frame_psnr = []
    total_frame_ssim = []

    output_i3d_activations = []
    real_i3d_activations = []

    print('Start evaluation...')

    time_all = 0
    len_all = 0

    # create results directory
    if args.ckpt is not None:
        ckpt = args.ckpt.split('/')[-1]
    else:
        ckpt = 'random'
    if args.fov is not None:
        if args.reverse:
            result_path = os.path.join('results', f'{args.model}+_{ckpt}_{args.fov}_{args.dataset}')
        else:
            result_path = os.path.join('results', f'{args.model}_{ckpt}_{args.fov}_{args.dataset}')
    else:
        if args.reverse:
            result_path = os.path.join('results', f'{args.model}+_{ckpt}_{args.dataset}')
        else:
            result_path = os.path.join('results', f'{args.model}_{ckpt}_{args.dataset}')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    eval_summary = open(
        os.path.join(result_path, f"{args.model}_{args.dataset}_metrics.txt"),
        "w")

    i3d_model = init_i3d_model()

    for index, items in enumerate(test_loader):

        for blk in model.transformer:
            try:
                blk.attn.m_k = []
                blk.attn.m_v = []
            except:
                pass

        frames, masks, video_name, frames_PIL = items

        video_length = frames.size(1)
        frames, masks = frames.to(device), masks.to(device)
        ori_frames = frames_PIL     # 原始帧，可视为真值
        ori_frames = [
            ori_frames[i].squeeze().cpu().numpy() for i in range(video_length)
        ]
        comp_frames = [None] * video_length     # 补全帧

        len_all += video_length

        # complete holes by our model
        # 当这个循环走完的时候，一段视频已经被补全了
        for f in range(0, video_length, args.neighbor_stride):
            if args.same_memory:
                # 尽可能与video in-painting的测试逻辑一致
                # 输入的时间维度T保持一致
                if (f - args.neighbor_stride > 0) and (f + args.neighbor_stride + 1 < video_length):
                    # 视频首尾均不会越界，不需要补充额外帧
                    neighbor_ids = [
                        i for i in range(max(0, f - args.neighbor_stride),
                                         min(video_length, f + args.neighbor_stride + 1))
                    ]  # neighbor_ids即为Local Frames, 局部帧
                else:
                    # 视频越界，补充额外帧保证记忆缓存的时间通道维度一致，后面也可以尝试放到trans里直接复制特征的时间维度
                    neighbor_ids = [
                        i for i in range(max(0, f - args.neighbor_stride),
                                         min(video_length, f + args.neighbor_stride + 1))
                    ]  # neighbor_ids即为Local Frames, 局部帧
                    repeat_num = (args.neighbor_stride * 2 + 1) - len(neighbor_ids)

                    lf_id_avg = sum(neighbor_ids) / len(neighbor_ids)  # 计算 local frame id 平均值
                    first_id = neighbor_ids[0]
                    for ii in range(0, repeat_num):
                        # 保证局部窗口的大小一致，防止缓存通道数变化
                        if lf_id_avg < (video_length // 2):
                            # 前半段视频也向前找局部id，防止和下一个窗口的输入完全一样
                            new_id = video_length - 1 - ii
                        else:
                            # 后半段视频向前找局部id
                            new_id = first_id - 1 - ii
                        neighbor_ids.append(new_id)

                    neighbor_ids = sorted(neighbor_ids)    # 重新排序

            else:
                # 与记忆力模型的训练逻辑一致
                if not args.recurrent:
                    if video_length < (f + args.neighbor_stride):
                        neighbor_ids = [
                            i for i in range(f, video_length)
                        ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次，视频尾部可能不足5帧局部帧，复制最后一帧补全数量
                        for repeat_idx in range(0, args.neighbor_stride - len(neighbor_ids)):
                            neighbor_ids.append(neighbor_ids[-1])
                    else:
                        neighbor_ids = [
                            i for i in range(f, f + args.neighbor_stride)
                        ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次
                else:
                    # 在recurrent模式下，每次局部窗口都为1
                    neighbor_ids = [f]

            # 为了保证时间维度一致, 允许输入相同id的帧
            if args.same_memory:
                ref_ids = get_ref_index_mem(video_length, neighbor_ids, same_id=False)  # ref_ids即为Non-Local Frames, 非局部帧
            elif args.past_ref:
                ref_ids = get_ref_index_mem_random(neighbor_ids, video_length, num_ref_frame=3, before_nlf=True)  # 只允许过去的参考帧
            else:
                ref_ids = get_ref_index_mem_random(neighbor_ids, video_length, num_ref_frame=3)  # 与序列训练同样的非局部帧输入逻辑

            ref_ids = sorted(ref_ids)  # 重新排序
            selected_imgs_lf = frames[:1, neighbor_ids, :, :, :]
            selected_imgs_nlf = frames[:1, ref_ids, :, :, :]
            selected_imgs = torch.cat((selected_imgs_lf, selected_imgs_nlf), dim=1)
            selected_masks_lf = masks[:1, neighbor_ids, :, :, :]
            selected_masks_nlf = masks[:1, ref_ids, :, :, :]
            selected_masks = torch.cat((selected_masks_lf, selected_masks_nlf), dim=1)

            with torch.no_grad():
                masked_frames = selected_imgs * (1 - selected_masks)

                torch.cuda.synchronize()
                time_start = time.time()

                pred_img, _ = model(masked_frames, len(neighbor_ids))   # forward里会输入局部帧数量来对两种数据分开处理

                # 水平与竖直翻转增强
                if args.reverse:
                    masked_frames_horizontal_aug = torch.from_numpy(masked_frames.cpu().numpy()[:, :, :, :, ::-1].copy()).cuda()
                    pred_img_horizontal_aug, _ = model(masked_frames_horizontal_aug, len(neighbor_ids))
                    pred_img_horizontal_aug = torch.from_numpy(pred_img_horizontal_aug.cpu().numpy()[:, :, :, ::-1].copy()).cuda()
                    masked_frames_vertical_aug = torch.from_numpy(masked_frames.cpu().numpy()[:, :, :, ::-1, :].copy()).cuda()
                    pred_img_vertical_aug, _ = model(masked_frames_vertical_aug, len(neighbor_ids))
                    pred_img_vertical_aug = torch.from_numpy(pred_img_vertical_aug.cpu().numpy()[:, :, ::-1, :].copy()).cuda()

                    pred_img = 1 / 3 * (pred_img + pred_img_horizontal_aug + pred_img_vertical_aug)

                torch.cuda.synchronize()
                time_end = time.time()
                time_sum = time_end - time_start
                time_all += time_sum

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + ori_frames[idx] * (1 - binary_masks[i])

                    if comp_frames[idx] is None:
                        # 如果第一次补全Local Frame中的某帧，直接记录到补全帧list (comp_frames) 里
                        # good_fusion下所有img多出一个‘次数’通道，用来记录所有的结果
                        comp_frames[idx] = img[np.newaxis, :, :, :]

                    # 直接把所有结果都记录下来，最后沿着通道平均
                    else:
                        comp_frames[idx] = np.concatenate((comp_frames[idx], img[np.newaxis, :, :, :]), axis=0)
                    ########################################################################################

        # 对于good_fusion, 推理一遍后需要沿着axis=0取平均
        for idx, comp_frame in zip(range(0, video_length), comp_frames):
            comp_frame = comp_frame.astype(np.float32).sum(axis=0)/comp_frame.shape[0]
            comp_frames[idx] = comp_frame

        # calculate metrics
        cur_video_psnr = []
        cur_video_ssim = []
        comp_PIL = []  # to calculate VFID
        frames_PIL = []
        for ori, comp in zip(ori_frames, comp_frames):
            psnr, ssim = calc_psnr_and_ssim(ori, comp)

            cur_video_psnr.append(psnr)
            cur_video_ssim.append(ssim)

            total_frame_psnr.append(psnr)
            total_frame_ssim.append(ssim)

            frames_PIL.append(Image.fromarray(ori.astype(np.uint8)))
            comp_PIL.append(Image.fromarray(comp.astype(np.uint8)))
        cur_psnr = sum(cur_video_psnr) / len(cur_video_psnr)
        cur_ssim = sum(cur_video_ssim) / len(cur_video_ssim)

        # saving i3d activations
        frames_i3d, comp_i3d = calculate_i3d_activations(frames_PIL,
                                                         comp_PIL,
                                                         i3d_model,
                                                         device=device)
        real_i3d_activations.append(frames_i3d)
        output_i3d_activations.append(comp_i3d)

        print(
            f'[{index+1:3}/{len(test_loader)}] Name: {str(video_name):25} | PSNR/SSIM: {cur_psnr:.4f}/{cur_ssim:.4f}'
        )
        eval_summary.write(
            f'[{index+1:3}/{len(test_loader)}] Name: {str(video_name):25} | PSNR/SSIM: {cur_psnr:.4f}/{cur_ssim:.4f}\n'
        )

        print('Average run time: (%f) per frame' % (time_all/len_all))

        # saving images for evaluating warpping errors
        if args.save_results:
            save_frame_path = os.path.join(result_path, video_name[0])
            os.makedirs(save_frame_path, exist_ok=False)

            for i, frame in enumerate(comp_frames):
                cv2.imwrite(
                    os.path.join(save_frame_path,
                                 str(i).zfill(5) + '.png'),
                    cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))

    avg_frame_psnr = sum(total_frame_psnr) / len(total_frame_psnr)
    avg_frame_ssim = sum(total_frame_ssim) / len(total_frame_ssim)

    fid_score = calculate_vfid(real_i3d_activations, output_i3d_activations)
    print('Finish evaluation... Average Frame PSNR/SSIM/VFID: '
          f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{fid_score:.3f}')
    eval_summary.write(
        'Finish evaluation... Average Frame PSNR/SSIM/VFID: '
        f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{fid_score:.3f}')
    eval_summary.close()

    print('All average forward run time: (%f) per frame' % (time_all / len_all))

    return len(total_frame_psnr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FlowLens')
    parser.add_argument('--cfg_path', default='configs/KITTI360EX-I_FlowLens_small_re.json')
    parser.add_argument('--dataset', choices=['KITTI360-EX'], type=str)       # 相当于train的‘name’
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--output_size', type=int, nargs='+', default=[432, 240])
    parser.add_argument('--object', action='store_true', default=False)     # if true, use object removal mask
    parser.add_argument('--fov', choices=['fov5', 'fov10', 'fov20'], type=str)  # 对于KITTI360-EX, 测试需要输入fov
    parser.add_argument('--past_ref', action='store_true', default=True)  # 对于KITTI360-EX, 测试时只允许使用之前的参考帧
    parser.add_argument('--model', choices=['flowlens'], type=str)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--same_memory', action='store_true', default=False,
                        help='test with memory ability in video in-painting style')
    parser.add_argument('--reverse', action='store_true', default=False,
                        help='test with horizontal and vertical reverse augmentation')
    parser.add_argument('--model_win_size', type=int, nargs='+', default=[5, 9])
    parser.add_argument('--model_output_size', type=int, nargs='+', default=[60, 108])
    parser.add_argument('--recurrent', action='store_true', default=False,
                        help='keep window = 1, stride = 1 to not use any local future info')
    args = parser.parse_args()

    if args.dataset == 'KITTI360-EX':
        # 对于KITTI360-EX, 测试时只允许使用之前的参考帧
        args.past_ref = True

    frame_num = main_worker(args)
