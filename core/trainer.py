import os
import glob
import logging
import importlib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from core.loss import AdversarialLoss
from core.dataset import TrainDataset
from model.modules.flow_comp import FlowCompletionLoss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.num_local_frames = config['train_data_loader']['num_local_frames']
        self.num_ref_frames = config['train_data_loader']['num_ref_frames']
        self.spynet_lr = config['trainer'].get('spynet_lr', 1.0)

        # setup data set and data loader
        self.same_mask = False      # 是否在切换到下一个视频前使用相同的mask
        self.train_dataset = TrainDataset(config['train_data_loader'],
                                          batch_size=config['trainer']['batch_size'],
                                          same_mask=self.same_mask)

        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])

        # 记忆训练的序列化训练loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)

        # set loss functions
        self.adversarial_loss = AdversarialLoss(
            type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()

        if config['model']['net'] == 'flowlens':

            # 是否使用spynet作为光流补全网络，目前仅用于消融实验
            config['model']['spy_net'] = config['model'].get('spy_net', 0)
            if config['model']['spy_net'] != 0:
                # default for FlowLens-S
                self.spy_net = True
            else:
                # default for FlowLens
                self.spy_net = False

            # 是否使用maskflownet-s作为教师 光流补全网络，目前仅用于消融实验，仅对于SpyNet网络生效
            config['model']['mfn_teach'] = config['model'].get('mfn_teach', 1)
            if config['model']['mfn_teach'] != 0:
                # default
                self.mfn_teach = True
            else:
                self.mfn_teach = False

            # 用哪个网络计算loss
            if self.spy_net:
                if not self.mfn_teach:
                    # spy net
                    self.flow_comp_loss = FlowCompletionLoss(estimator='spy').to(self.config['device'])  # default
                else:
                    # mask flow net s 监督 spynet
                    self.flow_comp_loss = \
                        FlowCompletionLoss(estimator='mfn', device=self.config['device']).to(self.config['device'])
            else:
                # mask flow net s 监督 mask flow net s
                self.flow_comp_loss = \
                    FlowCompletionLoss(estimator='mfn', device=self.config['device']).to(self.config['device'])

        # setup models including generator and discriminator
        net = importlib.import_module('model.' + config['model']['net'])

        # 定义transformer的深度
        if config['model']['depths'] != 0:
            self.depths = config['model']['depths']
        else:
            # 使用网络默认的深度
            self.depths = None

        # 定义trans block的window个数(token除以window划分大小)
        if config['model']['window_size'] != 0:
            self.window_size = config['model']['window_size']
        else:
            # 使用网络默认的window
            self.window_size = None

        # 定义trans block的输出大小
        if config['model']['output_size'] != 0:
            self.output_size = config['model']['output_size']
        else:
            # 使用网络默认的output_size
            self.output_size = None

        # 定义是大模型还是小模型
        if config['model']['small_model'] != 0:
            self.small_model = True
        else:
            self.small_model = False

        # 是否冻结dcn参数
        config['model']['freeze_dcn'] = config['model'].get('freeze_dcn', 0)
        if config['model']['freeze_dcn'] != 0:
            self.freeze_dcn = True
        else:
            # default
            self.freeze_dcn = False

        # 使用mix-focal主干
        self.netG = net.InpaintGenerator(
            freeze_dcn=self.freeze_dcn,
            spy_net=self.spy_net,
            depths=self.depths,
            window_size=self.window_size, output_size=self.output_size, small_model=self.small_model)

        print(self.netG)
        self.netG = self.netG.to(self.config['device'])
        if not self.config['model']['no_dis']:
            self.netD = net.Discriminator(
                in_channels=3,
                use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            self.netD = self.netD.to(self.config['device'])

        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()

        if config['distributed']:
            self.netG = DDP(self.netG,
                            device_ids=[self.config['local_rank']],
                            output_device=self.config['local_rank'],
                            broadcast_buffers=True,
                            find_unused_parameters=True)
            if not self.config['model']['no_dis']:
                self.netD = DDP(self.netD,
                                device_ids=[self.config['local_rank']],
                                output_device=self.config['local_rank'],
                                broadcast_buffers=True,
                                find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        spynet_params = []
        for name, param in self.netG.named_parameters():
            if 'update_spynet' in name:
                spynet_params.append(param)
            else:
                backbone_params.append(param)

        optim_params = [
            {
                'params': backbone_params,
                'lr': self.config['trainer']['lr']
            },
            {  # finetuning learning rate for spynet
                'params': spynet_params,
                'lr': self.config['trainer']['lr'] * self.spynet_lr
            },
        ]

        if not self.freeze_dcn:
            # default manner
            self.optimG = torch.optim.Adam(optim_params,
                                           betas=(self.config['trainer']['beta1'],
                                                  self.config['trainer']['beta2']))
        else:
            # freeze dcn in opt
            self.optimG = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, optim_params[0]['params'])},
                                            {'params': filter(lambda p: p.requires_grad, optim_params[1]['params']),
                                             'lr': self.config['trainer']['lr'] * self.spynet_lr}],
                                           lr=self.config['trainer']['lr'],
                                           betas=(self.config['trainer']['beta1'],
                                                  self.config['trainer']['beta2']))

        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(
                self.netD.parameters(),
                lr=self.config['trainer']['lr'],
                betas=(self.config['trainer']['beta1'],
                       self.config['trainer']['beta2']))

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config['trainer']['scheduler']
        scheduler_type = scheduler_opt.pop('type')

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheG = MultiStepRestartLR(
                self.optimG,
                milestones=scheduler_opt['milestones'],
                gamma=scheduler_opt['gamma'])
            self.scheD = MultiStepRestartLR(
                self.optimD,
                milestones=scheduler_opt['milestones'],
                gamma=scheduler_opt['gamma'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheG = CosineAnnealingRestartLR(
                self.optimG,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
            self.scheD = CosineAnnealingRestartLR(
                self.optimD,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()
        self.scheD.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimG.param_groups[0]['lr']

    def add_summary(self, writer, name, val):
        """Add tensorboard summary."""
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0

    def load(self):
        """Load netG (and netD)."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'),
                                'r').read().splitlines()[-1]
        else:
            ckpts = [
                os.path.basename(i).split('.pth')[0]
                for i in glob.glob(os.path.join(model_path, '*.pth'))
            ]
            ckpts.sort()
            latest_epoch = ckpts[-1].split('_')[-1] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path,
                                    f'gen_{int(latest_epoch):06d}.pth')
            dis_path = os.path.join(model_path,
                                    f'dis_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path,
                                    f'opt_{int(latest_epoch):06d}.pth')

            if self.config['global_rank'] == 0:
                print(f'Loading model from {gen_path}...')

            dataG = torch.load(gen_path, map_location='cpu')
            self.netG.load_state_dict(dataG)

            if not self.config['model']['no_dis']:
                dataD = torch.load(dis_path,
                                   map_location='cpu')
                self.netD.load_state_dict(dataD)

            data_opt = torch.load(opt_path, map_location='cpu')

            try:
                self.optimG.load_state_dict(data_opt['optimG'])
            except:
                # fine-tune时优化器参数可能不同
                pass

            self.scheG.load_state_dict(data_opt['scheG'])
            if not self.config['model']['no_dis']:
                self.optimD.load_state_dict(data_opt['optimD'])
                self.scheD.load_state_dict(data_opt['scheD'])
            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']

        else:
            if self.config['global_rank'] == 0:
                print('Warnning: There is no trained model found.'
                      'An initialized model will be used.')

    def save(self, it):
        """Save parameters every eval_epoch"""
        if self.config['global_rank'] == 0:
            # configure path
            gen_path = os.path.join(self.config['save_dir'],
                                    f'gen_{it:06d}.pth')
            dis_path = os.path.join(self.config['save_dir'],
                                    f'dis_{it:06d}.pth')
            opt_path = os.path.join(self.config['save_dir'],
                                    f'opt_{it:06d}.pth')
            print(f'\nsaving model to {gen_path} ...')

            # remove .module for saving
            if isinstance(self.netG, torch.nn.DataParallel) \
               or isinstance(self.netG, DDP):
                netG = self.netG.module
                if not self.config['model']['no_dis']:
                    netD = self.netD.module
            else:
                netG = self.netG
                if not self.config['model']['no_dis']:
                    netD = self.netD

            # save checkpoints
            torch.save(netG.state_dict(), gen_path)
            if not self.config['model']['no_dis']:
                # 可以选择不存储判别器，opt存下来的是优化器的状态
                torch.save(netD.state_dict(), dis_path)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict(),
                        'scheG': self.scheG.state_dict(),
                        'scheD': self.scheD.state_dict()
                    }, opt_path)
            else:
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'scheG': self.scheG.state_dict()
                    }, opt_path)

            latest_path = os.path.join(self.config['save_dir'], 'latest.ckpt')
            os.system(f"echo {it:06d} > {latest_path}")

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar,
                        initial=self.iteration,
                        dynamic_ncols=True,
                        smoothing=0.01)

        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
            "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log",
            filemode='w')

        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            # 序列视频输入用于记忆力训练
            if not self.same_mask:
                # 使用随机mask训练
                self._train_epoch(pbar)

            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    def _train_epoch(self, pbar):
        """Process input and calculate loss every training epoch in a sequence manner with memory"""
        device = self.config['device']

        for frames, masks, video_name, index, start_index in self.train_loader:
            self.iteration += 1

            # 当有新视频出现时，即start_index为0时，清空记忆缓存
            for start_idx in start_index:
                if start_idx == 0:
                    if self.config['world_size'] == 1:
                        # 单卡清除记忆缓存
                        for blk_idx in range(0, len(self.netG.transformer)):
                            try:
                                # 清空有记忆力的层的记忆缓存
                                self.netG.transformer[blk_idx].attn.m_k = []
                                self.netG.transformer[blk_idx].attn.m_v = []
                            except:
                                pass
                    else:
                        # 多卡清除记忆缓存
                        for blk_idx in range(0, len(self.netG.module.transformer)):
                            try:
                                # 清空有记忆力的层的记忆缓存 原版这里有bug没有清除缓存
                                self.netG.module.transformer[blk_idx].attn.m_k = []
                                self.netG.module.transformer[blk_idx].attn.m_v = []
                            except:
                                pass

            frames, masks = frames.to(device), masks.to(device)
            l_t = self.num_local_frames
            b, t, c, h, w = frames.size()

            masked_frames = (frames * (1 - masks).float())
            gt_local_frames = (frames[:, :l_t, ...] + 1) / 2

            pred_imgs, pred_flows = self.netG(masked_frames, l_t)
            pred_imgs = pred_imgs.view(b, -1, c, h, w)
            comp_imgs = frames * (1. - masks) + masks * pred_imgs

            # compute flow completion loss
            flow_loss = self.flow_comp_loss(pred_flows, gt_local_frames)

            gen_loss = 0
            dis_loss = 0

            if not self.config['model']['no_dis']:
                # discriminator adversarial loss
                real_clip = self.netD(frames)
                fake_clip = self.netD(comp_imgs.detach())
                dis_real_loss = self.adversarial_loss(real_clip, True, True)
                dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2
                self.add_summary(self.dis_writer, 'loss/dis_vid_fake',
                                 dis_fake_loss.item())
                self.add_summary(self.dis_writer, 'loss/dis_vid_real',
                                 dis_real_loss.item())
                self.optimD.zero_grad()
                dis_loss.backward()
                self.optimD.step()

                # generator adversarial loss
                gen_clip = self.netD(comp_imgs)
                gan_loss = self.adversarial_loss(gen_clip, True, False)
                gan_loss = gan_loss \
                           * self.config['losses']['adversarial_weight']
                gen_loss += gan_loss
                self.add_summary(self.gen_writer, 'loss/gan_loss',
                                 gan_loss.item())

            flow_loss = flow_loss * self.config['losses']['flow_weight']
            gen_loss += flow_loss
            self.add_summary(self.gen_writer, 'loss/flow_loss',
                             flow_loss.item())

            # generator l1 loss
            hole_loss = self.l1_loss(pred_imgs * masks, frames * masks)
            # 空洞内的loss
            hole_loss = hole_loss / torch.mean(masks) \
                        * self.config['losses']['hole_weight']
            gen_loss += hole_loss
            self.add_summary(self.gen_writer, 'loss/hole_loss',
                             hole_loss.item())

            valid_loss = self.l1_loss(pred_imgs * (1 - masks),
                                      frames * (1 - masks))
            # 非遮挡区域的loss
            valid_loss = valid_loss / torch.mean(1 - masks) \
                         * self.config['losses']['valid_weight']
            gen_loss += valid_loss
            self.add_summary(self.gen_writer, 'loss/valid_loss',
                             valid_loss.item())

            self.optimG.zero_grad()
            # gen loss 是对抗、光流、空洞和非遮挡区域loss之和
            gen_loss.backward()
            self.optimG.step()

            self.update_learning_rate()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                if not self.config['model']['no_dis']:
                    pbar.set_description((f"flow: {flow_loss.item():.3f}; "
                                          f"d: {dis_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}"))
                else:
                    pbar.set_description((f"flow: {flow_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}"))

                if self.iteration % self.train_args['log_freq'] == 0:
                    if not self.config['model']['no_dis']:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"flow: {flow_loss.item():.4f}; "
                                     f"d: {dis_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}")
                    else:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"flow: {flow_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}")

            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args['iterations']:
                break
