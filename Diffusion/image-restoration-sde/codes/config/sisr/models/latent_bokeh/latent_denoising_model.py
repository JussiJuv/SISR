import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion
from models.modules.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        #os.makedirs('image', exist_ok=True)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)
        self.latent_model = networks.define_L(opt).to(self.device)

        for param in self.latent_model.parameters():
                param.requires_grad = False

        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        
        self.load()

        #self.encode = self.latent_model.encode
        #self.decode = self.latent_model.decode

        # Load Latent Stats for Normalization
        stats_path = opt['latent_config'].get('stats_path')
        if stats_path and os.path.exists(stats_path):
            stats = torch.load(stats_path)
            self.latent_mean = stats['mean'].view(1, -1, 1, 1).to(self.device)
            self.latent_std = stats['std'].view(1, -1, 1, 1).to(self.device)
            print(f"Successfully loaded latent stats from {stats_path}")
        else:
            self.latent_mean = torch.tensor(0.0).to(self.device)
            self.latent_std = torch.tensor(1.0).to(self.device)
            print("Warning: No latent stats found. Training might be unstable.")

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()
            self.use_amp = bool(opt["train"].get("use_amp", True))
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def encode(self, img):
        img = (img + 1.0) * 0.5 
        z, hidden = self.latent_model.encode(img)
        z = (z - self.latent_mean) / self.latent_std
        return z, hidden

    def decode(self, z, hidden):
        z = (z * self.latent_std) + self.latent_mean
        out = self.latent_model.decode(z, hidden)
        out = (out * 2.0) - 1.0
        return out

    def feed_data(
        self,
        state,
        LQ,
        GT=None,
        *,
        latent_inputs=False,
        hidden=None,
        raw_LQ=None,
        raw_GT=None,
        src_lens=None,
        tgt_lens=None,
        disparity=None,
        alpha=None,
    ):
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.disparity = disparity
        self.alpha = alpha

        with torch.no_grad():
            self.state = state.to(self.device)

            if latent_inputs:
                # LQ and GT are already latent tensors here.
                self.condition = LQ.to(self.device)
                self.hidden_lq = hidden

                # Keep pixel-space copies for logging / saved images.
                self.raw_lq = raw_LQ.to(self.device) if raw_LQ is not None else None
                self.raw_gt = raw_GT.to(self.device) if raw_GT is not None else None

                # GT used by the loss must stay in latent space.
                self.state_0 = GT.to(self.device) if GT is not None else None
            else:
                # Pixel inputs: encode here.
                LQ_pix = (LQ.to(self.device) + 1.0) * 0.5
                z_lq, self.hidden_lq = self.latent_model.encode(LQ_pix)
                self.condition = (z_lq - self.latent_mean) / self.latent_std

                self.raw_lq = LQ.to(self.device)

                if GT is not None:
                    GT_pix = (GT.to(self.device) + 1.0) * 0.5
                    z_gt, _ = self.latent_model.encode(GT_pix)
                    self.state_0 = (z_gt - self.latent_mean) / self.latent_std
                    self.raw_gt = GT.to(self.device)
                else:
                    self.state_0 = None
                    self.raw_gt = None

    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad(set_to_none=True)
        timesteps = timesteps.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            noise = sde.noise_fn(self.state, timesteps.view(-1))
            score = sde.get_score_from_noise(noise, timesteps)

            xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
            xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
            loss = self.weight * self.loss_fn(xt_1_expection, xt_1_optimum)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.ema.update()

        self.log_dict["loss"] = loss.item()


    def test(self, sde=None, hidden=None, perform_ode=False, save_states=False):
        if hidden is None and hasattr(self, "hidden_lq"):
            hidden = self.hidden_lq

        sde.set_mu(self.condition)

        #lens_info = [self.src_lens, self.tgt_lens, self.disparity]

        self.model.eval()
        with torch.no_grad():
            if not perform_ode:
                # for SDE
                #latent = sde.reverse_sde(self.state, save_states=save_states, lens_info=lens_info)
                latent = sde.reverse_sde(self.state, save_states=save_states)
            else:
                # if perform Denoising ODE
                latent = sde.reverse_ode(self.state, save_states=save_states)

            self.full_lq = self.decode(self.condition, hidden)
            self.output = self.decode(latent, hidden)

        self.model.train()

        """ tvutils.save_image(self.condition[:, :4].data, f'image/condition.png', normalize=False)
        tvutils.save_image(self.state[:, :4].data, f'image/state.png', normalize=False)
        tvutils.save_image(latent[:, :4].data, f'image/latent.png', normalize=False)
        
        tvutils.save_image(self.full_lq.data, f'image/LQ.png', normalize=False)
        tvutils.save_image(self.output.data, f'image/SR.png', normalize=False)
        if self.state_0 is not None:
            tvutils.save_image(self.state_0.data, f'image/GT.png', normalize=False) """


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()

        if self.opt.get("space_mode") == "latent":
            # Prefer stored pixel tensors if available.
            input_img = self.raw_lq if getattr(self, "raw_lq", None) is not None else self.full_lq
            out_dict["Input"] = input_img.detach()[0].float().cpu()

            out_dict["Output"] = self.output.detach()[0].float().cpu()

            if need_GT:
                if getattr(self, "raw_gt", None) is not None:
                    out_dict["GT"] = self.raw_gt.detach()[0].float().cpu()
                elif self.state_0 is not None:
                    gt_pix = self.decode(self.state_0, self.hidden_lq)
                    out_dict["GT"] = gt_pix.detach()[0].float().cpu()
        else:
            out_dict["Input"] = self.condition.detach()[0].float().cpu()
            out_dict["Output"] = self.output.detach()[0].float().cpu()
            if need_GT:
                out_dict["GT"] = self.state_0.detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

        load_path_L = self.opt["path"]["pretrain_model_L"]
        if load_path_L is not None:
            logger.info("Loading model for L [{:s}] ...".format(load_path_L))
            self.load_network(load_path_L, self.latent_model, self.opt["path"]["strict_load"])


    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        # self.save_network(self.ema.ema_model, "EMA", 'lastest')
        if hasattr(self, 'ema'):
            self.save_network(self.ema.ema_model, "EMA", iter_label)
