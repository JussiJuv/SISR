# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable

import torch
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount
from training.latent_normalization import normalize_latent

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 1


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
    data_loader_val=None,
    eval_model=None,
    latent_ae=None,
    latent_stats=None,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    if args.space_mode == "latent" and latent_ae is None:
        raise ValueError("Latent mode requires a loaded latent autoencoder.")
    if args.space_mode == "latent" and args.discrete_flow_matching:
        raise ValueError("Latent mode currently supports only continuous flow matching.")
    
    if args.space_mode == "latent" and latent_stats is None:
        raise ValueError("latent_stats must be provided in latent mode")
        
    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
    else:
        path = CondOTProbPath()

    for data_iter_step, (hr, lr, _) in enumerate(data_loader):
        hr = hr.to(device, non_blocking=True)
        lr = lr.to(device, non_blocking=True)

        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        if args.discrete_flow_matching:
            hr_discrete = (hr * 255.0).to(torch.long)
            t = torch.rand(hr.shape[0], device=device)
            x_0 = (torch.zeros(hr.shape, dtype=torch.long, device=device) + MASK_TOKEN)
            path_sample = path.sample(t=t, x_0=x_0, x_1=hr_discrete)
            
            logits = model(path_sample.x_t, t=t, extra={"condition": lr}) 
            loss = torch.nn.functional.cross_entropy(
                logits.reshape([-1, 257]), hr_discrete.reshape([-1])
            ).mean()
        else:
            if args.space_mode == "latent":
                with torch.no_grad():
                    hr_pix = (hr + 1.0) * 0.5
                    lr_pix = (lr + 1.0) * 0.5
                    """ latent_lr, _ = latent_ae.encode(lr_pix)
                    latent_hr, _ = latent_ae.encode(hr_pix) """
                    latent_lr, _ = latent_ae.encode(lr_pix)
                    latent_hr, _ = latent_ae.encode(hr_pix)

                    latent_lr = normalize_latent(latent_lr, latent_stats["mean"], latent_stats["std"])
                    latent_hr = normalize_latent(latent_hr, latent_stats["mean"], latent_stats["std"])

                x_1 = latent_hr
                cond = latent_lr

                if args.warm_start_lr:
                    x_0 = latent_lr
                else:
                    x_0 = torch.randn_like(latent_hr)

            else:
                # Original pixel-space path
                hr_norm = hr
                lr_norm = lr
                x_1 = hr_norm
                cond = lr_norm

                if args.warm_start_lr:
                    x_0 = lr_norm
                else:
                    x_0 = torch.randn_like(hr_norm)

            if args.skewed_timesteps:
                t = skewed_timestep_sample(x_1.shape[0], device=device)
            else:
                t = torch.rand(x_1.shape[0], device=device)

            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            with torch.cuda.amp.autocast():
                if args.model == "nafnet":
                    actual_model = model.model if isinstance(model, EMA) else model
                    if isinstance(actual_model, DistributedDataParallel):
                        actual_model = actual_model.module

                    v_pred = actual_model(x_t, cond, t)
                else:
                    model_input = torch.cat([x_t, cond], dim=1)
                    v_pred = model(model_input, t, extra={})

                loss = torch.abs(v_pred - u_t).mean()

        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Backpropagation
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
            clip_grad=1.0,
        )
        
        # Metric updates
        batch_loss.update(loss * accum_iter)
        epoch_loss.update(loss * accum_iter)

        if apply_update:
            if isinstance(model, EMA):
                model.update_ema()
            elif isinstance(model, DistributedDataParallel) and isinstance(model.module, EMA):
                model.module.update_ema()

        current_lr = optimizer.param_groups[0]["lr"]
        #if data_iter_step % PRINT_FREQUENCY == 0:
        if data_iter_step % args.log_steps == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: "
                f"loss = {epoch_loss.compute():.6f}, lr = {current_lr:.8f}"
            )
            for handler in logger.handlers:
                handler.flush()
        # Check if we should validate based on the global step or iter step
        if args.val_freq > 0 and data_iter_step > 0 and data_iter_step % args.val_freq == 0:
            if eval_model is not None and data_loader_val is not None:
                logger.info(f"--- Running mid-epoch validation at step {data_iter_step} ---")
                eval_model(
                    model,
                    data_loader_val,
                    device,
                    epoch=epoch,
                    fid_samples=args.num_val_images,
                    args=args,
                )
                model.train(True)

    #lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
