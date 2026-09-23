# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable
from datetime import datetime

import PIL.Image
import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from training import distributed_mode
from training.edm_time_discretization import get_time_discretization
from training.train_loop import MASK_TOKEN
from training.latent_normalization import normalize_latent, denormalize_latent

from torchvision.utils import save_image

import torch.nn.functional as F


logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, condition: torch.Tensor = None
    ):
        current_t = t if isinstance(t, (float, int)) else t.item()
        print(f" [Validation] Step {self.nfe_counter:3d} | Progress: {current_t*100:3.0f}%", end='\r')
        
        inner_model = self.model
        #if hasattr(inner_model, "model"): inner_model = inner_model.model # EMA
        #if hasattr(inner_model, "module"): inner_model = inner_model.module # DDP
        
        while hasattr(inner_model, "module") or hasattr(inner_model, "model"):
            if hasattr(inner_model, "module"):
                inner_model = inner_model.module
            elif hasattr(inner_model, "model"):
                inner_model = inner_model.model
        
        is_nafnet = "ConditionalNAFNet" in str(type(inner_model))
        t_vec = torch.zeros(x.shape[0], device=x.device) + t

        with torch.cuda.amp.autocast(), torch.no_grad():
            if is_nafnet:
                # NAFNet path: (x_t, condition, t)
                conditional = self.model(x, condition, t_vec)
                
                if cfg_scale != 0.0:
                    null_condition = torch.zeros_like(condition)
                    condition_free = self.model(x, null_condition, t_vec)
                    result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
                else:
                    result = conditional
            else:
                # Old UNet path: (cat_input, t, extra)
                model_input = torch.cat([x, condition], dim=1)
                conditional = self.model(model_input, t_vec, extra={})
                
                if cfg_scale != 0.0:
                    null_input = torch.cat([x, torch.zeros_like(condition)], dim=1)
                    condition_free = self.model(null_input, t_vec, extra={})
                    result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
                else:
                    result = conditional

        self.nfe_counter += 1
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
    latent_ae=None,
    latent_stats=None,
):
    fid_samples = args.num_val_images

    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)
    
    if args.space_mode == "latent" and latent_ae is None:
        raise ValueError("Latent mode requires a loaded latent autoencoder.")

    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        p = torch.zeros(size=[257], dtype=torch.float32, device=device)
        p[256] = 1.0
        solver = MixtureDiscreteEulerSolver(
            model=cfg_scaled_model,
            path=path,
            vocabulary_size=257,
            source_distribution_p=p,
        )
    else:
        solver = ODESolver(velocity_model=cfg_scaled_model)
        ode_opts = args.ode_options

    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )

    num_synthetic = 0
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "debug_triplets").mkdir(parents=True, exist_ok=True)

    for data_iter_step, (hr, lr, filenames) in enumerate(data_loader):
        if num_synthetic >= fid_samples:
            break

        hr = hr.to(device, non_blocking=True)
        lr = lr.to(device, non_blocking=True)

        if args.space_mode == "latent":
            with torch.no_grad():
                hr_pix = (hr + 1.0) * 0.5
                lr_pix = (lr + 1.0) * 0.5
                latent_lr, h_lr = latent_ae.encode(lr_pix)
                latent_lr = normalize_latent(latent_lr, latent_stats["mean"], latent_stats["std"])

            fid_metric.update(hr_pix, real=True)

            cfg_scaled_model.reset_nfe_counter()

            if args.warm_start_lr:
                x_0 = latent_lr
            else:
                x_0 = torch.randn_like(latent_lr)

            if args.discrete_flow_matching:
                raise ValueError("Latent mode currently supports only continuous flow matching.")

            if args.edm_schedule:
                time_grid = get_time_discretization(nfes=ode_opts["nfe"])
            else:
                time_grid = torch.tensor([0.0, 1.0], device=device)

            synthetic_latent = solver.sample(
                time_grid=time_grid,
                x_init=x_0,
                method=args.ode_method,
                return_intermediates=False,
                atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                rtol=ode_opts["rtol"] if "rtol" in ode_opts else 1e-5,
                step_size=ode_opts["step_size"] if "step_size" in ode_opts else None,
                condition=latent_lr,
                cfg_scale=args.cfg_scale,
            )

            with torch.no_grad():
                synthetic_latent = denormalize_latent(synthetic_latent, latent_stats["mean"], latent_stats["std"])
                synthetic_samples = latent_ae.decode(synthetic_latent, h_lr)
                synthetic_samples = torch.clamp(synthetic_samples, 0.0, 1.0)

            synthetic_samples = synthetic_samples.to(torch.float32)
            fid_metric.update(synthetic_samples, real=False)

            if args.output_dir:
                for j in range(synthetic_samples.shape[0]):
                    base_name = os.path.splitext(filenames[j])[0]
                    save_path = Path(args.output_dir) / "snapshots" / f"epoch_{epoch}_{base_name}.png"
                    save_image(
                        synthetic_samples[j],
                        fp=save_path,
                        value_range=(0, 1),
                        normalize=False
                    )
                    
                    # Uncomment these to get GT and LR images saved
                    debug_folder = Path(args.output_dir) / "debug_triplets"
                    #save_image(hr[j] * 0.5 + 0.5, debug_folder / f"epoch_{epoch}_{base_name}_GT_HR.png", value_range=(0, 1), normalize=False)
                    #lr_up = F.interpolate(lr[j:j+1], size=hr.shape[2:], mode='bicubic')
                    #save_image(lr_up[0] * 0.5 + 0.5, debug_folder / f"epoch_{epoch}_{base_name}_INPUT_LR.png", value_range=(0, 1), normalize=False)
                    
                    save_image(synthetic_samples[j], debug_folder / f"epoch_{epoch}_{base_name}_MODEL_OUT.png", value_range=(0, 1), normalize=False)
                    
                    logger.info(f"Snapshot saved: {save_path}")

            num_synthetic += synthetic_samples.shape[0]

            if args.save_fid_samples and args.output_dir:
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.output_dir) / "fid_samples"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if not args.compute_fid:
            if num_synthetic >= fid_samples: return {}
            continue

        if data_iter_step % PRINT_FREQUENCY == 0:
            gc.collect()
            running_fid = fid_metric.compute()
            logger.info(
                f"Evaluating [{data_iter_step}/{len(data_loader)}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}"
            )

        if args.test_run:
            break

    return {"fid": float(fid_metric.compute().detach().cpu())}