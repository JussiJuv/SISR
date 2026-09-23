# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser
from torchvision import transforms
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from training import distributed_mode
from training.data_transform import get_train_transform
from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch
from training.latent_ae import build_latent_autoencoder

from models.latent_bokeh.optimizer import Lion

logger = logging.getLogger(__name__)


def main(args):
    distributed_mode.init_distributed_mode(args)

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.output_dir and distributed_mode.is_main_process():
        file_handler = logging.FileHandler(
            os.path.join(args.output_dir, "train_log.txt"),
            mode="w",
            encoding="utf-8",
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)

    device = torch.device(args.device)
    latent_ae = build_latent_autoencoder(args, device)

    latent_stats = None
    if args.space_mode == "latent":
        if not hasattr(args, "latent_stats_path") or not args.latent_stats_path:
            raise ValueError("--latent_stats_path is required when --space_mode latent")
        latent_stats = torch.load(args.latent_stats_path, map_location="cpu")
        latent_stats["mean"] = latent_stats["mean"].to(device)
        latent_stats["std"] = latent_stats["std"].to(device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_train_transform()

    if args.dataset == "div2k":
        from training.datasets import PairedImageDataset
        dataset_train = PairedImageDataset(
            hr_path=args.data_path,
            lr_path=args.lr_data_path,
            transform=transform_train,
            image_size=args.image_size
        )
        val_hr_path = args.data_path.replace("train", "valid")
        val_lr_path = args.lr_data_path.replace("train", "valid")
        
        dataset_val = PairedImageDataset(
            hr_path=val_hr_path,
            lr_path=val_lr_path,
            transform=None,
            image_size=None,
            #transform=transform_train,
            #image_size=args.image_size
        )

    elif args.dataset == "imagenet":
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    logger.info(dataset_train)

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        #batch_size=args.batch_size,
        batch_size=1, # TEMPORARY FORCED TO 1
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    logger.info("Initializing Model")
    model = instantiate_model(
        #architechture=args.dataset,
        architechture=args.model,
        is_discrete=args.discrete_flow_matching,
        use_ema=args.use_ema,
        space_mode=args.space_mode,
    )

    model.to(device)

    model_without_ddp = model
    #logger.info(str(model_without_ddp))

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    if args.optimizer == "lion":
        optimizer = Lion(
            model_without_ddp.parameters(),
            lr=args.lr,
            betas=tuple(args.optimizer_betas),
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model_without_ddp.parameters(),
            lr=args.lr,
            betas=tuple(args.optimizer_betas),
            weight_decay=args.weight_decay
        )

    warmup_epochs = args.warmup_epochs

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - warmup_epochs,
        eta_min=args.min_lr,
    )

    lr_schedule = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    # --- SMOKE TEST: VERIFY SAVING WORKS ---
    if args.test_run:
        logger.info("Running Smoke Test: Verifying evaluation and saving logic...")
        test_stats = eval_model(
            model,
            data_loader_val,
            device,
            epoch=999,
            fid_samples=1,
            args=args,
            latent_ae=latent_ae
        )
        logger.info("Smoke Test Successful! Validation image saved.")
    # ---------------------------------------
    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
                data_loader_val=data_loader_val,
                eval_model=eval_model,
                latent_ae=latent_ae,
                latent_stats=latent_stats,
            )
            lr_schedule.step()
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "lr": current_lr,
            }
        else:
            log_stats = {
                "epoch": epoch,
            }

        save_freq = getattr(args, "save_frequency", 1) 
        if args.output_dir and not args.eval_only:
            if (epoch + 1) % save_freq == 0 or (epoch + 1) == args.epochs:
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
                logger.info(f"Checkpoint saved at epoch {epoch}")

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)
            
            # Logic for setting fid_samples based on rank...
            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (args.fid_samples // num_tasks)
            else:
                fid_samples = args.fid_samples // num_tasks
            
            eval_stats = eval_model(
                model,
                data_loader_val,
                device,
                epoch=epoch,
                fid_samples=args.num_val_images,
                args=args,
                latent_ae=latent_ae
            )
            log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
