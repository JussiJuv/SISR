import argparse
import logging
import math
import os
import random
import sys
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
# from IPython import embed

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

import shutil

# torch.autograd.set_detect_anomaly(True)

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            #os.system("rm ./log")
            """ if os.path.exists('./log'):
                if os.path.islink('./log'):
                    os.unlink('./log')
                else:
                    shutil.rmtree('./log')
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log") """


        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            #tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
            tb_logger = SummaryWriter(log_dir=opt["path"]["tb_logger"])
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")


    #### create train and val dataloader
    dataset_ratio = 1
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
    
            if opt["dist"]:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
            else:
                train_sampler = None
    
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
    
            # number of optimizer steps per epoch on this rank
            train_size = len(train_loader)
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
    
            if rank <= 0:
                #global_batch_size = dataset_opt["batch_size"] * (world_size if opt["dist"] else 1)
                global_batch_size = dataset_opt["batch_size"]
                logger.info(
                    "Number of train images: {:,d}, global batch size: {:,d}, iters per epoch: {:,d}".format(
                        len(train_set), global_batch_size, train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt) 
    device = model.device

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']
    # limit how many validation batches to run per validation
    val_max_images = int(opt['train'].get('val_max_images', 10))
    val_save_dir = opt["path"].get(
        "val_images",
        os.path.join(opt["path"]["experiments_root"], "val_images")
    )
    os.makedirs(val_save_dir, exist_ok=True)
    train_json_log = Path(opt["path"]["experiments_root"]) / "log.txt"
    if rank <= 0:
        train_json_log.write_text("", encoding="utf-8")

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    # Detect if we are in latent mode
    space_mode = opt.get('space_mode', 'pixel')

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)

    for epoch in range(start_epoch, start_epoch + total_epochs):
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            LQ, GT = train_data["LQ"], train_data["GT"]
            LQ = util.upscale(LQ, scale)

            # Latent vs pixle
            if space_mode == 'latent':
                with torch.no_grad():
                    z_LQ, _ = model.encode(LQ.to(device))
                    z_GT, _ = model.encode(GT.to(device))

                timesteps, states = sde.generate_random_states(x0=z_GT, mu=z_LQ)

                #model.feed_data(states, LQ.to(device), GT.to(device)) # xt, mu, x0
                model.feed_data(states, z_LQ, z_GT, latent_inputs=True, raw_LQ=LQ, raw_GT=GT)
            else:
                # Pixel
                timesteps, states = sde.generate_random_states(x0=GT.to(device), mu=LQ.to(device))
                model.feed_data(states, LQ, GT)

            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            logs = model.get_current_log()
            epoch_loss_sum += float(logs.get("loss", 0.0))
            epoch_loss_count += 1

            if current_step % opt["logger"]["print_freq"] == 0:
                epoch_display = epoch + 1
                step_in_epoch = (current_step - 1) % train_size + 1
                
                message = "<epoch:{:3d}/{:3d}, iter:{:8,d} ({:3d}/{:3d}), lr:{:.3e}> ".format(
                    epoch_display,
                    total_epochs,
                    current_step,
                    step_in_epoch,
                    train_size,
                    model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                avg_psnr = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):
                    if idx >= val_max_images:
                        break

                    LQ, GT = val_data["LQ"], val_data["GT"]
                    LQ = util.upscale(LQ, scale)

                    if space_mode == "latent":
                        with torch.no_grad():
                            LQ_upscaled = LQ.to(device)
                            z_LQ, hidden = model.encode(LQ_upscaled)
                            z_GT, _ = model.encode(GT.to(device))
                            noisy_state = sde.noise_state(z_LQ)

                        model.feed_data(
                            noisy_state,
                            z_LQ,
                            z_GT,
                            latent_inputs=True,
                            raw_LQ=LQ_upscaled,
                            raw_GT=GT.to(device),
                            hidden=hidden,
                        )
                        model.test(sde, hidden)
                    else:
                        noisy_state = sde.noise_state(LQ.to(device))
                        # valid Predictor
                        model.feed_data(noisy_state, LQ, GT)
                        model.test(sde)
                        
                    visuals = model.get_current_visuals()

                    iter_save_dir = os.path.join(val_save_dir, f"iter_{current_step:06d}")
                    os.makedirs(iter_save_dir, exist_ok=True)
                    
                    output = util.tensor2img(visuals["Output"].squeeze())
                    gt_img = util.tensor2img(visuals["GT"].squeeze())
                    lq_img = util.tensor2img(LQ.squeeze())
                    
                    cv2.imwrite(os.path.join(iter_save_dir, f"{idx:03d}_LQ.png"), lq_img)
                    cv2.imwrite(os.path.join(iter_save_dir, f"{idx:03d}_Output.png"), output)
                    cv2.imwrite(os.path.join(iter_save_dir, f"{idx:03d}_GT.png"), gt_img)
                    
                    comparison = np.concatenate((lq_img, output, gt_img), axis=1)
                    cv2.imwrite(os.path.join(iter_save_dir, f"{idx:03d}_comparison.png"), comparison)

                    # calculate PSNR
                    avg_psnr += util.calculate_psnr(output, gt_img)
                    idx += 1

                avg_psnr = avg_psnr / idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

                # log
                logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    )
                )
                print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    ))
                # tensorboard logger
                """ if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step) """
                if 'tb_logger' in locals() and tb_logger is not None and rank <= 0:
                    try:
                        tb_logger.add_scalar("psnr", avg_psnr, current_step)

                        if all(k in visuals for k in ('LQ', 'Output', 'GT')):
                            img_lq = visuals['LQ'][0].detach().cpu()
                            img_out = visuals['Output'][0].detach().cpu()
                            img_gt = visuals['GT'][0].detach().cpu()

                            def _to_01(t):
                                t = t.float()
                                tmin = float(t.min())
                                tmax = float(t.max())
                                if (tmax - tmin) < 1e-8:
                                    return t - tmin
                                return (t - tmin) / (tmax - tmin + 1e-8)
                            
                            grid = torchvision.utils.make_grid(
                                [_to_01(img_lq), _to_01(img_out), _to_01(img_gt)],
                                nrow=3, normalize=False
                            )

                            tb_logger.add_image('Validation/Comparison_LQ_Output_GT', grid, current_step)
                            

                    except Exception as e:
                        logger.warning(f'Failed to write validation images to TensorBoard/disk: {e}')
                else:
                    if opt.get('use_tb_logger', False) and 'debug' not in opt['name']:
                        logger.warning('use_tb_logger=True but tb_logger is not available.')


            if error.value:
                sys.exit(0)
            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

        if rank <= 0 and epoch_loss_count > 0:
            epoch_loss = epoch_loss_sum / epoch_loss_count
            with open(train_json_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "train_loss": epoch_loss,
                    "epoch": epoch,
                    "lr": float(model.get_current_learning_rate()),
                }) + "\n")

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    if 'tb_logger' in locals() and tb_logger is not None:
        tb_logger.close()


if __name__ == "__main__":
    main()