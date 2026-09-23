import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torchvision.utils as tvutils

sys.path.insert(0, "../../")
import utils as util
import options as option
from models import create_model
from data import create_dataloader, create_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, required=True, help="Path to options YAML file.")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Main output directory")
    parser.add_argument("--save_subdir", type=str, default="validation_samples", help="Subfolder for images")
    parser.add_argument("--num_val_images", type=int, default=None, help="Limit number of images to process")
    
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    #limit = opt.get("num_val_images", len(test_set))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    logger = logging.getLogger("base")

    """ out_dir = Path(args.output_dir)
    save_dir = out_dir / args.save_subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "validate_log.txt" """
    path_cfg = opt.get("path", {})
    out_dir = Path(path_cfg.get("results_root", args.output_dir))
    save_dir = Path(path_cfg.get("validation_dir", out_dir / args.save_subdir))
    save_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "validate_log.txt"

    model = create_model(opt)
    device = model.device
    
    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"], 
        T=opt["sde"]["T"], 
        schedule=opt["sde"]["schedule"], 
        eps=opt["sde"]["eps"], 
        device=device
    )
    sde.set_model(model.model)
    scale = opt['degradation']['scale']

    phase, dataset_opt = next(iter(opt["datasets"].items()))
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    
    #limit = args.num_val_images if args.num_val_images is not None else len(test_set)
    limit = int(opt.get("num_val_images", len(test_set)))
    logger.info(f"Validating {limit} image(s) | Mode: {opt.get('space_mode', 'pixel')}")

    run_start = time.perf_counter()
    image_records = []
    model.model.eval()

    for i, test_data in enumerate(test_loader):
        if i >= limit:
            break

        img_path = test_data["GT_path"][0] if test_data["GT_path"] else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        LQ, GT = test_data["LQ"], test_data["GT"]
        
        sample_start = time.perf_counter()

        if opt.get("space_mode") == "latent":
            with torch.no_grad():
                LQ_upscaled = util.upscale(LQ, scale).to(device)
                z_LQ, hidden_feat = model.encode(LQ_upscaled)
                z_GT, _ = model.encode(GT.to(device))
                noisy_state = sde.noise_state(z_LQ)

            model.feed_data(
                noisy_state,
                z_LQ,
                z_GT,
                latent_inputs=True,
                raw_LQ=LQ_upscaled,
                raw_GT=GT.to(device),
                hidden=hidden_feat,
            )
            model.test(sde, hidden_feat, save_states=False)
        else:
            with torch.no_grad():
                LQ_upscaled = util.upscale(LQ, scale).to(device)
                GT = GT.to(device)
                timesteps, noisy_state = sde.generate_random_states(x0=GT, mu=LQ_upscaled)

            model.feed_data(noisy_state, LQ_upscaled, GT)
            model.test(sde, save_states=False)
        
            
        sample_seconds = time.perf_counter() - sample_start
        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        
        SR_img = torch.clamp(SR_img, 0.0, 1.0)

        save_path = save_dir / f"{img_name}_pred.png"
        tvutils.save_image(SR_img, save_path, normalize=False)
        
        logger.info(f"[{i+1}/{limit}] Generated {img_name} in {sample_seconds:.3f}s")
        image_records.append({"name": img_name, "seconds": sample_seconds})

    total_seconds = time.perf_counter() - run_start

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"total_time_seconds: {total_seconds:.6f}\n")
        f.write(f"num_images: {len(image_records)}\n")
        f.write(f"space_mode: {opt.get('space_mode', 'pixel')}\n")
        f.write("\nper_image_timings:\n")
        for rec in image_records:
            f.write(f"{rec['name']}: time_seconds={rec['seconds']:.6f}\n")

    logger.info(f"Validation completed. Results in {save_dir}")

if __name__ == "__main__":
    main()