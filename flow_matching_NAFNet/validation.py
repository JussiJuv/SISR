from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import time

REPO_ROOT = Path(__file__).resolve().parent
EXAMPLE_DIR = REPO_ROOT / "examples" / "image"

for p in (REPO_ROOT, EXAMPLE_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import PIL.Image
import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from torchvision.utils import save_image

from models.model_configs import instantiate_model
from training.datasets import PairedImageDataset
from training.eval_loop import CFGScaledModel
from training.edm_time_discretization import get_time_discretization
from training.train_loop import MASK_TOKEN
from training.latent_ae import build_latent_autoencoder
from training.latent_normalization import normalize_latent, denormalize_latent

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Standalone validation", add_help=True)

    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint.pth")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Where to save validation outputs")
    parser.add_argument("--data_path", type=str, default="", help="Path to DIV2K HR validation folder")
    parser.add_argument("--lr_data_path", type=str, default="", help="Path to DIV2K LR validation folder")
    parser.add_argument("--dataset", type=str, default="div2k", choices=["div2k", "imagenet", "cifar10"], help="Dataset type")
    parser.add_argument("--model", type=str, default="nafnet", help="Model architecture key")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run validation on")
    parser.add_argument("--num_val_images", type=int, default=1, help="Number of validation images to process")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Optional overrides for sampling; defaults are taken from checkpoint args when possible.
    parser.add_argument("--cfg_scale", type=float, default=None, help="Classifier-free guidance scale")
    parser.add_argument("--ode_method", type=str, default=None, help="ODE solver method")
    parser.add_argument("--ode_options", type=json.loads, default=None, help='ODE solver options as JSON, e.g. "{\"step_size\": 0.01}"')
    parser.add_argument("--sym", type=float, default=None, help="Symmetric term for discrete sampling")
    parser.add_argument("--temp", type=float, default=None, help="Temperature for discrete sampling")
    parser.add_argument("--sym_func", action="store_true", help="Use the fixed symmetric function for discrete sampling")
    parser.add_argument("--sampling_dtype", type=str, choices=["float32", "float64"], default=None, help="Discrete solver dtype")
    parser.add_argument("--discrete_flow_matching", action="store_true", help="Force discrete flow matching mode")
    parser.add_argument("--discrete_fm_steps", type=int, default=None, help="Number of discrete FM steps")
    parser.add_argument("--use_ema", action="store_true", help="Force EMA mode")
    parser.add_argument("--image_size", type=int, default=None, help="Optional validation resize")
    parser.add_argument("--save_subdir", type=str, default="validation_samples", help="Subfolder inside output_dir for PNGs")
    parser.add_argument("--warm_start_lr", action="store_true", help="Use LR as the starting state x0 instead of Gaussian noise.")
    parser.add_argument("--space_mode", choices=["pixel", "latent"], default="pixel", help="Run validation in pixel space or latent space.")
    parser.add_argument("--latent_ae_ckpt", type=str, default="", help="Path to the pretrained latent autoencoder checkpoint. Required when --space_mode latent.")
    parser.add_argument("--latent_stats_path", type=str, default="", help="Path to latent mean/std file (.pt)")

    return parser


def resolve_checkpoint_path(args: argparse.Namespace) -> Optional[Path]:
    candidates = []
    if args.checkpoint:
        candidates.append(Path(args.checkpoint))
    if args.output_dir:
        candidates.append(Path(args.output_dir) / "checkpoint.pth")

    for path in candidates:
        if path.is_file():
            return path
    return None


def load_checkpoint(path: Path) -> Dict[str, Any]:
    logger.info("Loading checkpoint: %s", path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} does not look valid (missing 'model' key).")
    return checkpoint


def merge_args(cli_args: argparse.Namespace, ckpt_args: Optional[argparse.Namespace]) -> argparse.Namespace:
    merged = argparse.Namespace()

    if ckpt_args is not None:
        for k, v in vars(ckpt_args).items():
            setattr(merged, k, v)

    cli_dict = vars(cli_args)
    for key, value in cli_dict.items():
        if key in {"sym_func", "discrete_flow_matching", "use_ema", "warm_start_lr"}:
            if value:
                setattr(merged, key, value)
            continue
        if value is not None and value != "":
            setattr(merged, key, value)

    defaults = {
        "dataset": "div2k",
        "model": "nafnet",
        "cfg_scale": 0.2,
        "ode_method": "euler",
        "ode_options": {"step_size": 1.0},
        "sym": 0.0,
        "temp": 1.0,
        "sampling_dtype": "float32",
        "discrete_fm_steps": 1024,
        "discrete_flow_matching": False,
        "use_ema": False,
        "image_size": None,
        "num_workers": 4,
        "pin_mem": True,
        "save_subdir": "validation_samples",
        "space_mode": "pixel",
        "latent_ae_ckpt": "",
    }
    for key, value in defaults.items():
        if not hasattr(merged, key):
            setattr(merged, key, value)

    merged.image_size = None

    return merged


def normalize_div2k_paths(args: argparse.Namespace) -> None:
    if args.dataset != "div2k":
        return

    if not getattr(args, "data_path", None):
        raise ValueError("For DIV2K validation you must provide --data_path")
    if not getattr(args, "lr_data_path", None):
        raise ValueError("For DIV2K validation you must provide --lr_data_path")

    if "train" in args.data_path.lower():
        args.data_path = args.data_path.replace("train", "valid")
    if "train" in args.lr_data_path.lower():
        args.lr_data_path = args.lr_data_path.replace("train", "valid")


def build_dataset(args: argparse.Namespace):
    if args.dataset != "div2k":
        raise NotImplementedError("This standalone validation script currently targets the DIV2K paired setup.")

    return PairedImageDataset(
        hr_path=args.data_path,
        lr_path=args.lr_data_path,
        transform=None,
        image_size=args.image_size,
    )


@torch.no_grad()
def run_validation(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    latent_ae = build_latent_autoencoder(args, device) if args.space_mode == "latent" else None

    latent_stats = None
    if args.space_mode == "latent":
        if not args.latent_stats_path:
            raise ValueError("--latent_stats_path is required for latent mode")

        latent_stats = torch.load(args.latent_stats_path, map_location=device)

        # Ensure proper shape for broadcasting: (C,) → (1, C, 1, 1)
        latent_stats["mean"] = latent_stats["mean"].view(1, -1, 1, 1).to(device)
        latent_stats["std"] = latent_stats["std"].view(1, -1, 1, 1).to(device)

    checkpoint_path = resolve_checkpoint_path(args)
    if checkpoint_path is None:
        logger.error("No checkpoint found. Checked --checkpoint and %s/checkpoint.pth", args.output_dir)
        sys.exit(1)

    checkpoint = load_checkpoint(checkpoint_path)
    ckpt_args = checkpoint.get("args", None)
    merged_args = merge_args(args, ckpt_args)
    normalize_div2k_paths(merged_args)
    
    if merged_args.space_mode == "latent" and not merged_args.latent_ae_ckpt:
        raise ValueError("--latent_ae_ckpt is required when --space_mode latent")

    out_dir = Path(merged_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / merged_args.save_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()
    image_records = []
    log_path = out_dir / "validate_log.txt"

    logger.info("Using dataset=%s, model=%s", merged_args.dataset, merged_args.model)
    logger.info("Validation data: HR=%s | LR=%s", merged_args.data_path, merged_args.lr_data_path)
    logger.info("Saving outputs to: %s", save_dir)

    model = instantiate_model(
        architechture=merged_args.model,
        is_discrete=bool(merged_args.discrete_flow_matching),
        use_ema=bool(merged_args.use_ema),
        space_mode=merged_args.space_mode,
    )
    model.to(device)

    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    dataset_val = build_dataset(merged_args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=merged_args.num_workers,
        pin_memory=merged_args.pin_mem,
        drop_last=False,
    )

    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.eval()

    if merged_args.discrete_flow_matching:
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
        ode_opts = merged_args.ode_options if merged_args.ode_options is not None else {"step_size": 1.0}

    saved = 0
    for data_iter_step, (hr, lr, filenames) in enumerate(data_loader_val):
        if saved >= merged_args.num_val_images:
            break

        hr = hr.to(device, non_blocking=True)
        lr = lr.to(device, non_blocking=True)
        cfg_scaled_model.reset_nfe_counter()

        sample_start = time.perf_counter()

        if merged_args.space_mode == "latent":
            if merged_args.discrete_flow_matching:
                raise ValueError("Latent validation currently supports only continuous flow matching.")

            if latent_ae is None:
                raise ValueError("Latent mode requires a loaded latent autoencoder.")
            
            # dataset returns [-1, 1], latent AE expects [0, 1]
            hr_pix = (hr + 1.0) * 0.5
            lr_pix = (lr + 1.0) * 0.5

            with torch.no_grad():
                latent_lr, h_lr = latent_ae.encode(lr_pix)
                latent_lr = normalize_latent(latent_lr, latent_stats["mean"], latent_stats["std"])

            condition = latent_lr

            if merged_args.warm_start_lr:
                x_0 = latent_lr
            else:
                x_0 = torch.randn_like(latent_lr)

            if getattr(merged_args, "edm_schedule", False):
                time_grid = get_time_discretization(nfes=ode_opts["nfe"])
            else:
                time_grid = torch.tensor([0.0, 1.0], device=device)

            synthetic_latent = solver.sample(
                time_grid=time_grid,
                x_init=x_0,
                method=merged_args.ode_method,
                return_intermediates=False,
                atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                rtol=ode_opts["rtol"] if "rtol" in ode_opts else 1e-5,
                step_size=ode_opts["step_size"] if "step_size" in ode_opts else None,
                condition=condition,
                cfg_scale=merged_args.cfg_scale,
            )

            with torch.no_grad():
                synthetic_latent = denormalize_latent(synthetic_latent, latent_stats["mean"], latent_stats["std"])
                synthetic_samples = latent_ae.decode(synthetic_latent, h_lr)
                synthetic_samples = torch.clamp(synthetic_samples, 0.0, 1.0)

        else:
            condition = lr

            if merged_args.discrete_flow_matching:
                x_0 = torch.zeros(hr.shape, dtype=torch.long, device=device) + MASK_TOKEN
                sym = (
                    lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
                ) if merged_args.sym_func else merged_args.sym
                dtype = torch.float64 if merged_args.sampling_dtype == "float64" else torch.float32

                synthetic_samples = solver.sample(
                    x_init=x_0,
                    step_size=1.0 / merged_args.discrete_fm_steps,
                    verbose=False,
                    div_free=sym,
                    dtype_categorical=dtype,
                    condition=condition,
                    cfg_scale=merged_args.cfg_scale,
                )
            else:
                if merged_args.warm_start_lr:
                    x_0 = condition
                else:
                    x_0 = torch.randn(hr.shape, dtype=torch.float32, device=device)

                if getattr(merged_args, "edm_schedule", False):
                    time_grid = get_time_discretization(nfes=ode_opts["nfe"])
                else:
                    time_grid = torch.tensor([0.0, 1.0], device=device)

                synthetic_samples = solver.sample(
                    time_grid=time_grid,
                    x_init=x_0,
                    method=merged_args.ode_method,
                    return_intermediates=False,
                    atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                    rtol=ode_opts["rtol"] if "rtol" in ode_opts else 1e-5,
                    step_size=ode_opts["step_size"] if "step_size" in ode_opts else None,
                    condition=condition,
                    cfg_scale=merged_args.cfg_scale,
                )
                synthetic_samples = torch.clamp(synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0)

        sample_seconds = time.perf_counter() - sample_start
        nfe_total = cfg_scaled_model.get_nfe()
        nfe_per_image = nfe_total / synthetic_samples.shape[0]

        image_records.append({
            "name": os.path.splitext(os.path.basename(filenames[0]))[0],
            "seconds": sample_seconds,
            "nfe_total": nfe_total,
            "nfe_per_image": nfe_per_image,
        })

        logger.info(
            "Generated image %s in %.3f s | NFE total = %d | NFE/image = %.1f",
            image_records[-1]["name"],
            sample_seconds,
            nfe_total,
            nfe_per_image,
        )

        batch_size = synthetic_samples.shape[0]
        remaining = merged_args.num_val_images - saved
        if batch_size > remaining:
            synthetic_samples = synthetic_samples[:remaining]
            batch_size = synthetic_samples.shape[0]

        logger.info(
            "Generated %d image(s) in batch %d using %d function evaluations.",
            batch_size,
            data_iter_step,
            cfg_scaled_model.get_nfe(),
        )

        for j in range(batch_size):
            base_name = os.path.splitext(os.path.basename(filenames[j]))[0]
            save_path = save_dir / f"{base_name}_pred.png"
            save_image(
                synthetic_samples[j],
                fp=save_path,
                value_range=(0, 1),
                normalize=False,
            )
            logger.info("Saved %s", save_path)
            saved += 1

        if saved >= merged_args.num_val_images:
            break

    total_seconds = time.perf_counter() - run_start

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"total_time_seconds: {total_seconds:.6f}\n")
        f.write(f"num_images: {saved}\n")
        f.write(f"ode_method: {merged_args.ode_method}\n")
        if image_records:
            f.write(f"function_evaluations_per_image: {image_records[0]['nfe_per_image']:.1f}\n")
        f.write("\nper_image_timings:\n")
        for rec in image_records:
            f.write(
                f"{rec['name']}: time_seconds={rec['seconds']:.6f}\n"
            )

    logger.info("Validation completed. Saved %d image(s).", saved)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_validation(args)


if __name__ == "__main__":
    main()