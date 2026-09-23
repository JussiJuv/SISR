from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.datasets import PairedImageDataset
from training.latent_ae import build_latent_autoencoder


@torch.no_grad()
def compute_stats(args):
    device = torch.device(args.device)

    latent_ae = build_latent_autoencoder(args, device)
    latent_ae.eval()

    dataset = PairedImageDataset(
        hr_path=args.data_path,
        lr_path=args.lr_data_path,
        transform=None,
        image_size=128,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sum_ = None
    sumsq = None
    count = 0

    for hr, lr, _ in loader:
        """ lr_pix = (lr.to(device) + 1.0) * 0.5
        latent_lr, _ = latent_ae.encode(lr_pix) """
        hr_pix = (hr.to(device) + 1.0) * 0.5
        latent_hr, _ = latent_ae.encode(hr_pix)
        b, c, h, w = latent_hr.shape
        x = latent_hr.permute(1, 0, 2, 3).contiguous().view(c, -1)

        #b, c, h, w = latent_lr.shape
        #x = latent_lr.permute(1, 0, 2, 3).contiguous().view(c, -1)

        if sum_ is None:
            sum_ = x.sum(dim=1)
            sumsq = (x ** 2).sum(dim=1)
        else:
            sum_ += x.sum(dim=1)
            sumsq += (x ** 2).sum(dim=1)

        count += x.shape[1]

    mean = sum_ / count
    var = (sumsq / count) - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-8))

    out = {
        "mean": mean.cpu(),
        "std": std.cpu(),
    }
    torch.save(out, args.out_path)
    print(f"Saved latent stats to {args.out_path}")
    print("mean:", mean)
    print("std :", std)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--lr_data_path", required=True)
    parser.add_argument("--latent_ae_ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--space_mode", default="latent")
    args = parser.parse_args()
    compute_stats(args)


if __name__ == "__main__":
    main()