from __future__ import annotations

import torch

from models.latent_bokeh import networks


def build_latent_autoencoder(args, device: torch.device):
    if args.space_mode != "latent":
        return None

    if not args.latent_ae_ckpt:
        raise ValueError("--latent_ae_ckpt is required when --space_mode latent")

    ckpt = torch.load(args.latent_ae_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # This matches the working test setup you already verified.
    opt = {
        "network_L": {
            "which_model": "UNet",
            "setting": {
                "in_ch": 3,
                "out_ch": 3,
                "ch": 64,
                "ch_mult": [1, 2, 4],
                "embed_dim": 4,
            },
        },
        "gpu_ids": [0],
    }

    model = networks.define_L(opt)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model