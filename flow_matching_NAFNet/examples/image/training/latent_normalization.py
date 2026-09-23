import torch


def normalize_latent(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    z:    [B, C, H, W]
    mean: [C] or [1, C, 1, 1]
    std:  [C] or [1, C, 1, 1]
    """
    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1:
        std = std.view(1, -1, 1, 1)
    return (z - mean) / (std + 1e-8)


def denormalize_latent(z_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Inverse of normalize_latent.
    """
    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1:
        std = std.view(1, -1, 1, 1)
    return z_norm * (std + 1e-8) + mean