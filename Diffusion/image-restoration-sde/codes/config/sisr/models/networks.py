import logging

import torch

from models import modules as M
from .latent_bokeh.modules.UNet_arch import UNet
from .model_configs import get_g_config

logger = logging.getLogger("base")

from models.model_configs import get_g_config

def define_G(opt):
    opt_net = opt["network_G"]
    if "preset" in opt_net:
        cfg = get_g_config(opt_net["preset"])
        which_model = cfg["which_model_G"]
        setting = cfg["setting"]
    else:
        which_model = opt_net["which_model_G"]
        setting = opt_net["setting"]

    netG = getattr(M, which_model)(**setting)
    return netG


# Discriminator
def define_D(opt):
    opt_net = opt["network_D"]
    setting = opt_net["setting"]
    netD = getattr(M, which_model)(**setting)
    return netD


# Perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt["gpu_ids"]
    device = torch.device("cuda" if gpu_ids else "cpu")
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = M.VGGFeatureExtractor(
        feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device
    )
    netF.eval()  # No need to train
    return netF

# Define Latent AutoEncoder (L)
def define_L(opt):
    opt_net = opt['network_L']
    which_model = opt_net['which_model']
    setting = opt_net['setting'] # Helper to shorten lines

    if which_model == 'UNet':
        # Wrap the numeric values in int() to prevent the string formatting error
        netL = UNet(
            in_ch=int(setting['in_ch']),
            out_ch=int(setting['out_ch']),
            ch=int(setting['ch']),
            ch_mult=[int(x) for x in setting['ch_mult']], # Cast every item in the list
            embed_dim=int(setting['embed_dim'])
        )
    else:
        raise NotImplementedError(f'Latent model [{which_model}] not recognized.')

    return netL