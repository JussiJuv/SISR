# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from models.unet import UNetModel
from .nafnet_sr import ConditionalNAFNet

MODEL_CONFIGS = {
    "nafnet":{
        "img_channel": 3,
        "width": 64,
        "middle_blk_num": 1,
        "enc_blk_nums": [1, 1, 1, 28],
        "dec_blk_nums": [1, 1, 1, 1],
    },
    "latent_nafnet": {
        "img_channel": 4,
        "width": 64,
        "middle_blk_num": 1,
        "enc_blk_nums": [1, 1, 1, 28],
        "dec_blk_nums": [1, 1, 1, 1],
    },
    "imagenet": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 3,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 3, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "imagenet_discrete": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.2,
        "channel_mult": [2, 3, 4, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "cifar10": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2, 2],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "cifar10_discrete": {
        "in_channels": 3,
        "model_channels": 96,
        "out_channels": 3,
        "num_res_blocks": 5,
        "attention_resolutions": [2],
        "dropout": 0.4,
        "channel_mult": [3, 4, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "div2k": {
        "in_channels": 6,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 3,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 3, 4],
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
    },
}


def instantiate_model(
    architechture: str,
    is_discrete: bool,
    use_ema: bool,
    space_mode: str = "pixel",
) -> Union[UNetModel, DiscreteUNetModel, ConditionalNAFNet]:
    print(f"DEBUG: Requesting architecture: {architechture}")
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    config = MODEL_CONFIGS[architechture]

    if architechture == "nafnet":
        if space_mode == "latent":
            config = MODEL_CONFIGS["latent_nafnet"]
        else:
            config = MODEL_CONFIGS["nafnet"]
        model = ConditionalNAFNet(**config)
    elif is_discrete:
        if architechture + "_discrete" in MODEL_CONFIGS:
            config = MODEL_CONFIGS[architechture + "_discrete"]
        model = DiscreteUNetModel(
            vocab_size=257,
            **config,
        )
    else:
        model = UNetModel(**config)

    if use_ema:
        return EMA(model=model)
    else:
        return model
