from copy import deepcopy

def _nafnet_cfg(img_channel, width, enc_blk_nums, middle_blk_num, dec_blk_nums):
    return {
        "which_model_G": "ConditionalNAFNet",
        "setting": {
            "width": width,
            "img_channel": img_channel,
            "enc_blk_nums": enc_blk_nums,
            "middle_blk_num": middle_blk_num,
            "dec_blk_nums": dec_blk_nums,
        },
    }

MODEL_CONFIGS = {
    "pixel_default": _nafnet_cfg(
        img_channel=3,
        width=64,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    ),
    "latent_default": _nafnet_cfg(
        img_channel=4,
        width=64,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    ),
    "latent_big": _nafnet_cfg(
        img_channel=4,
        width=64,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2],
    ),
    "pixel_big": _nafnet_cfg(
        img_channel=3,
        width=64,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2],
    ),
}

def get_g_config(preset: str):
    return deepcopy(MODEL_CONFIGS[preset])