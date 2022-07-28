import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch

import config


# Local configs
DENORM_WEIGHT = torch.Tensor(config.IMG_NORM_STD).view(1, 3, 1, 1)
DENORM_BIAS = torch.Tensor(config.IMG_NORM_MEAN).view(1, 3, 1, 1)

PIXEL_MAX_VALUE = config.PIXEL_MAX_VALUE

GRAY_COEF = torch.Tensor(config.GRAY_COEF).view(1, 3, 1, 1)
_GRAY_COEF = GRAY_COEF / PIXEL_MAX_VALUE
GRAY_BIAS = config.GRAY_BIAS


# Denormalize images.
def denormalize(img, device=torch.device('cpu')):
    """
    <input>
        img : (n_batch, 3, img_height, img_width)
        device : torch.device
    """
    return img * DENORM_WEIGHT.to(device) + DENORM_BIAS.to(device)


# Quantize images.
# DO NOT USE IT WHEN TRAINING.
def quantize(img):
    return (img.clamp(0, 1) * PIXEL_MAX_VALUE).round() / PIXEL_MAX_VALUE


# Extract Y channel values from RGB images.
def convert_y_channel(rgb_img, add_bias=False, device=torch.device('cpu')):
    """
    <input>
        rgb_img : (n_batch, 3 (BGR order), img_height, img_width)
    """
    # Add bias for Y channel conversion in pixel scale (for SSIM).
    if add_bias:
        return (rgb_img * GRAY_COEF.to(device)).sum(dim=1) + GRAY_BIAS
    # Y channel conversion of RGB image difference in [0,1] scale (for PSNR)
    else:
        return (rgb_img * _GRAY_COEF.to(device)).sum(dim=1)