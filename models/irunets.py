import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from models.transformer import get_transformer_encoder, get_transformer_decoder
from models.submodels import DownsamplingLayer

from utils.functions import simple_upscale
import utils.config as config



# U-shaped Transformer for image restoration
class IRUNet(nn.Module):
    """
    Args:
        img_res : size of (output) image
        d_embed : embedding dimension
        n_layer : number of layers; [block1, block2, block3, block4]
        hidden_dim_rate : rate of dimension of hidden layer in FFNN
        dropout : dropout ratio
        path_dropout : dropout ratio for path drop
        version : model version
    """
    def __init__(self, img_res=192, d_embed=48, n_layer=[4,6,6,8], hidden_dim_rate=4,
                 dropout=0, path_dropout=0.1, sr_upscale=1, version=1):
        super(IRUNet, self).__init__()
        self.img_res = img_res
        self.d_embed = d_embed
        self.version = version
        self.sr_upscale = sr_upscale
        
        # resolution synchronizing module
        if sr_upscale == 1:
            self.resolution_synchronizing = nn.Identity()
        else:
            self.resolution_synchronizing = self.upscale_img
        
        # initial feature mapping layer
        self.initial_feature_mapping = nn.Conv2d(3, d_embed, kernel_size=3, stride=1, padding=1)
        trunc_normal_(self.initial_feature_mapping.weight, std=.02)
        nn.init.zeros_(self.initial_feature_mapping.bias)
        
        # reconstruction layer
        self.reconstruction_layer = nn.Conv2d(d_embed*2, 3, kernel_size=3, stride=1, padding=1)
        trunc_normal_(self.reconstruction_layer.weight, std=.02)
        nn.init.zeros_(self.reconstruction_layer.bias)
        
        # downsampling layers
        self.downsampling_layer_1 = DownsamplingLayer(2, d_embed, 2)
        self.downsampling_layer_2 = DownsamplingLayer(2, 2*d_embed, 2)
        self.downsampling_layer_3 = DownsamplingLayer(2, 4*d_embed, 2)
        
        # transformer encoders
        self.encoder_1 = get_transformer_encoder(d_embed=d_embed,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 convolutional_ff=True,
                                                 n_layer=n_layer[0],
                                                 n_head=2,
                                                 n_class=2,
                                                 d_ff=d_embed*hidden_dim_rate,
                                                 n_patch=img_res,
                                                 window_size=8,
                                                 dropout=dropout,
                                                 path_dropout=path_dropout)
        
        self.encoder_2 = get_transformer_encoder(d_embed=d_embed*2,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 convolutional_ff=True,
                                                 n_layer=n_layer[1],
                                                 n_head=4,
                                                 n_class=4,
                                                 d_ff=2*d_embed*hidden_dim_rate,
                                                 n_patch=img_res//2,
                                                 window_size=8,
                                                 dropout=dropout,
                                                 path_dropout=path_dropout)
        
        self.encoder_3 = get_transformer_encoder(d_embed=d_embed*4,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 convolutional_ff=True,
                                                 n_layer=n_layer[2],
                                                 n_head=8,
                                                 n_class=4,
                                                 d_ff=4*d_embed*hidden_dim_rate,
                                                 n_patch=img_res//4,
                                                 window_size=8,
                                                 dropout=dropout,
                                                 path_dropout=path_dropout)
        
        self.encoder_4 = get_transformer_encoder(d_embed=d_embed*8,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 convolutional_ff=True,
                                                 n_layer=n_layer[3],
                                                 n_head=16,
                                                 n_class=4,
                                                 d_ff=8*d_embed*hidden_dim_rate,
                                                 n_patch=img_res//8,
                                                 window_size=8,
                                                 dropout=dropout,
                                                 path_dropout=path_dropout)
        
        # transformer decoders
        self.decoder_3 = get_transformer_decoder(d_embed=d_embed*4,
                                                 key_d_embed=d_embed*8,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 convolutional_ff=True,
                                                 n_layer=n_layer[2],
                                                 n_head=8,
                                                 n_class=4,
                                                 d_ff=4*d_embed*hidden_dim_rate,
                                                 query_n_patch=img_res//4,
                                                 query_window_size=8,
                                                 key_n_patch=img_res//8,
                                                 key_window_size=4,
                                                 dropout=dropout,
                                                 path_dropout=path_dropout)
        
        self.decoder_2 = get_transformer_decoder(d_embed=d_embed*2,
                                                 key_d_embed=d_embed*4,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 convolutional_ff=True,
                                                 n_layer=n_layer[1],
                                                 n_head=4,
                                                 n_class=4,
                                                 d_ff=2*d_embed*hidden_dim_rate,
                                                 query_n_patch=img_res//2,
                                                 query_window_size=8,
                                                 key_n_patch=img_res//4,
                                                 key_window_size=4,
                                                 dropout=dropout,
                                                 path_dropout=path_dropout)
        
        self.decoder_1 = get_transformer_decoder(d_embed=d_embed,
                                                 key_d_embed=d_embed*2,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 convolutional_ff=True,
                                                 n_layer=n_layer[0],
                                                 n_head=2,
                                                 n_class=2,
                                                 d_ff=d_embed*hidden_dim_rate,
                                                 query_n_patch=img_res,
                                                 query_window_size=8,
                                                 key_n_patch=img_res//2,
                                                 key_window_size=4,
                                                 dropout=dropout,
                                                 path_dropout=path_dropout)
        
        
    def upscale_img(self, img):
        return simple_upscale(img, self.sr_upscale)
    
        
    def forward(self, lq_img, load_mask=True):
        """
        <input>
            lq_img : (n_batch, 3, lq_img_height, lq_img_width), low-quality image
            load_mask : Use self.mask in each encoder or decoder as mask input if True
            
        <output>
            hq_img : (n_batch, 3, hq_img_height, hq_img_width), high-quality image
        """
        # Synchronize resolution.
        with torch.no_grad():
            lq_img = self.resolution_synchronizing(lq_img)
        
        # Make an initial feature map.
        x_init = self.initial_feature_mapping(lq_img)  # (B, C, H, W)
        
        # Encode features.
        if load_mask:
            x_1 = self.encoder_1(x_init.permute(0, 2, 3, 1))  # (B, H, W, C)

            x_2 = self.downsampling_layer_1(x_1)  # (B, H/2, W/2, 2C)
            x_2 = self.encoder_2(x_2)

            x_3 = self.downsampling_layer_2(x_2)  # (B, H/4, W/4, 4C)
            x_3 = self.encoder_3(x_3)

            x_4 = self.downsampling_layer_3(x_3)  # (B, H/8, W/8, 8C)
            x_4 = self.encoder_4(x_4)
            
        else:
            x_1 = self.encoder_1(x_init.permute(0, 2, 3, 1), False)  # (B, H, W, C)

            x_2 = self.downsampling_layer_1(x_1)  # (B, H/2, W/2, 2C)
            x_2 = self.encoder_2(x_2, False)

            x_3 = self.downsampling_layer_2(x_2)  # (B, H/4, W/4, 4C)
            x_3 = self.encoder_3(x_3, False)

            x_4 = self.downsampling_layer_3(x_3)  # (B, H/8, W/8, 8C)
            x_4 = self.encoder_4(x_4, False)
        
        # Decode features.
        if load_mask:
            x_3 = self.decoder_3(x_3, x_4)  # (B, H/4, W/4, 4C)
            x_2 = self.decoder_2(x_2, x_3)  # (B, H/2, W/2, 2C)
            x_1 = self.decoder_1(x_1, x_2)  # (B, H, W, C)
            
        else:
            x_3 = self.decoder_3(x_3, x_4, False)  # (B, H/4, W/4, 4C)
            x_2 = self.decoder_2(x_2, x_3, False)  # (B, H/2, W/2, 2C)
            x_1 = self.decoder_1(x_1, x_2, False)  # (B, H, W, C)
            
        # Reconstruct image.
        x_init = torch.cat((x_init, x_1.permute(0, 3, 1, 2)), dim=1)  # (B, 2C, H, W)
        return self.reconstruction_layer(x_init)  # (B, 3, H, W)
    
    
    def evaluate(self, lq_img, *args):
        _, _, H, W = lq_img.shape
        IMG_SIZE_UNIT = 64
        
        # Set padding options.
        H_pad = ((H - 1) // IMG_SIZE_UNIT + 1) * IMG_SIZE_UNIT
        W_pad = ((W - 1) // IMG_SIZE_UNIT + 1) * IMG_SIZE_UNIT
        
        # Pad boundaries.
        pad_img = F.pad(lq_img, (0, W_pad-W, 0, H_pad-H), mode='reflect')
        
        # Restore image.
        hq_img = self.forward(pad_img, False)
        return hq_img[..., :H*self.sr_upscale, :W*self.sr_upscale]