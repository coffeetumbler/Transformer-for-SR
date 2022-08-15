import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import get_transformer_encoder, get_transformer_decoder
from models.submodels import EmbeddingLayer, ReconstructionBlock, UpsamplingLayer



# Whole SR Transformer model
class SRTransformer(nn.Module):
    """
    Args:
        lr_img_res : size of low-resolution image
        upscale : upscaling factor, 2 to 4
        intermediate_upscale : boolean for intermediate upscaling when upscale==4
        patch_size : size of image patch
        window_size : size of window
        d_embed : embedding dimension
        encoder_n_layer : number of layers in encoder
        decoder_n_layer : number of layers in decoder
        n_head : number of heads in self-attention module
        interpolated_decoder_input : True if decoder inputs are interpolated images
        raw_decoder_input : True is decoder inputs are from raw degraded images
        dropout : dropout ratio
    """
    def __init__(self, lr_img_res=48, upscale=2, intermediate_upscale=False,
                       patch_size=2, window_size=4, d_embed=128,
                       encoder_n_layer=12, decoder_n_layer=12, n_head=4,
                       interpolated_decoder_input=False, raw_decoder_input=True, dropout=0.1):
        super(SRTransformer, self).__init__()
        assert lr_img_res % (patch_size * window_size) == 0
        assert d_embed % n_head == 0
        
        self.lr_img_res = lr_img_res
        self.upscale = upscale
        self.hr_img_res = lr_img_res * upscale
        self.upscaled_patch_size = patch_size * upscale
        
        # numbers of patches along an axis
        self.encoder_n_patch = lr_img_res // patch_size
        decoder_n_patch = self.encoder_n_patch * upscale
        
        # encoder embedding layer and reconstruction block
        self.encoder_embedding_layer = EmbeddingLayer(patch_size, d_embed)
        self.reconstruction_block = ReconstructionBlock(patch_size, d_embed, d_embed*4, dropout)
        
        # encoder
        self.transformer_encoder = get_transformer_encoder(d_embed=d_embed,
                                                           positional_encoding=None,
                                                           relative_position_embedding=True,
                                                           n_layer=encoder_n_layer,
                                                           n_head=n_head,
                                                           d_ff=d_embed*4,
                                                           n_patch=self.encoder_n_patch,
                                                           window_size=window_size,
                                                           dropout=dropout)
        
        # decoder input query configs
        self.no_decoder_input = not(interpolated_decoder_input)
        if self.no_decoder_input:
            self.raw_decoder_input = raw_decoder_input
            if raw_decoder_input:
                self.upsampling = nn.Conv2d(3, 3*(upscale**2), kernel_size=1, stride=1)
            else:
                self.upsampling = nn.Linear(d_embed, 3*(upscale**2)*(patch_size**2))
            nn.init.xavier_uniform_(self.upsampling.weight)
            nn.init.zeros_(self.upsampling.bias)

        # decoder and decoder embedding layer
        if upscale > 1 and upscale < 4:
            self.decoder_embedding_layer = EmbeddingLayer(patch_size, d_embed)
            self.transformer_decoder = get_transformer_decoder(d_embed=d_embed,
                                                               positional_encoding=None,
                                                               relative_position_embedding=True,
                                                               n_layer=decoder_n_layer,
                                                               n_head=n_head,
                                                               d_ff=d_embed*4,
                                                               query_n_patch=decoder_n_patch,
                                                               query_window_size=window_size*upscale,
                                                               key_n_patch=self.encoder_n_patch,
                                                               key_window_size=window_size,
                                                               dropout=dropout)
            self.intermediate_upscale = False
            
        elif upscale == 4:
            self.intermediate_upscale = intermediate_upscale
            # two-stage decoder
            if intermediate_upscale:
                self.decoder_embedding_layer = EmbeddingLayer(patch_size, d_embed)
                self.transformer_decoder = TwoStageDecoder(patch_size=patch_size,
                                                           window_size=window_size,
                                                           d_embed=d_embed,
                                                           n_layer=decoder_n_layer,
                                                           n_head=n_head,
                                                           encoder_n_patch=self.encoder_n_patch,
                                                           dropout=dropout)
            # one-stage decoder
            else:
                self.decoder_embedding_layer = EmbeddingLayer(patch_size*2, d_embed)
                self.transformer_decoder = OneStageDecoder(window_size=window_size,
                                                           d_embed=d_embed,
                                                           n_layer=decoder_n_layer,
                                                           n_head=n_head,
                                                           encoder_n_patch=self.encoder_n_patch,
                                                           dropout=dropout)
        
    def forward(self, lr_img, lr_img_upscaled=None):
        """
        <input>
            lr_img : (n_batch, 3, img_height, img_width), low-res image
            lr_img_upscaled : (n_batch, 3, upscale*img_height, upscale*img_width), upscaled lr_img by bicubic interpolation
            
        <output>
            hr_img : (n_batch, 3, upscale*img_height, upscale*img_width), high-res image
        """
        # Encode low-resolution image
        encoder_out = self.encoder_embedding_layer(lr_img)
        encoder_out = self.transformer_encoder(encoder_out)
        
        # Make initial upscaled lr_img
        if self.no_decoder_input:
            # Initial lr_img_upscaled from lr_img
            if self.raw_decoder_input:
                lr_img_upscaled = self.upsampling(lr_img).view(-1, 3, self.upscale, self.upscale,
                                                               self.lr_img_res, self.lr_img_res)
                lr_img_upscaled = lr_img_upscaled.transpose(3, 4).contiguous().view(-1, 3, self.hr_img_res, self.hr_img_res)
            # Initial lr_img_upscaled from encoder_out
            else:
                lr_img_upscaled = self.upsampling(encoder_out).view(-1, self.encoder_n_patch, self.encoder_n_patch,
                                                                    self.upscaled_patch_size, self.upscaled_patch_size, 3)
                lr_img_upscaled = lr_img_upscaled.permute(0, 5, 1, 3, 2, 4).contiguous()
                lr_img_upscaled = lr_img_upscaled.view(-1, 3, self.hr_img_res, self.hr_img_res)
        
        # Make high-resolution image
        out = self.decoder_embedding_layer(lr_img_upscaled)
        if self.intermediate_upscale:
            out, lr_img_upscaled = self.transformer_decoder(out, encoder_out, lr_img_upscaled)
        else:
            out = self.transformer_decoder(out, encoder_out)
        return lr_img_upscaled + self.reconstruction_block(out)
    
    

# One-stage decoder
class OneStageDecoder(nn.Module):
    """
    Args:
        window_size : size of window
        d_embed : embedding dimension
        n_layer : number of layers in decoder
        n_head : number of heads in self-attention module
        encoder_n_patch : number of patches along an axis in encoder
        dropout : dropout ratio
    """
    def __init__(self, window_size, d_embed, n_layer,
                       n_head, encoder_n_patch, dropout):
        super(OneStageDecoder, self).__init__()
        
        # x2 upscale
        self.transformer_decoder_1 = get_transformer_decoder(d_embed=d_embed,
                                                             positional_encoding=None,
                                                             relative_position_embedding=True,
                                                             n_layer=n_layer//2,
                                                             n_head=n_head,
                                                             d_ff=d_embed*4,
                                                             query_n_patch=encoder_n_patch*2,
                                                             query_window_size=window_size*2,
                                                             key_n_patch=encoder_n_patch,
                                                             key_window_size=window_size,
                                                             dropout=dropout)
        # x2 upscale
        self.transformer_decoder_2 = get_transformer_decoder(d_embed=d_embed,
                                                             positional_encoding=None,
                                                             relative_position_embedding=True,
                                                             n_layer=n_layer//2,
                                                             n_head=n_head,
                                                             d_ff=d_embed*4,
                                                             query_n_patch=encoder_n_patch*4,
                                                             query_window_size=window_size*2,
                                                             key_n_patch=encoder_n_patch,
                                                             key_window_size=window_size//2,
                                                             dropout=dropout)
        # upsampling
        self.upsampling = UpsamplingLayer(upscale=2, d_embed=d_embed)
        
    def forward(self, x, z):
        x = self.transformer_decoder_1(x, z)
        x = self.upsampling(x)
        return self.transformer_decoder_2(x, z)
    
    
# Two-stage decoder
class TwoStageDecoder(nn.Module):
    """
    Args:
        patch_size : size of image patch
        window_size : size of window
        d_embed : embedding dimension
        n_layer : number of layers in decoder
        n_head : number of heads in self-attention module
        encoder_n_patch : number of patches along an axis in encoder
        dropout : dropout ratio
    """
    def __init__(self, patch_size, window_size, d_embed,
                       n_layer, n_head, encoder_n_patch, dropout):
        super(TwoStageDecoder, self).__init__()
        
        # x2 upscale
        self.transformer_decoder_1 = get_transformer_decoder(d_embed=d_embed,
                                                             positional_encoding=None,
                                                             relative_position_embedding=True,
                                                             n_layer=n_layer//2,
                                                             n_head=n_head,
                                                             d_ff=d_embed*4,
                                                             query_n_patch=encoder_n_patch*2,
                                                             query_window_size=window_size*2,
                                                             key_n_patch=encoder_n_patch,
                                                             key_window_size=window_size,
                                                             dropout=dropout)
        # x2 upscale
        self.transformer_decoder_2 = get_transformer_decoder(d_embed=d_embed,
                                                             positional_encoding=None,
                                                             relative_position_embedding=True,
                                                             n_layer=n_layer//2,
                                                             n_head=n_head,
                                                             d_ff=d_embed*4,
                                                             query_n_patch=encoder_n_patch*4,
                                                             query_window_size=window_size*2,
                                                             key_n_patch=encoder_n_patch,
                                                             key_window_size=window_size//2,
                                                             dropout=dropout)
        
        # intermediate reconstruction and embedding layers
        self.reconstruction_block = ReconstructionBlock(patch_size, d_embed, d_embed*4, dropout)
        self.upsampling = lambda x : F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)
        self.embedding_layer = EmbeddingLayer(patch_size, d_embed)
        
    def forward(self, x, z, origin_img):
        # Coarse decoding
        x = self.transformer_decoder_1(x, z)
        
        # Reconstruct intermediate images and upsample them.
        origin_img = self.reconstruction_block(x) + origin_img
        origin_img = self.upsampling(origin_img)
        
        # Embedding and fine decoding
        x = self.embedding_layer(origin_img)
        return self.transformer_decoder_2(x, z), origin_img