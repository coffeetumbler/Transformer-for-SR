import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn

from models.transformer import get_transformer_encoder, get_transformer_decoder
from models.submodels import EmbeddingLayer, ReconstructionBlock, UpsamplingLayer



# Whole SR Transformer model
class SRTransformer(nn.Module):
    """
    Args:
        lr_img_res : size of low-resolution image
        upscale : upscaling factor, 2 to 4
        patch_size : size of image patch
        window_size : size of window
        d_embed : embedding dimension
        encoder_n_layer : number of layers in encoder
        decoder_n_layer : number of layers in decoder
        n_head : number of heads in self-attention module
        dropout : dropout ratio
    """
    def __init__(self, lr_img_res=48, upscale=2, patch_size=2, window_size=4, d_embed=128,
                 encoder_n_layer=12, decoder_n_layer=12, n_head=4, dropout=0.1):
        super(SRTransformer, self).__init__()
        assert lr_img_res % (patch_size * window_size) == 0
        assert d_embed % n_head == 0
        
        # numbers of patches along an axis
        encoder_n_patch = lr_img_res // patch_size
        decoder_n_patch = encoder_n_patch * upscale
        
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
                                                           n_patch=encoder_n_patch,
                                                           window_size=window_size,
                                                           dropout=dropout)

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
                                                               key_n_patch=encoder_n_patch,
                                                               key_window_size=window_size,
                                                               dropout=dropout)
            
        elif upscale == 4:
            self.decoder_embedding_layer = EmbeddingLayer(patch_size*2, d_embed)
            
            # x2 upscale
            self.transformer_decoder_1 = get_transformer_decoder(d_embed=d_embed,
                                                                 positional_encoding=None,
                                                                 relative_position_embedding=True,
                                                                 n_layer=decoder_n_layer//2,
                                                                 n_head=n_head,
                                                                 d_ff=d_embed*4,
                                                                 query_n_patch=decoder_n_patch//2,
                                                                 query_window_size=window_size*2,
                                                                 key_n_patch=encoder_n_patch,
                                                                 key_window_size=window_size,
                                                                 dropout=dropout)
            # x2 upscale
            self.transformer_decoder_2 = get_transformer_decoder(d_embed=d_embed,
                                                                 positional_encoding=None,
                                                                 relative_position_embedding=True,
                                                                 n_layer=decoder_n_layer//2,
                                                                 n_head=n_head,
                                                                 d_ff=d_embed*4,
                                                                 query_n_patch=decoder_n_patch,
                                                                 query_window_size=window_size*2,
                                                                 key_n_patch=encoder_n_patch,
                                                                 key_window_size=window_size//2,
                                                                 dropout=dropout)
            # upsampling
            self.upsampling = UpsamplingLayer(upscale=2, d_embed=d_embed)
            
            # x4 upscale
            self.transformer_decoder = self.forward_upscale_4
            
    # Forward for x4 upscaling decoder
    def forward_upscale_4(self, decoder_query, encoder_output):
        decoder_query = self.transformer_decoder_1(decoder_query, encoder_output)
        decoder_query = self.upsampling(decoder_query)
        return self.transformer_decoder_2(decoder_query, encoder_output)
    
        
    def forward(self, lr_img, lr_img_upscaled):
        """
        <input>
            lr_img : (n_batch, 3, img_height, img_width), low-res image
            lr_img_upscaled : (n_batch, 3, upscale*img_height, upscale*img_width), upscaled lr_img by bicubic interpolation
            
        <output>
            hr_img : (n_batch, 3, upscale*img_height, upscale*img_width), high-res image
        """
        # Encode low-resolution image
        lr_img = self.encoder_embedding_layer(lr_img)
        lr_img = self.transformer_encoder(lr_img)
        
        # Make high-resolution image
        out = self.decoder_embedding_layer(lr_img_upscaled)
        out = self.transformer_decoder(out, lr_img)
        return lr_img_upscaled + self.reconstruction_block(out)