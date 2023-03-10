import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from models.transformer import get_transformer_encoder, get_transformer_decoder
from models.submodels import EmbeddingLayer, ReconstructionBlock, UpsamplingLayer, UnembeddingLayer

from utils.functions import clone_layer, partition_window, cyclic_shift
from utils.image_processing import imresize
import utils.config as config

IMG_SIZE_UNIT = config.IMG_SIZE_UNIT



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
                       patch_size=2, window_size=4, d_embed=128, hidden_dim_rate=2,
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
                                                           d_ff=d_embed*hidden_dim_rate,
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
                                                               d_ff=d_embed*hidden_dim_rate,
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
                lr_img_upscaled = lr_img_upscaled.permute(0, 1, 4, 2, 5, 3).contiguous().view(-1, 3, self.hr_img_res, self.hr_img_res)
            # Initial lr_img_upscaled from encoder_out
            else:
                lr_img_upscaled = self.upsampling(encoder_out).view(-1, self.encoder_n_patch, self.encoder_n_patch,
                                                                    self.upscaled_patch_size, self.upscaled_patch_size, 3)
                lr_img_upscaled = lr_img_upscaled.permute(0, 5, 1, 3, 2, 4).contiguous()
                lr_img_upscaled = lr_img_upscaled.view(-1, 3, self.hr_img_res, self.hr_img_res)
        else:
            device = lr_img.device
            with torch.no_grad():
                lr_img_upscaled = torch.stack([imresize(_lr_img, self.upscale, device=device) for _lr_img in lr_img])
        
        # Make high-resolution image
        out = self.decoder_embedding_layer(lr_img_upscaled)
        if self.intermediate_upscale:
            out, lr_img_upscaled = self.transformer_decoder(out, encoder_out, lr_img_upscaled)
        else:
            out = self.transformer_decoder(out, encoder_out)
        return lr_img_upscaled + self.reconstruction_block(out)
#         return self.reconstruction_block(out)
    
    
    
# Whole IR Transformer model
class IRTransformer(nn.Module):
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
        n_residual_block : number of residual blocks in decoder
        dropout : dropout ratio
    """
    def __init__(self, lr_img_res=48, upscale=2, patch_size=2, window_size=4,
                       d_embed=128, encoder_n_layer=12, decoder_n_layer=12, n_head=4,
                       hidden_dim_rate=2, dropout=0.1):
        super(IRTransformer, self).__init__()
        assert lr_img_res % (patch_size * window_size) == 0
        assert d_embed % n_head == 0
        
        self.lr_img_res = lr_img_res
        self.upscale = upscale
        self.hr_img_res = lr_img_res * upscale
        self.patch_size = patch_size
        self.upscaled_patch_size = patch_size * upscale
        
        # numbers of patches along an axis
        self.encoder_n_patch = lr_img_res // patch_size
        self.decoder_n_patch = self.encoder_n_patch * upscale

        # encoder embedding layer and reconstruction block
        self.encoder_embedding_layer = EmbeddingLayer(patch_size, d_embed)
        # test code
#         self.encoder_embedding_layer = EmbeddingLayer(patch_size, d_embed, d_embed//2)
#         self.half_d_embed = d_embed // 2
        
        # encoder
        self.transformer_encoder = get_transformer_encoder(d_embed=d_embed,
                                                           positional_encoding=None,
                                                           relative_position_embedding=True,
                                                           n_layer=encoder_n_layer,
                                                           n_head=n_head,
                                                           d_ff=d_embed*hidden_dim_rate,
                                                           n_patch=self.encoder_n_patch,
                                                           window_size=window_size,
                                                           dropout=dropout)
        
        # initial decoder input layer
#         self.image_upsampling = nn.Linear(d_embed, d_embed*(upscale**2)*(patch_size**2))
#         nn.init.xavier_uniform_(self.image_upsampling.weight)
#         nn.init.zeros_(self.image_upsampling.bias)
        self.feature_upsampling = UpsamplingLayer(upscale, d_embed)
        self.smoothing_layer = nn.Conv2d(d_embed, d_embed, kernel_size=2*upscale-1, stride=1, padding=upscale-1, groups=d_embed)
        nn.init.xavier_uniform_(self.smoothing_layer.weight)
        nn.init.zeros_(self.smoothing_layer.bias)

        # decoder and decoder embedding layer
#         self.decoder_embedding_layer = EmbeddingLayer(patch_size, d_embed)
        # test code
#         self.decoder_embedding_layer = EmbeddingLayer(patch_size, d_embed, d_embed//2)

        self.transformer_decoder = get_transformer_decoder(d_embed=d_embed,
                                                           positional_encoding=None,
                                                           relative_position_embedding=True,
                                                           n_layer=decoder_n_layer,
                                                           n_head=n_head,
                                                           d_ff=d_embed*hidden_dim_rate,
                                                           # query_n_patch=decoder_n_patch,
                                                           # query_window_size=window_size*upscale,
                                                           # test code
                                                           n_patch=self.decoder_n_patch,
                                                           self_window_size=window_size*2,
                                                           query_window_size=window_size*2,
                                                           # test code //
                                                           key_n_patch=self.encoder_n_patch,
                                                           # key_window_size=window_size,
                                                           # test code
                                                           key_window_size=window_size*2//upscale,
                                                           dropout=dropout)
        
        self.reconstruction_block = ReconstructionBlock(patch_size, d_embed*2, d_embed*4, dropout)
        # test code 
#         self.reconstruction_block = ReconstructionBlock(1, d_embed//2, d_embed*2, dropout)
        
        # test code
#         self.mixing_layer = nn.Linear(2*d_embed, d_embed)
#         nn.init.xavier_uniform_(self.mixing_layer.weight)
#         nn.init.zeros_(self.mixing_layer.bias)
#         self.feature_upsampling = UpsamplingLayer(upscale=upscale, d_embed=d_embed)
        # test code
        
        # test code #
#         self.initial_feature_mapping = nn.Conv2d(3, d_embed//2, kernel_size=3, stride=1, padding=1)
#         nn.init.xavier_uniform_(self.initial_feature_mapping.weight)
#         nn.init.zeros_(self.initial_feature_mapping.bias)
#         self.initial_feature_mapping = nn.Sequential(nn.Conv2d(3, d_embed//2, kernel_size=3, stride=1, padding=1),
#                                                      nn.BatchNorm2d(d_embed//2),
#                                                      nn.GELU(),
#                                                      nn.Conv2d(d_embed//2, d_embed//2, kernel_size=3, stride=1, padding=1),
#                                                      nn.BatchNorm2d(d_embed//2),
#                                                      nn.GELU(),
#                                                      nn.Conv2d(d_embed//2, d_embed//2, kernel_size=3, stride=1, padding=1),
#                                                      nn.BatchNorm2d(d_embed//2))
#         for m in self.initial_feature_mapping:
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
        
#         self.decoder_unembedding = nn.Conv2d(d_embed, (d_embed//2)*(patch_size**2), kernel_size=3, stride=1, padding=1)
#         nn.init.xavier_uniform_(self.decoder_unembedding.weight)
#         nn.init.zeros_(self.decoder_unembedding.bias)
        
#         self.lq_feature_mapping = nn.Sequential(nn.Conv2d(d_embed//2, d_embed, kernel_size=3, stride=1, padding=1),
#                                                 nn.GELU(),
#                                                 nn.Conv2d(d_embed, d_embed, kernel_size=3, stride=1, padding=1),
#                                                 nn.GELU(),
#                                                 nn.Conv2d(d_embed, (d_embed//2)*(upscale**2), kernel_size=3, stride=1, padding=1))
#         for m in self.lq_feature_mapping:
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
                
#         self.encoder_unembedding = nn.Conv2d(d_embed, (d_embed//2)*(patch_size**2), kernel_size=1, stride=1)
#         nn.init.xavier_uniform_(self.encoder_unembedding.weight)
#         nn.init.zeros_(self.encoder_unembedding.bias)
        # test code //

    def forward(self, lr_img):
        """
        <input>
            lr_img : (n_batch, 3, img_height, img_width), low-res image
            
        <output>
            hr_img : (n_batch, 3, upscale*img_height, upscale*img_width), high-res image
        """
        # test code
#         lr_img = self.initial_feature_mapping(lr_img)
#         lr_img_upscaled = self.lq_feature_mapping(lr_img).view(-1, self.half_d_embed, self.upscale, self.upscale,
#                                                                self.lr_img_res, self.lr_img_res)
#         lr_img_upscaled = lr_img_upscaled.permute(0, 1, 4, 2, 5, 3).contiguous().view(-1, self.half_d_embed, self.hr_img_res, self.hr_img_res)
        
        # Encode low-resolution image
        encoder_out = self.encoder_embedding_layer(lr_img)
        encoder_out = self.transformer_encoder(encoder_out)
        
        # Make initial upscaled lr_img
#         _lr_img_upscaled = self.image_upsampling(encoder_out).view(-1, self.encoder_n_patch, self.encoder_n_patch,
#                                             self.upscaled_patch_size, self.upscaled_patch_size, self.half_d_embed)
#         _lr_img_upscaled = _lr_img_upscaled.permute(0, 5, 1, 3, 2, 4).contiguous()
#         lr_img_upscaled = lr_img_upscaled + _lr_img_upscaled.view(-1, self.half_d_embed, self.hr_img_res, self.hr_img_res)

#         lr_img_upscaled = self.encoder_unembedding(encoder_out.permute(0, 3, 1, 2)).view(-1, self.half_d_embed, self.patch_size, self.patch_size, self.encoder_n_patch, self.encoder_n_patch)
#         lr_img_upscaled = lr_img + lr_img_upscaled.permute(0, 1, 4, 2, 5, 3).contiguous().view(-1, self.half_d_embed, self.lr_img_res, self.lr_img_res)
#         lr_img_upscaled = self.lq_feature_mapping(lr_img_upscaled).view(-1, self.half_d_embed, self.upscale, self.upscale, self.lr_img_res, self.lr_img_res)
#         lr_img_upscaled = lr_img_upscaled.transpose(3, 4).contiguous().view(-1, self.half_d_embed, self.hr_img_res, self.hr_img_res)

        # test code
        # with torch.no_grad():
        #     lr_img_upscaled = lr_img.repeat_interleave(self.upscale, dim=-1).repeat_interleave(self.upscale, dim=-2)

        # test code
#         out = self.feature_upsampling(encoder_out)

        # Make upscaled feature map
        lr_img_upscaled = self.feature_upsampling(encoder_out).permute(0, 3, 1, 2)
        lr_img_upscaled = self.smoothing_layer(lr_img_upscaled).permute(0, 2, 3, 1)

        # Make high-resolution image
        out = self.transformer_decoder(lr_img_upscaled, encoder_out)
        out = torch.cat((lr_img_upscaled, out), dim=-1)
        return self.reconstruction_block(out)
        
        # test code
#         out = self.decoder_embedding_layer(lr_img_upscaled)
#         out = self.mixing_layer(torch.cat((self.decoder_embedding_layer(lr_img_upscaled), out), dim=-1))
#         out = self.transformer_decoder(out, encoder_out)
#         return lr_img_upscaled + self.reconstruction_block(out)

        # test code #
#         out = self.transformer_decoder(out, encoder_out)
#         out = self.decoder_unembedding(out.permute(0, 3, 1, 2)).view(-1, self.half_d_embed, self.patch_size, self.patch_size, self.decoder_n_patch, self.decoder_n_patch)
#         out = lr_img_upscaled + out.permute(0, 1, 4, 2, 5, 3).contiguous().view(-1, self.half_d_embed, self.hr_img_res, self.hr_img_res)
#         return self.reconstruction_block(out.permute(0, 2, 3, 1))
        # test code //
    
    

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
    
    
########################################################################



# # Whole SR Transformer model - Encoder only version 2
# class SREncoder(nn.Module):
#     """
#     Args:
#         lr_img_res : size of low-resolution image
#         upscale : upscaling factor, 2 to 4
#         d_embed : embedding dimension
#         n_layer : number of layers in whole encoders
#         dropout : dropout ratio
#     """
#     def __init__(self, lr_img_res=48, upscale=2, d_embed=128, n_layer=12, hidden_dim_rate=2, dropout=0.1):
#         super(SREncoder, self).__init__()
#         self.lr_img_res = lr_img_res
#         self.upscale = upscale
#         self.hr_img_res = lr_img_res * upscale
#         self.d_embed = d_embed
#         self.init_d_embed = d_embed // 8
        
#         self.version = 2
        
# #         assert self.hr_img_res % (4 * window_size) == 0
#         assert d_embed % 8 == 0
        
#         # initial feature mapping layer
#         self.initial_feature_mapping = nn.Conv2d(3, (d_embed//8)*(upscale**2), kernel_size=3, stride=1, padding=1)
#         nn.init.xavier_uniform_(self.initial_feature_mapping.weight)
#         nn.init.zeros_(self.initial_feature_mapping.bias)
        
#         # embedding layers
#         self.embedding_layer_1 = EmbeddingLayer(4, d_embed, d_embed//8)
#         self.embedding_layer_2 = EmbeddingLayer(2, d_embed, d_embed//4)
#         self.embedding_layer_3 = EmbeddingLayer(1, d_embed, d_embed)

#         # transformer encoders
#         self.encoder_1 = get_transformer_encoder(d_embed=d_embed,
#                                                  positional_encoding=None,
#                                                  relative_position_embedding=True,
#                                                  n_layer=n_layer//3,
#                                                  n_head=8,
#                                                  d_ff=d_embed*hidden_dim_rate,
#                                                  n_patch=self.lr_img_res//2,
#                                                  window_size=6,
#                                                  dropout=dropout)
        
#         self.encoder_2 = get_transformer_encoder(d_embed=d_embed,
#                                                  positional_encoding=None,
#                                                  relative_position_embedding=True,
#                                                  n_layer=n_layer//3,
#                                                  n_head=8,
#                                                  d_ff=d_embed*hidden_dim_rate,
#                                                  n_patch=self.lr_img_res,
#                                                  window_size=8,
#                                                  dropout=dropout)
        
#         self.decoder = get_transformer_decoder(d_embed=d_embed,
#                                                positional_encoding=None,
#                                                relative_position_embedding=True,
#                                                n_layer=n_layer//3,
#                                                n_head=8,
#                                                d_ff=d_embed*hidden_dim_rate,
#                                                n_patch=self.hr_img_res,
#                                                self_window_size=4*upscale,
#                                                query_window_size=4*upscale,
#                                                key_n_patch=self.lr_img_res,
#                                                key_window_size=4,
#                                                dropout=dropout)
        
#         # unembedding layers
#         self.unembedding_layer_1 = UnembeddingLayer(4, d_embed//8, d_embed)
#         self.unembedding_layer_2 = UnembeddingLayer(2, d_embed//4, d_embed)
#         self.unembedding_layer_3 = UnembeddingLayer(1, d_embed, d_embed)

#         # deep feature mappings
#         self.feature_mapping_1 = nn.Conv2d(d_embed//8, d_embed//4, kernel_size=3, stride=1, padding=1)
#         self.feature_mapping_2 = nn.Conv2d(d_embed//4, d_embed, kernel_size=3, stride=1, padding=1)
#         nn.init.xavier_uniform_(self.feature_mapping_1.weight)
#         nn.init.zeros_(self.feature_mapping_1.bias)
#         nn.init.xavier_uniform_(self.feature_mapping_2.weight)
#         nn.init.zeros_(self.feature_mapping_2.bias)
        
#         # reconstruction block
#         self.reconstruction_block = ReconstructionBlock(1, d_embed, d_embed*2, dropout)

        
#     def forward(self, lr_img):
#         """
#         <input>
#             lr_img : (n_batch, 3, img_height, img_width), low-res image
            
#         <output>
#             hr_img : (n_batch, 3, upscale*img_height, upscale*img_width), high-res image
#         """
#         # Make an initial upscaled image
#         lr_img = self.initial_feature_mapping(lr_img).view(-1, self.init_d_embed, self.upscale, self.upscale, self.lr_img_res, self.lr_img_res)
#         lr_img = lr_img.permute(0, 1, 4, 2, 5, 3).contiguous().view(-1, self.init_d_embed, self.hr_img_res, self.hr_img_res)

#         # Encode low-resolution image.
#         encoder_out = self.embedding_layer_1(lr_img)
#         encoder_out = self.encoder_1(encoder_out)
#         lr_img = lr_img + self.unembedding_layer_1(encoder_out)
#         lr_img = self.feature_mapping_1(lr_img)

#         encoder_out = self.embedding_layer_2(lr_img)
#         encoder_out = self.encoder_2(encoder_out)
#         lr_img = lr_img + self.unembedding_layer_2(encoder_out)
#         lr_img = self.feature_mapping_2(lr_img)

#         # Decode high-resolution image.
#         out = self.embedding_layer_3(lr_img)
#         out = self.decoder(out, encoder_out)
#         lr_img = lr_img + self.unembedding_layer_3(out)

#         # Reconstruct high-resolution image.
#         return self.reconstruction_block(lr_img.permute(0, 2, 3, 1))



# # Whole SR Transformer model - Encoder only version 4
# class SREncoder(nn.Module):
#     """
#     Args:
#         lr_img_res : size of low-resolution image
#         upscale : upscaling factor, 2 to 4
#         d_embed : embedding dimension
#         n_layer : number of layers in whole encoders
#         dropout : dropout ratio
#     """
#     def __init__(self, lr_img_res=48, upscale=2, d_embed=128, n_layer=12, hidden_dim_rate=2, dropout=0.1, version=4):
#         super(SREncoder, self).__init__()
#         self.lr_img_res = lr_img_res
#         self.upscale = upscale
#         self.hr_img_res = lr_img_res * upscale
#         self.d_embed = d_embed
        
#         self.version = version
        
#         assert d_embed % 4 == 0
        
#         # initial feature mapping layer
#         self.initial_feature_mapping = nn.Conv2d(3, d_embed//4, kernel_size=5, stride=1, padding=2)
#         nn.init.xavier_uniform_(self.initial_feature_mapping.weight)
#         nn.init.zeros_(self.initial_feature_mapping.bias)
        
#         # embedding layers
#         self.embedding_layer_1 = EmbeddingLayer(4, 4*d_embed, d_embed//4)
#         self.embedding_layer_2 = EmbeddingLayer(2, 2*d_embed, d_embed//2)
# #         self.embedding_layer_3 = EmbeddingLayer(1, d_embed, d_embed)

#         # transformer encoders
#         self.encoder_1 = get_transformer_encoder(d_embed=4*d_embed,
#                                                  positional_encoding=None,
#                                                  relative_position_embedding=True,
#                                                  n_layer=4,
#                                                  n_head=8,
#                                                  d_ff=4*d_embed*hidden_dim_rate,
#                                                  n_patch=self.hr_img_res//4,
#                                                  window_size=6,
#                                                  dropout=dropout)
        
#         self.encoder_2 = get_transformer_encoder(d_embed=2*d_embed,
#                                                  positional_encoding=None,
#                                                  relative_position_embedding=True,
#                                                  n_layer=12,
#                                                  n_head=8,
#                                                  d_ff=2*d_embed*hidden_dim_rate,
#                                                  n_patch=self.hr_img_res//2,
#                                                  window_size=8,
#                                                  dropout=dropout)
        
#         self.encoder_3 = get_transformer_encoder(d_embed=d_embed,
#                                                  positional_encoding=None,
#                                                  relative_position_embedding=True,
#                                                  n_layer=n_layer//3,
#                                                  n_head=4,
#                                                  d_ff=d_embed*hidden_dim_rate,
#                                                  n_patch=self.hr_img_res,
#                                                  window_size=8,
#                                                  dropout=dropout)
        
#         # unembedding layers
#         self.unembedding_layer_1 = UnembeddingLayer(4, d_embed//4, 4*d_embed)
#         self.unembedding_layer_2 = UnembeddingLayer(2, d_embed//2, 2*d_embed)
# #         self.unembedding_layer_3 = UnembeddingLayer(1, d_embed, d_embed)

#         # frequency smoothing layers
#         self.smoothing_layer_1 = nn.Conv2d(d_embed//4, d_embed//4, kernel_size=7, stride=1, padding=3, groups=d_embed//4)
#         self.smoothing_layer_2 = nn.Conv2d(d_embed//2, d_embed//2, kernel_size=3, stride=1, padding=1, groups=d_embed//2)
#         nn.init.xavier_uniform_(self.smoothing_layer_1.weight)
#         nn.init.zeros_(self.smoothing_layer_1.bias)
#         nn.init.xavier_uniform_(self.smoothing_layer_2.weight)
#         nn.init.zeros_(self.smoothing_layer_2.bias)

# #         with torch.no_grad():
# #             init_filter = torch.ones(1, 7, 7)
# #             init_filter[0, 3] = 2
# #             init_filter[0, :, 3] = 2
# #             init_filter[0, 3, 3] = 4
# #             self.smoothing_layer_1.weight[:, 0] = init_filter / init_filter.sum()
# #             nn.init.zeros_(self.smoothing_layer_1.bias)

# #             init_filter = torch.Tensor([[[1,2,1],[2,4,2],[1,2,1]]])
# #             self.smoothing_layer_2.weight[:, 0] = init_filter / init_filter.sum()
# #             nn.init.zeros_(self.smoothing_layer_2.bias)

#         # deep feature mappings
#         self.feature_mapping_1 = nn.Conv2d(d_embed//2, d_embed//2, kernel_size=3, stride=1, padding=1)
#         self.feature_mapping_2 = nn.Conv2d(d_embed, d_embed, kernel_size=3, stride=1, padding=1)
#         nn.init.xavier_uniform_(self.feature_mapping_1.weight)
#         nn.init.zeros_(self.feature_mapping_1.bias)
#         nn.init.xavier_uniform_(self.feature_mapping_2.weight)
#         nn.init.zeros_(self.feature_mapping_2.bias)
        
#         # reconstruction block
#         self.reconstruction_block = ReconstructionBlock(1, d_embed*2, d_embed*4, dropout)

        
#     def forward(self, lr_img):
#         """
#         <input>
#             lr_img : (n_batch, 3, img_height, img_width), low-res image
            
#         <output>
#             hr_img : (n_batch, 3, upscale*img_height, upscale*img_width), high-res image
#         """
#         # Make an initial upscaled image
#         with torch.no_grad():
#             lr_img = lr_img.repeat_interleave(self.upscale, dim=-1).repeat_interleave(self.upscale, dim=-2)
#         lr_img = self.initial_feature_mapping(lr_img)

#         # Encode low-resolution image.
#         out = self.embedding_layer_1(lr_img)
#         out = self.encoder_1(out)
#         lr_img = torch.cat((lr_img, self.smoothing_layer_1(self.unembedding_layer_1(out))), dim=1)
#         lr_img = self.feature_mapping_1(lr_img)

#         out = self.embedding_layer_2(lr_img)
#         out = self.encoder_2(out)
#         lr_img = torch.cat((lr_img, self.smoothing_layer_2(self.unembedding_layer_2(out))), dim=1)
#         lr_img = self.feature_mapping_2(lr_img)

#         lr_img = lr_img.permute(0, 2, 3, 1)
#         out = self.encoder_3(lr_img)
#         lr_img = torch.cat((lr_img, out), dim=-1)

#         # Reconstruct high-resolution image.
#         return self.reconstruction_block(lr_img)



# Whole SR Transformer model - Encoder only version 5 & 6 & 7
class SREncoder(nn.Module):
    """
    Args:
        lr_img_res : size of low-resolution image
        upscale : upscaling factor, 2 to 4
        d_embed : embedding dimension
        n_layer : number of layers in whole encoders
        hidden_dim_rate : rate of dimension of hidden layer in FFNN
        dropout : dropout ratio
        version : encoder model version
    """
    def __init__(self, lr_img_res=48, upscale=2, d_embed=128, n_layer=12, hidden_dim_rate=2, dropout=0.1, version=6):
        super(SREncoder, self).__init__()
        self.lr_img_res = lr_img_res
        self.upscale = upscale
        self.hr_img_res = lr_img_res * upscale
        self.d_embed = d_embed
        
        self.version = version
        if version < 7:
            assert d_embed % 8 == 0
        
        # initial feature mapping layer
        self.initial_feature_mapping = nn.Conv2d(3, d_embed//8, kernel_size=2*upscale+1, stride=1, padding=upscale)
        nn.init.xavier_uniform_(self.initial_feature_mapping.weight)
#         trunc_normal_(self.initial_feature_mapping.weight, std=.02)
        nn.init.zeros_(self.initial_feature_mapping.bias)
    
        n_head_1 = 8 if version == 5 else 4
        n_head_2 = 4 if version < 7 else 8
        n_head_3 = 8 if version < 7 else 16
        
        encoder_1_d_embed = d_embed*(upscale**2)//2 if version == 5 else d_embed*(upscale**2)//8
        encoder_1_d_embed = ((encoder_1_d_embed - 1) // n_head_1 + 1) * n_head_1
        encoder_2_d_embed = d_embed*(upscale**2)//4
        encoder_2_d_embed = ((encoder_2_d_embed - 1) // n_head_2 + 1) * n_head_2
        encoder_3_d_embed = d_embed*(upscale**2)//2
        encoder_3_d_embed = ((encoder_3_d_embed - 1) // n_head_3 + 1) * n_head_3
        encoder_4_d_embed = ((d_embed - 1) // 4 + 1) * 4
        
        d_feature_1 = (encoder_1_d_embed-1)//(upscale**2)+1
        d_feature_2 = (encoder_2_d_embed-1)//(upscale**2)+1
        d_feature_3 = (encoder_3_d_embed-1)//(upscale**2)+1
        
        # embedding layers
        self.embedding_layer_1 = EmbeddingLayer(upscale*2 if version == 5 else upscale, encoder_1_d_embed, d_embed//8)
        self.embedding_layer_2 = EmbeddingLayer(upscale, encoder_2_d_embed, d_embed//4)
        self.embedding_layer_3 = EmbeddingLayer(upscale, encoder_3_d_embed, d_embed//2)

        # transformer encoders
        self.encoder_1 = get_transformer_encoder(d_embed=encoder_1_d_embed,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 n_layer=n_layer//4 if version < 7 else n_layer[0],
                                                 n_head=n_head_1,
                                                 d_ff=encoder_1_d_embed*hidden_dim_rate,
                                                 n_patch=self.lr_img_res//2 if version == 5 else self.lr_img_res,
                                                 window_size=6 if version == 5 else (12 if version == 6 else 8),
                                                 dropout=dropout)
        
        self.encoder_2 = get_transformer_encoder(d_embed=encoder_2_d_embed,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 n_layer=n_layer//4 if version < 7 else n_layer[1],
                                                 n_head=n_head_2,
                                                 d_ff=encoder_2_d_embed*hidden_dim_rate,
                                                 n_patch=self.lr_img_res,
                                                 window_size=8,
                                                 dropout=dropout)
        
        self.encoder_3 = get_transformer_encoder(d_embed=encoder_3_d_embed,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 n_layer=n_layer//4 if version < 7 else n_layer[2],
                                                 n_head=n_head_3,
                                                 d_ff=encoder_3_d_embed*hidden_dim_rate,
                                                 n_patch=self.lr_img_res,
                                                 window_size=8,
                                                 dropout=dropout)
        
        self.encoder_4 = get_transformer_encoder(d_embed=encoder_4_d_embed,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 n_layer=n_layer//4 if version < 7 else n_layer[3],
                                                 n_head=4,
                                                 d_ff=encoder_4_d_embed*hidden_dim_rate,
                                                 n_patch=self.hr_img_res,
                                                 window_size=8,
                                                 dropout=dropout)
        
        # unembedding layers
        self.unembedding_layer_1 = UnembeddingLayer(upscale*2 if version == 5 else upscale, d_feature_1, encoder_1_d_embed)
        self.unembedding_layer_2 = UnembeddingLayer(upscale, d_feature_2, encoder_2_d_embed)
        self.unembedding_layer_3 = UnembeddingLayer(upscale, d_feature_3, encoder_3_d_embed)

        # frequency smoothing layers
        self.smoothing_layer_1 = nn.Conv2d(d_feature_1, d_feature_1, kernel_size=upscale*4-1 if version == 5 else upscale*2-1,
                                           stride=1, padding=upscale*2-1 if version == 5 else upscale-1, groups=d_feature_1)
        self.smoothing_layer_2 = nn.Conv2d(d_feature_2, d_feature_2, kernel_size=upscale*2-1, stride=1, padding=upscale-1, groups=d_feature_2)
        self.smoothing_layer_3 = nn.Conv2d(d_feature_3, d_feature_3, kernel_size=upscale*2-1, stride=1, padding=upscale-1, groups=d_feature_3)
        nn.init.xavier_uniform_(self.smoothing_layer_1.weight)
#         trunc_normal_(self.smoothing_layer_1.weight, std=.02)
        nn.init.zeros_(self.smoothing_layer_1.bias)
        nn.init.xavier_uniform_(self.smoothing_layer_2.weight)
#         trunc_normal_(self.smoothing_layer_2.weight, std=.02)
        nn.init.zeros_(self.smoothing_layer_2.bias)
        nn.init.xavier_uniform_(self.smoothing_layer_3.weight)
#         trunc_normal_(self.smoothing_layer_3.weight, std=.02)
        nn.init.zeros_(self.smoothing_layer_3.bias)
    
        # test code #
        self.smoothing_layer_4 = nn.Conv2d(encoder_4_d_embed, encoder_4_d_embed, kernel_size=3, stride=1, padding=1, groups=encoder_4_d_embed)
        nn.init.xavier_uniform_(self.smoothing_layer_4.weight)
        nn.init.zeros_(self.smoothing_layer_4.bias)
        # test code //

        # deep feature mappings
#         self.feature_mapping_1 = nn.Conv2d(d_embed//4, d_embed//4, kernel_size=3, stride=1, padding=1)
#         self.feature_mapping_2 = nn.Conv2d(d_embed//2, d_embed//2, kernel_size=3, stride=1, padding=1)
#         self.feature_mapping_3 = nn.Conv2d(d_embed, d_embed, kernel_size=3, stride=1, padding=1)
        # test code #
        self.feature_mapping_1 = nn.Conv2d(d_feature_1+d_embed//8, d_embed//4, kernel_size=1, stride=1, padding=0)
        self.feature_mapping_2 = nn.Conv2d(d_feature_2+d_embed//4, d_embed//2, kernel_size=1, stride=1, padding=0)
        self.feature_mapping_3 = nn.Conv2d(d_feature_3+d_embed//2, encoder_4_d_embed, kernel_size=1, stride=1, padding=0)
        # test code //
        nn.init.xavier_uniform_(self.feature_mapping_1.weight)
        nn.init.zeros_(self.feature_mapping_1.bias)
        nn.init.xavier_uniform_(self.feature_mapping_2.weight)
        nn.init.zeros_(self.feature_mapping_2.bias)
        nn.init.xavier_uniform_(self.feature_mapping_3.weight)
        nn.init.zeros_(self.feature_mapping_3.bias)
        
        # reconstruction block
        self.reconstruction_block = ReconstructionBlock(1, encoder_4_d_embed*2, encoder_4_d_embed*4, dropout)

        
    def forward(self, lr_img, load_mask=True, padding_mask=[None for _ in range(4)], padding_info=None, save_memory=False):
        """
        <input>
            lr_img : (n_batch, 3, img_height, img_width), low-res image
            load_mask : Use self.mask in each encoder as mask input if True
            
        <output>
            hr_img : (n_batch, 3, upscale*img_height, upscale*img_width), high-res image
        """
        _, _, H, W = lr_img.shape
        
        # Make an initial upscaled image
        with torch.no_grad():
            lr_img = lr_img.repeat_interleave(self.upscale, dim=-1).repeat_interleave(self.upscale, dim=-2)
        lr_img = self.initial_feature_mapping(lr_img)
        
        if padding_info != None and len(padding_info) > 5:
            lr_img = nn.functional.pad(lr_img, (padding_info[4], padding_info[3]-padding_info[5],
                                                padding_info[1], padding_info[0]-padding_info[2]))

        # Encode low-resolution image.
        out = self.embedding_layer_1(lr_img)
        
        if padding_info != None and len(padding_info) < 5:
            out = torch.stack((out[0, :padding_info[0], :padding_info[1]], out[0, :padding_info[0], -padding_info[1]:],
                               out[0, -padding_info[0]:, :padding_info[1]], out[0, -padding_info[0]:, -padding_info[1]:]))
        
        if not save_memory:
            out = self.encoder_1(out, load_mask, padding_mask[0])
#             out = self.encoder_1(out)
        else:
            _out = []
            for i in range(len(out)):
                _out.append(self.encoder_1(out[[i]], load_mask, padding_mask[0]))
            out = torch.cat(_out, dim=0)
        
        if padding_info != None and len(padding_info) < 5:
            _IMG_SIZE_UNIT = IMG_SIZE_UNIT // 6
            out = self.merge_features(out, H, W, padding_info[0], padding_info[1], _IMG_SIZE_UNIT)
        
        out = self.unembedding_layer_1(out)
        
        if padding_info != None and len(padding_info) > 5:
            lr_img = lr_img[:, :, padding_info[1]:padding_info[2], padding_info[4]:padding_info[5]]
            out = out[:, :, padding_info[1]:padding_info[2], padding_info[4]:padding_info[5]]
        
        lr_img = torch.cat((lr_img, self.smoothing_layer_1(out)), dim=1)
        lr_img = self.feature_mapping_1(lr_img)
        
        if padding_info != None and len(padding_info) > 5:
            lr_img = nn.functional.pad(lr_img, (padding_info[4], padding_info[3]-padding_info[5],
                                                padding_info[1], padding_info[0]-padding_info[2]))

        out = self.embedding_layer_2(lr_img)
    
        if padding_info != None and len(padding_info) < 5:
            out = torch.stack((out[0, :padding_info[0], :padding_info[1]], out[0, :padding_info[0], -padding_info[1]:],
                               out[0, -padding_info[0]:, :padding_info[1]], out[0, -padding_info[0]:, -padding_info[1]:]))
    
        if not save_memory:
            out = self.encoder_2(out, load_mask, padding_mask[1])
#             out = self.encoder_2(out)
        else:
            _out = []
            for i in range(len(out)):
                _out.append(self.encoder_2(out[[i]], load_mask, padding_mask[1]))
            out = torch.cat(_out, dim=0)
        
        if padding_info != None and len(padding_info) < 5:
            out = self.merge_features(out, H, W, padding_info[0], padding_info[1], _IMG_SIZE_UNIT)
            
        out = self.unembedding_layer_2(out)
        
        if padding_info != None and len(padding_info) > 5:
            lr_img = lr_img[:, :, padding_info[1]:padding_info[2], padding_info[4]:padding_info[5]]
            out = out[:, :, padding_info[1]:padding_info[2], padding_info[4]:padding_info[5]]
            
        lr_img = torch.cat((lr_img, self.smoothing_layer_2(out)), dim=1)
        lr_img = self.feature_mapping_2(lr_img)
        
        if padding_info != None and len(padding_info) > 5:
            lr_img = nn.functional.pad(lr_img, (padding_info[4], padding_info[3]-padding_info[5],
                                                padding_info[1], padding_info[0]-padding_info[2]))
        
        out = self.embedding_layer_3(lr_img)
        
        if padding_info != None and len(padding_info) < 5:
            out = torch.stack((out[0, :padding_info[0], :padding_info[1]], out[0, :padding_info[0], -padding_info[1]:],
                               out[0, -padding_info[0]:, :padding_info[1]], out[0, -padding_info[0]:, -padding_info[1]:]))
            
        if not save_memory:
            out = self.encoder_3(out, load_mask, padding_mask[2])
#             out = self.encoder_3(out)
        else:
            _out = []
            for i in range(len(out)):
                _out.append(self.encoder_3(out[[i]], load_mask, padding_mask[2]))
            out = torch.cat(_out, dim=0)
        
        if padding_info != None and len(padding_info) < 5:
            out = self.merge_features(out, H, W, padding_info[0], padding_info[1], _IMG_SIZE_UNIT)
            
        out = self.unembedding_layer_3(out)
        
        if padding_info != None and len(padding_info) > 5:
            lr_img = lr_img[:, :, padding_info[1]:padding_info[2], padding_info[4]:padding_info[5]]
            out = out[:, :, padding_info[1]:padding_info[2], padding_info[4]:padding_info[5]]
            
        lr_img = torch.cat((lr_img, self.smoothing_layer_3(out)), dim=1)
        lr_img = self.feature_mapping_3(lr_img)
        
        out = lr_img.permute(0, 2, 3, 1)
        
        if padding_info != None and len(padding_info) > 5:
            out = nn.functional.pad(out, (0, 0, padding_info[4], padding_info[3]-padding_info[5],
                                          padding_info[1], padding_info[0]-padding_info[2]))

        if padding_info != None and len(padding_info) < 5:
            out = torch.stack((out[0, :padding_info[2], :padding_info[3]],
                               out[0, :padding_info[2], -padding_info[3]:],
                               out[0, -padding_info[2]:, :padding_info[3]],
                               out[0, -padding_info[2]:, -padding_info[3]:]))
        
        if not save_memory:
            out = self.encoder_4(out, load_mask, padding_mask[3])
#             out = self.encoder_4(out)
        else:
            _out = []
            for i in range(len(out)):
                _out.append(self.encoder_4(out[[i]], load_mask, padding_mask[3]))
            out = torch.cat(_out, dim=0)
            
        if padding_info != None and len(padding_info) > 5:
            out = out[:, padding_info[1]:padding_info[2], padding_info[4]:padding_info[5]]
            
        if padding_info != None and len(padding_info) < 5:
            H, W = self.upscale*H, self.upscale*W
            out = self.merge_features(out, H, W, padding_info[2], padding_info[3], _IMG_SIZE_UNIT)
        
        out = self.smoothing_layer_4(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # test code
        lr_img = torch.cat((lr_img.permute(0, 2, 3, 1), out), dim=-1)

        # Reconstruct high-resolution image.
        return self.reconstruction_block(lr_img)
    
    
    def merge_features(self, features, H, W, h, w, boundary):
        _h = h - boundary
        _w = w - boundary
        
        out = torch.zeros(1, H, W, features.shape[-1], device=features.device)
        out[0, :_h, :_w] += features[0, :-boundary, :-boundary]
        out[0, :_h, -_w:] += features[1, :-boundary, boundary:]
        out[0, -_h:, :_w] += features[2, boundary:, :-boundary]
        out[0, -_h:, -_w:] += features[3, boundary:, boundary:]
        
        mask = torch.zeros(1, H, W, 1, device=features.device)
        mask[0, :_h, :_w] += 1
        mask[0, :_h, -_w:] += 1
        mask[0, -_h:, :_w] += 1
        mask[0, -_h:, -_w:] += 1
        
        return out / mask
    
#     def flops(self):
#         flops = 0
#         N = self.hr_img_res * self.hr_img_res
#         # initial feature mapping
#         flops += N * 3 * (self.d_embed//8) * 25
#         # embedding and unembedding layers
#         flops += 2 * N * (self.d_embed//8) * (2*self.d_embed)
#         flops += 2 * N * (self.d_embed//4) * (self.d_embed)
#         flops += 2 * N * (self.d_embed//2) * (2*self.d_embed)
#         # encoder blocks
#         flops += self.encoder_1.flops() + self.encoder_2.flops() + self.encoder_3.flops() + self.encoder_4.flops()
#         # frequency smoothing layers
#         flops += N * ((self.d_embed//8) * 49 + (self.d_embed//4) * 9 + (self.d_embed//2) * 9)
#         # deep feature mappings
#         flops += N * ((self.d_embed//4) ** 2) * 9
#         flops += N * ((self.d_embed//2) ** 2) * 9
#         flops += N * (self.d_embed ** 2) * 9
#         # reconstruction block
#         flops += N * ((2*self.d_embed) * (4*self.d_embed) + (4*self.d_embed) * 3)
#         return flops

    def evaluate(self, lr_img, pad_boundary=True, trim_boundary=True, save_memory=False):
        """
        <input>
            lr_img : (1, 3, img_height, img_width), low-res image
            trim_boundary : Trim inner boundary of patches if True
            
        <output>
            hr_img : (1, 3, upscale*img_height, upscale*img_width), high-res image
        """
        _, _, H, W = lr_img.shape
        if self.version == 7:
            IMG_SIZE_UNIT = 8
        
        if pad_boundary:
    #         _, _, H_0, W_0 = lr_img.shape
    #         lr_img = lr_img[:, :, :H_0-H_0%2, :W_0-W_0%2]
    #         lr_img = nn.functional.pad(lr_img, (0, W_0%2, 0, H_0%2), mode='constant')
    
    #         pad_values = ((torch.ones(3) - torch.Tensor(config.IMG_NORM_MEAN)) / torch.Tensor(config.IMG_NORM_STD))
    #         pad_values = pad_values.view(3, 1, 1).to(lr_img.device)

            # Set padding options.
            H_pad = ((H - 1) // IMG_SIZE_UNIT + 1) * IMG_SIZE_UNIT
            W_pad = ((W - 1) // IMG_SIZE_UNIT + 1) * IMG_SIZE_UNIT
    #         H_pad = (H // _IMG_SIZE_UNIT) * _IMG_SIZE_UNIT
    #         W_pad = (W // _IMG_SIZE_UNIT) * _IMG_SIZE_UNIT

    #         lpad_h = (H_pad - H) // 2
    # #         lpad_h = (lpad_h // 2) * 2
    #         rpad_h = H + lpad_h

    #         lpad_w = (W_pad - W) // 2
    # #         lpad_w = (lpad_w // 2) * 2
    #         rpad_w = W + lpad_w

            # Pad black pixels to lr_img.
    #         padding_mask = nn.functional.pad(torch.ones(1, H, W, device=lr_img.device), (lpad_w, W_pad-rpad_w, lpad_h, H_pad-rpad_h))
            padding_mask = nn.functional.pad(torch.ones(1, H, W, device=lr_img.device), (0, W_pad-W, 0, H_pad-H))
    #         lr_img = nn.functional.pad(lr_img, (lpad_w, W_pad-rpad_w, lpad_h, H_pad-rpad_h), mode='reflect')
#             lr_img = nn.functional.pad(lr_img, (0, W_pad-W, 0, H_pad-H), mode='reflect')
    #         lr_img[0, :, :max(lpad_h-4,0)] = 0
    #         lr_img[0, :, min(rpad_h+4,H_pad):] = 0
    #         lr_img[0, :, :, :max(lpad_w-4,0)] = 0
    #         lr_img[0, :, :, min(rpad_w+4,W_pad):] = 0

    #         mask_1 = nn.functional.max_pool2d(padding_mask, kernel_size=2, stride=2).expand(8, -1, -1).view(1, 8, H_pad//2, W_pad//2, 1)
            mask_1 = padding_mask.clone().expand(self.encoder_1.n_head, -1, -1).view(1, self.encoder_1.n_head, H_pad, W_pad, 1)
            mask_2 = padding_mask.clone().expand(self.encoder_2.n_head, -1, -1).view(1, self.encoder_2.n_head, H_pad, W_pad, 1)
            mask_3 = padding_mask.clone().expand(self.encoder_3.n_head, -1, -1).view(1, self.encoder_3.n_head, H_pad, W_pad, 1)
            mask_4 = padding_mask.repeat_interleave(self.upscale, dim=-1).repeat_interleave(self.upscale, dim=-2)
            mask_4 = mask_4.expand(4, -1, -1).view(1, 4, H_pad*self.upscale, W_pad*self.upscale, 1)

            mask_1 = partition_window(cyclic_shift(mask_1, self.encoder_1.window_size//2), self.encoder_1.window_size)
            mask_2 = partition_window(cyclic_shift(mask_2, self.encoder_2.window_size//2), self.encoder_2.window_size)
            mask_3 = partition_window(cyclic_shift(mask_3, self.encoder_3.window_size//2), self.encoder_3.window_size)
            mask_4 = partition_window(cyclic_shift(mask_4, self.encoder_4.window_size//2), self.encoder_4.window_size)

            mask_1 = (mask_1 - mask_1.transpose(-1, -2)) != 0
            mask_2 = (mask_2 - mask_2.transpose(-1, -2)) != 0
            mask_3 = (mask_3 - mask_3.transpose(-1, -2)) != 0
            mask_4 = (mask_4 - mask_4.transpose(-1, -2)) != 0

    #         hr_img = self.forward(lr_img, False, [mask_1, mask_2, mask_3, mask_4],
    #                               list(self.upscale*np.array([H_pad, lpad_h, rpad_h, W_pad, lpad_w, rpad_w])))
            hr_img = self.forward(lr_img, False, [mask_1, mask_2, mask_3, mask_4],
                                  list(self.upscale*np.array([H_pad, 0, H, W_pad, 0, W])))
#             hr_img = self.forward(lr_img, False)

#             return nn.functional.pad(hr_img, (0, self.upscale*(W_0%2), 0, self.upscale*(H_0%2)), mode='constant')
            return hr_img # hr_img[:, :, self.upscale*lpad_h:self.upscale*rpad_h, self.upscale*lpad_w:self.upscale*rpad_w]
#             return hr_img[:, :, :self.upscale*H, :self.upscale*W]
            
        else:
            # Set padding options.
#             H_pad = ((H + 1 // 2) // IMG_SIZE_UNIT + 2) * IMG_SIZE_UNIT
#             W_pad = ((W + 1 // 2) // IMG_SIZE_UNIT + 2) * IMG_SIZE_UNIT
            H_pad = H // IMG_SIZE_UNIT * IMG_SIZE_UNIT
            W_pad = W // IMG_SIZE_UNIT * IMG_SIZE_UNIT

            # Upscale the image.
            upscaled_H, upscaled_W = self.upscale*H_pad, self.upscale*W_pad
            hr_img = self.forward(lr_img, False, [None for _ in range(4)], [H_pad, W_pad, upscaled_H, upscaled_W], save_memory)
            return hr_img
    

    
# # Whole SR Transformer model - Encoder only version 1 & 3
# class SREncoder(nn.Module):
#     """
#     Args:
#         lr_img_res : size of low-resolution image
#         upscale : upscaling factor, 2 to 4
#         d_embed : embedding dimension
#         n_layer : number of layers in whole encoders
#         dropout : dropout ratio
#     """
#     def __init__(self, lr_img_res=48, upscale=2, d_embed=128, n_layer=12, hidden_dim_rate=2, dropout=0.1, version=3):
#         super(SREncoder, self).__init__()
#         self.lr_img_res = lr_img_res
#         self.upscale = upscale
#         self.hr_img_res = lr_img_res * upscale
#         self.d_embed = d_embed
#         self.init_d_embed = d_embed // 4
        
#         self.version = version
        
#         assert d_embed % 8 == 0
        
#         # initial feature mapping layer
#         if version == 1:
#             self.initial_feature_mapping = nn.Conv2d(3, (d_embed//4)*(upscale**2), kernel_size=3, stride=1, padding=1)
#         elif version == 3:
#             self.initial_feature_mapping = nn.Conv2d(3, d_embed//4, kernel_size=5, stride=1, padding=2)
#         nn.init.xavier_uniform_(self.initial_feature_mapping.weight)
#         nn.init.zeros_(self.initial_feature_mapping.bias)
        
#         # embedding layers
#         self.embedding_layer_1 = EmbeddingLayer(4, d_embed, d_embed//4)
#         self.embedding_layer_2 = EmbeddingLayer(2, d_embed, d_embed//2)
#         self.embedding_layer_3 = EmbeddingLayer(1, d_embed, d_embed)
# #         self.embedding_layer_4 = EmbeddingLayer(1, d_embed, d_embed)

#         # transformer encoders
#         self.encoder_1 = get_transformer_encoder(d_embed=d_embed,
#                                                  positional_encoding=None,
#                                                  relative_position_embedding=True,
#                                                  n_layer=n_layer//3,
#                                                  n_head=8,
#                                                  d_ff=d_embed*hidden_dim_rate,
#                                                  n_patch=self.hr_img_res//4,
#                                                  window_size=6,
#                                                  dropout=dropout)
        
#         self.encoder_2 = get_transformer_encoder(d_embed=d_embed,
#                                                  positional_encoding=None,
#                                                  relative_position_embedding=True,
#                                                  n_layer=n_layer//3,
#                                                  n_head=8,
#                                                  d_ff=d_embed*hidden_dim_rate,
#                                                  n_patch=self.hr_img_res//2,
#                                                  window_size=8,
#                                                  dropout=dropout)
        
#         self.encoder_3 = get_transformer_encoder(d_embed=d_embed,
#                                                  positional_encoding=None,
#                                                  relative_position_embedding=True,
#                                                  n_layer=n_layer//3,
#                                                  n_head=8,
#                                                  d_ff=d_embed*hidden_dim_rate,
#                                                  n_patch=self.hr_img_res,
#                                                  window_size=8,
#                                                  dropout=dropout)
        
# #         self.encoder_4 = get_transformer_encoder(d_embed=d_embed,
# #                                                  positional_encoding=None,
# #                                                  relative_position_embedding=True,
# #                                                  n_layer=n_layer//4,
# #                                                  n_head=8,
# #                                                  d_ff=d_embed*hidden_dim_rate,
# #                                                  n_patch=self.hr_img_res,
# #                                                  window_size=8,
# #                                                  dropout=dropout)
        
#         # unembedding layers
#         self.unembedding_layer_1 = UnembeddingLayer(4, d_embed//4, d_embed)
#         self.unembedding_layer_2 = UnembeddingLayer(2, d_embed//2, d_embed)
#         self.unembedding_layer_3 = UnembeddingLayer(1, d_embed, d_embed)
# #         self.unembedding_layer_4 = UnembeddingLayer(1, d_embed, d_embed)

#         # frequency smoothing layers
#         if version == 3:
#             self.smoothing_layer_1 = nn.Conv2d(d_embed//4, d_embed//4, kernel_size=7, stride=1, padding=3, groups=d_embed//4)
#             self.smoothing_layer_2 = nn.Conv2d(d_embed//2, d_embed//2, kernel_size=3, stride=1, padding=1, groups=d_embed//2)
# #             nn.init.xavier_uniform_(self.smoothing_layer_1.weight)
# #             nn.init.zeros_(self.smoothing_layer_1.bias)
# #             nn.init.xavier_uniform_(self.smoothing_layer_2.weight)
# #             nn.init.zeros_(self.smoothing_layer_2.bias)

#             with torch.no_grad():
#                 init_filter = torch.ones(1, 7, 7)
#                 init_filter[0, 3] = 2
#                 init_filter[0, :, 3] = 2
#                 init_filter[0, 3, 3] = 4
#                 self.smoothing_layer_1.weight[:, 0] = init_filter / init_filter.sum()
#                 nn.init.zeros_(self.smoothing_layer_1.bias)
                
#                 init_filter = torch.Tensor([[[1,2,1],[2,4,2],[1,2,1]]])
#                 self.smoothing_layer_2.weight[:, 0] = init_filter / init_filter.sum()
#                 nn.init.zeros_(self.smoothing_layer_2.bias)

#         # deep feature mappings
#         self.feature_mapping_1 = nn.Conv2d(d_embed//4, d_embed//2, kernel_size=3, stride=1, padding=1)
#         self.feature_mapping_2 = nn.Conv2d(d_embed//2, d_embed, kernel_size=3, stride=1, padding=1)
#         nn.init.xavier_uniform_(self.feature_mapping_1.weight)
#         nn.init.zeros_(self.feature_mapping_1.bias)
#         nn.init.xavier_uniform_(self.feature_mapping_2.weight)
#         nn.init.zeros_(self.feature_mapping_2.bias)
        
#         # reconstruction block
#         self.reconstruction_block = ReconstructionBlock(1, d_embed, d_embed*2, dropout)

        
#     def forward(self, lr_img):
#         """
#         <input>
#             lr_img : (n_batch, 3, img_height, img_width), low-res image
            
#         <output>
#             hr_img : (n_batch, 3, upscale*img_height, upscale*img_width), high-res image
#         """
#         # Make an initial upscaled image
#         if self.version in (1,3):
#             if self.version < 3:
#                 lr_img = self.initial_feature_mapping(lr_img).view(-1, self.init_d_embed, self.upscale, self.upscale, self.lr_img_res, self.lr_img_res)
#                 lr_img = lr_img.permute(0, 1, 4, 2, 5, 3).contiguous().view(-1, self.init_d_embed, self.hr_img_res, self.hr_img_res)
#             else:
#                 with torch.no_grad():
#                     lr_img = lr_img.repeat_interleave(self.upscale, dim=-1).repeat_interleave(self.upscale, dim=-2)
#                 lr_img = self.initial_feature_mapping(lr_img)

#             # Encode low-resolution image.
#             out = self.embedding_layer_1(lr_img)
#             out = self.encoder_1(out)
#             if self.version < 3:
#                 lr_img = lr_img + self.unembedding_layer_1(out)
#             else:
#                 lr_img = lr_img + self.smoothing_layer_1(self.unembedding_layer_1(out))
#             lr_img = self.feature_mapping_1(lr_img)

#             out = self.embedding_layer_2(lr_img)
#             out = self.encoder_2(out)
#             if self.version < 3:
#                 lr_img = lr_img + self.unembedding_layer_2(out)
#             else:
#                 lr_img = lr_img + self.smoothing_layer_2(self.unembedding_layer_2(out))
#             lr_img = self.feature_mapping_2(lr_img)

#             out = self.embedding_layer_3(lr_img)
#             out = self.encoder_3(out)
#             if self.version < 3:
#                 lr_img = lr_img.permute(0, 2, 3, 1) + out
#             else:
#                 lr_img = (lr_img + self.unembedding_layer_3(out)).permute(0, 2, 3, 1)

#     #         out = self.embedding_layer_4(lr_img)
#     #         out = self.encoder_4(out)
#     #         lr_img = lr_img + self.unembedding_layer_4(out)

#             # Reconstruct high-resolution image.
#             return self.reconstruction_block(lr_img)

#         elif self.version == 2:
#             # Make an initial upscaled image
#             lr_img = self.initial_feature_mapping(lr_img).view(-1, self.init_d_embed, self.upscale, self.upscale, self.lr_img_res, self.lr_img_res)
#             lr_img = lr_img.permute(0, 1, 4, 2, 5, 3).contiguous().view(-1, self.init_d_embed, self.hr_img_res, self.hr_img_res)

#             # Encode low-resolution image.
#             encoder_out = self.embedding_layer_1(lr_img)
#             encoder_out = self.encoder_1(encoder_out)
#             lr_img = lr_img + self.unembedding_layer_1(encoder_out)
#             lr_img = self.feature_mapping_1(lr_img)

#             encoder_out = self.embedding_layer_2(lr_img)
#             encoder_out = self.encoder_2(encoder_out)
#             lr_img = lr_img + self.unembedding_layer_2(encoder_out)
#             lr_img = self.feature_mapping_2(lr_img)

#             # Decode high-resolution image.
#             out = self.embedding_layer_3(lr_img)
#             out = self.decoder(out, encoder_out)
#             lr_img = lr_img + self.unembedding_layer_3(out)

#             # Reconstruct high-resolution image.
#             return self.reconstruction_block(lr_img.permute(0, 2, 3, 1))
        
#         elif self.version == 4:
#             # Make an initial upscaled image
#             with torch.no_grad():
#                 lr_img = lr_img.repeat_interleave(self.upscale, dim=-1).repeat_interleave(self.upscale, dim=-2)
#             lr_img = self.initial_feature_mapping(lr_img)

#             # Encode low-resolution image.
#             out = self.embedding_layer_1(lr_img)
#             out = self.encoder_1(out)
#             lr_img = torch.cat((lr_img, self.smoothing_layer_1(self.unembedding_layer_1(out))), dim=1)
#             lr_img = self.feature_mapping_1(lr_img)

#             out = self.embedding_layer_2(lr_img)
#             out = self.encoder_2(out)
#             lr_img = torch.cat((lr_img, self.smoothing_layer_2(self.unembedding_layer_2(out))), dim=1)
#             lr_img = self.feature_mapping_2(lr_img)

#             lr_img = lr_img.permute(0, 2, 3, 1)
#             out = self.encoder_3(lr_img)
#             lr_img = torch.cat((lr_img, out), dim=-1)

#             # Reconstruct high-resolution image.
#             return self.reconstruction_block(lr_img)
        
#         elif self.version == 5:
#             # Make an initial upscaled image
#             with torch.no_grad():
#                 lr_img = lr_img.repeat_interleave(self.upscale, dim=-1).repeat_interleave(self.upscale, dim=-2)
#             lr_img = self.initial_feature_mapping(lr_img)

#             # Encode low-resolution image.
#             out = self.embedding_layer_1(lr_img)
#             out = self.encoder_1(out)
#             lr_img = torch.cat((lr_img, self.smoothing_layer_1(self.unembedding_layer_1(out))), dim=1)
#             lr_img = self.feature_mapping_1(lr_img)

#             out = self.embedding_layer_2(lr_img)
#             out = self.encoder_2(out)
#             lr_img = torch.cat((lr_img, self.smoothing_layer_2(self.unembedding_layer_2(out))), dim=1)
#             lr_img = self.feature_mapping_2(lr_img)

#             out = self.embedding_layer_3(lr_img)
#             out = self.encoder_3(out)
#             lr_img = torch.cat((lr_img, self.smoothing_layer_3(self.unembedding_layer_3(out))), dim=1)
#             lr_img = self.feature_mapping_3(lr_img)

#             lr_img = lr_img.permute(0, 2, 3, 1)
#             out = self.encoder_4(lr_img)
#             lr_img = torch.cat((lr_img, out), dim=-1)

#             # Reconstruct high-resolution image.
#             return self.reconstruction_block(lr_img)

        
        
# # Whole SR Transformer model - Encoder only version
# class SREncoder(nn.Module):
#     """
#     Args:
#         lr_img_res : size of low-resolution image
#         upscale : upscaling factor, 2 to 4
#         patch_size : size of image patch
#         window_size : size of window
#         d_embed : embedding dimension
#         lr_encoder_n_layer : number of layers in low-resolution encoder
#         hr_encoder_n_layer : number of layers in high-resolution encoder
#         n_head : number of heads in self-attention module
#         residual_connection : True if a reconstructed image is created by residual connection
#         dropout : dropout ratio
#     """
#     def __init__(self, lr_img_res=48, upscale=2, patch_size=2, window_size=4,
#                        d_embed=128, n_layer=12, hidden_dim_rate=2,
#                        n_head=4, residual_connection=True, dropout=0.1):
#         super(SREncoder, self).__init__()
#         assert lr_img_res % (patch_size * window_size) == 0
#         assert d_embed % n_head == 0
        
#         self.lr_img_res = lr_img_res
#         self.upscale = upscale
#         self.hr_img_res = lr_img_res * upscale
#         self.residual_connection = residual_connection
        
#         # numbers of patches along an axis
#         encoder_n_patch = self.hr_img_res // patch_size
        
#         # embedding layer
#         self.embedding_layer = EmbeddingLayer(patch_size, d_embed)

#         # transformer encoder
#         self.encoder = get_transformer_encoder(d_embed=d_embed,
#                                                   positional_encoding=None,
#                                                   relative_position_embedding=True,
#                                                   n_layer=n_layer,
#                                                   n_head=n_head,
#                                                   d_ff=d_embed*hidden_dim_rate,
#                                                   n_patch=encoder_n_patch,
#                                                   window_size=window_size,
#                                                   dropout=dropout)
        
#         """
#         # high-resolution encoder and upsampling layer
#         if upscale > 1 and upscale < 4:
#             self.encoder_upsampling = UpsamplingLayer(upscale, d_embed)
#             self.hr_encoder = get_transformer_encoder(d_embed=d_embed,
#                                                       positional_encoding=None,
#                                                       relative_position_embedding=True,
#                                                       n_layer=hr_encoder_n_layer,
#                                                       n_head=n_head,
#                                                       d_ff=d_embed*4,
#                                                       n_patch=hr_encoder_n_patch,
#                                                       window_size=window_size*upscale,
#                                                       dropout=dropout)
#         elif upscale == 4:
#             self.encoder_upsampling = UpsamplingLayer(2, d_embed)
#             self.hr_encoder_1 = get_transformer_encoder(d_embed=d_embed,
#                                                         positional_encoding=None,
#                                                         relative_position_embedding=True,
#                                                         n_layer=hr_encoder_n_layer//2,
#                                                         n_head=n_head,
#                                                         d_ff=d_embed*4,
#                                                         n_patch=lr_encoder_n_patch*2,
#                                                         window_size=window_size*2,
#                                                         dropout=dropout)
#             self.intermediate_upsampling = UpsamplingLayer(2, d_embed)
#             self.hr_encoder_2 = get_transformer_encoder(d_embed=d_embed,
#                                                         positional_encoding=None,
#                                                         relative_position_embedding=True,
#                                                         n_layer=hr_encoder_n_layer//2,
#                                                         n_head=n_head,
#                                                         d_ff=d_embed*4,
#                                                         n_patch=hr_encoder_n_patch,
#                                                         window_size=window_size*2,
#                                                         dropout=dropout)
#             self.hr_encoder = self.upscale_4_encoder
        
#         # image upsampling layer
#         if residual_connection:
#             self.img_upsampling = nn.Conv2d(3, 3*(upscale**2), kernel_size=1, stride=1)
#             nn.init.xavier_uniform_(self.img_upsampling.weight)
#             nn.init.zeros_(self.img_upsampling.bias)
#         """

#         # reconstruction block
#         self.reconstruction_block = ReconstructionBlock(patch_size, d_embed, d_embed*4, dropout)

#         # test code #
#         self.upsampling = nn.Conv2d(3, 3*(upscale**2), kernel_size=3, stride=1, padding=1)
#         nn.init.xavier_uniform_(self.upsampling.weight)
#         nn.init.zeros_(self.upsampling.bias)
#         # test code //

#     # Computation for high-resolution encoder of upscale 4
#     def upscale_4_encoder(self, x):
#         x = self.hr_encoder_1(x)
#         x = self.intermediate_upsampling(x)
#         return self.hr_encoder_2(x)
        
#     def forward(self, lr_img):
#         """
#         <input>
#             lr_img : (n_batch, 3, img_height, img_width), low-res image
            
#         <output>
#             hr_img : (n_batch, 3, upscale*img_height, upscale*img_width), high-res image
#         """
#         # Make an initial upscaled image
#         lr_img = self.upsampling(lr_img).view(-1, 3, self.upscale, self.upscale, self.lr_img_res, self.lr_img_res)
#         lr_img = lr_img.transpose(3, 4).contiguous().view(-1, 3, self.hr_img_res, self.hr_img_res)
#         # device = lr_img.device
#         # with torch.no_grad():
#             # lr_img = torch.stack([imresize(_lr_img, self.upscale, device=device) for _lr_img in lr_img])
#             # lr_img = lr_img.repeat_interleave(self.upscale, dim=-1).repeat_interleave(self.upscale, dim=-2)

#         # Encode low-resolution image.
#         out = self.embedding_layer(lr_img)
#         out = self.encoder(out)
        
#         # Encode high-resolution information.
#         # out = self.encoder_upsampling(out)
#         # out = self.hr_encoder(out)
        
#         # Reconstruct high-resolution image.
#         if self.residual_connection:  # with residual connection
#             # Make an initial upscaled image.
#             # lr_img = self.img_upsampling(lr_img).view(-1, 3, self.upscale, self.upscale, self.lr_img_res, self.lr_img_res)
#             # lr_img = lr_img.transpose(3, 4).contiguous().view(-1, 3, self.hr_img_res, self.hr_img_res)
#             return lr_img + self.reconstruction_block(out)
#         else:  # without residual connection
#             return self.reconstruction_block(out)



##############################################


# Whole SR Transformer model - Encoder only for ablation
class SREncoderAblation(nn.Module):
    """
    Args:
        lr_img_res : size of low-resolution image
        upscale : upscaling factor, 2 to 4
        d_embed : embedding dimension
        n_layer : number of layers in whole encoders
        dropout : dropout ratio
    """
    def __init__(self, lr_img_res=48, upscale=2, d_embed=128, n_layer=12, hidden_dim_rate=2, dropout=0.1, version=1.1):
        super(SREncoderAblation, self).__init__()
        self.lr_img_res = lr_img_res
        self.upscale = upscale
        self.hr_img_res = lr_img_res * upscale
        self.d_embed = d_embed
        
        self.version = version
        
        assert d_embed % 8 == 0
        
        # initial feature mapping layer
        self.initial_feature_mapping = nn.Conv2d(3, d_embed//8, kernel_size=2*upscale+1, stride=1, padding=upscale)
        nn.init.xavier_uniform_(self.initial_feature_mapping.weight)
        nn.init.zeros_(self.initial_feature_mapping.bias)
        
        encoder_1_d_embed = d_embed*(upscale**2)//2
        encoder_2_d_embed = d_embed*(upscale**2)//4
        encoder_3_d_embed = d_embed*(upscale**2)//2
        
        # embedding layers
        self.embedding_layer_1 = EmbeddingLayer(upscale*2, encoder_1_d_embed, d_embed//8)
        self.embedding_layer_2 = EmbeddingLayer(upscale, encoder_2_d_embed, d_embed//4)
        self.embedding_layer_3 = EmbeddingLayer(upscale, encoder_3_d_embed, d_embed//2)

        # transformer encoders
        self.encoder_1 = get_transformer_encoder(d_embed=encoder_1_d_embed,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 n_layer=n_layer//4,
                                                 n_head=8,
                                                 d_ff=encoder_1_d_embed*hidden_dim_rate,
                                                 n_patch=self.lr_img_res//2,
                                                 window_size=6,
                                                 dropout=dropout)
        
        self.encoder_2 = get_transformer_encoder(d_embed=encoder_2_d_embed,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 n_layer=n_layer//4,
                                                 n_head=4,
                                                 d_ff=encoder_2_d_embed*hidden_dim_rate,
                                                 n_patch=self.lr_img_res,
                                                 window_size=8,
                                                 dropout=dropout)
        
        self.encoder_3 = get_transformer_encoder(d_embed=encoder_3_d_embed,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 n_layer=n_layer//4,
                                                 n_head=8,
                                                 d_ff=encoder_3_d_embed*hidden_dim_rate,
                                                 n_patch=self.lr_img_res,
                                                 window_size=6,
                                                 dropout=dropout)
        
        self.encoder_4 = get_transformer_encoder(d_embed=d_embed,
                                                 positional_encoding=None,
                                                 relative_position_embedding=True,
                                                 n_layer=n_layer//4,
                                                 n_head=4,
                                                 d_ff=d_embed*hidden_dim_rate,
                                                 n_patch=self.hr_img_res,
                                                 window_size=8,
                                                 dropout=dropout)
        
        # unembedding layers
        self.unembedding_layer_1 = UnembeddingLayer(upscale*2, d_embed//8, encoder_1_d_embed)
        self.unembedding_layer_2 = UnembeddingLayer(upscale, d_embed//4, encoder_2_d_embed)
        self.unembedding_layer_3 = UnembeddingLayer(upscale, d_embed//2, encoder_3_d_embed)

        # frequency smoothing layers
        if version != 1.1:
            self.smoothing_layer_1 = nn.Conv2d(d_embed//8, d_embed//8, kernel_size=upscale*4-1, stride=1,
                                               padding=upscale*2-1, groups=d_embed//8 if version != 1.2 else 1)
            self.smoothing_layer_2 = nn.Conv2d(d_embed//4, d_embed//4, kernel_size=upscale*2-1, stride=1,
                                               padding=upscale-1, groups=d_embed//4 if version != 1.2 else 1)
            self.smoothing_layer_3 = nn.Conv2d(d_embed//2, d_embed//2, kernel_size=upscale*2-1, stride=1,
                                               padding=upscale-1, groups=d_embed//2 if version != 1.2 else 1)
            nn.init.xavier_uniform_(self.smoothing_layer_1.weight)
            nn.init.zeros_(self.smoothing_layer_1.bias)
            nn.init.xavier_uniform_(self.smoothing_layer_2.weight)
            nn.init.zeros_(self.smoothing_layer_2.bias)
            nn.init.xavier_uniform_(self.smoothing_layer_3.weight)
            nn.init.zeros_(self.smoothing_layer_3.bias)
        else:
            self.smoothing_layer_1 = nn.Identity()
            self.smoothing_layer_2 = nn.Identity()
            self.smoothing_layer_3 = nn.Identity()

        # deep feature mappings
        self.feature_mapping_1 = nn.Conv2d(d_embed//4, d_embed//4, kernel_size=3, stride=1, padding=1)
        self.feature_mapping_2 = nn.Conv2d(d_embed//2, d_embed//2, kernel_size=3, stride=1, padding=1)
        self.feature_mapping_3 = nn.Conv2d(d_embed, d_embed, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.feature_mapping_1.weight)
        nn.init.zeros_(self.feature_mapping_1.bias)
        nn.init.xavier_uniform_(self.feature_mapping_2.weight)
        nn.init.zeros_(self.feature_mapping_2.bias)
        nn.init.xavier_uniform_(self.feature_mapping_3.weight)
        nn.init.zeros_(self.feature_mapping_3.bias)
        
        # reconstruction block
        self.reconstruction_block = ReconstructionBlock(1, d_embed*2, d_embed*4, dropout)

        
    def forward(self, lr_img, load_mask=True):
        """
        <input>
            lr_img : (n_batch, 3, img_height, img_width), low-res image
            load_mask : Use self.mask in each encoder as mask input if True
            
        <output>
            hr_img : (n_batch, 3, upscale*img_height, upscale*img_width), high-res image
        """
        # Make an initial upscaled image
        with torch.no_grad():
            lr_img = lr_img.repeat_interleave(self.upscale, dim=-1).repeat_interleave(self.upscale, dim=-2)
        lr_img = self.initial_feature_mapping(lr_img)

        # Encode low-resolution image.
        out = self.embedding_layer_1(lr_img)
        out = self.encoder_1(out, load_mask)
        lr_img = torch.cat((lr_img, self.smoothing_layer_1(self.unembedding_layer_1(out))), dim=1)
        lr_img = self.feature_mapping_1(lr_img)

        out = self.embedding_layer_2(lr_img)
        out = self.encoder_2(out, load_mask)
        lr_img = torch.cat((lr_img, self.smoothing_layer_2(self.unembedding_layer_2(out))), dim=1)
        lr_img = self.feature_mapping_2(lr_img)
        
        out = self.embedding_layer_3(lr_img)
        out = self.encoder_3(out, load_mask)
        lr_img = torch.cat((lr_img, self.smoothing_layer_3(self.unembedding_layer_3(out))), dim=1)
        lr_img = self.feature_mapping_3(lr_img)

        lr_img = lr_img.permute(0, 2, 3, 1)
        out = self.encoder_4(lr_img, load_mask)
        lr_img = torch.cat((lr_img, out), dim=-1)

        # Reconstruct high-resolution image.
        return self.reconstruction_block(lr_img)


    def evaluate(self, lr_img, trim_boundary=True):
        """
        <input>
            lr_img : (1, 3, img_height, img_width), low-res image
            trim_boundary : Trim inner boundary of patches if True
            
        <output>
            hr_img : (1, 3, upscale*img_height, upscale*img_width), high-res image
        """
        _, _, H, W = lr_img.shape
#         pad_values = ((torch.ones(3) - torch.Tensor(config.IMG_NORM_MEAN)) / torch.Tensor(config.IMG_NORM_STD))
#         pad_values = pad_values.view(3, 1, 1).to(lr_img.device)
        
        # Set padding options.
#         H_pad = ((H - 1) // _IMG_SIZE_UNIT + 1) * _IMG_SIZE_UNIT
#         W_pad = ((W - 1) // _IMG_SIZE_UNIT + 1) * _IMG_SIZE_UNIT
        H_pad = (H // IMG_SIZE_UNIT) * IMG_SIZE_UNIT
        W_pad = (W // IMG_SIZE_UNIT) * IMG_SIZE_UNIT
        
#         lpad_h = (H_pad - H) // 2
#         rpad_h = H + lpad_h
#         lpad_w = (W_pad - W) // 2
#         rpad_w = W + lpad_w
        
        # Pad black pixels to lr_img.
#         lr_img = nn.functional.pad(lr_img, (lpad_w, W_pad-rpad_w, lpad_h, H_pad-rpad_h), mode='reflect')
#         lr_img = nn.functional.pad(lr_img, (0, W_pad-W, 0, H_pad-H), mode='reflect')
#         lr_img[0, :, :max(lpad_h-4,0)] = 0
#         lr_img[0, :, min(rpad_h+4,H_pad):] = 0
#         lr_img[0, :, :, :max(lpad_w-4,0)] = 0
#         lr_img[0, :, :, min(rpad_w+4,W_pad):] = 0

#         hr_img = self.forward(lr_img, False)
#         return hr_img[:, :, :self.upscale*H, :self.upscale*W]

        lr_img = torch.stack((lr_img[0, :, :H_pad, :W_pad], lr_img[0, :, :H_pad, -W_pad:], lr_img[0, :, -H_pad:, :W_pad], lr_img[0, :, -H_pad:, -W_pad:]))
        
        # Upscale the image.
        results = []
        for _lr_img in lr_img:
            results.append(self.forward(_lr_img.unsqueeze(0), False))

        hr_img = torch.zeros(1, 3, self.upscale*H, self.upscale*W, device=lr_img.device)
        mask = torch.zeros_like(hr_img)
    
        upscaled_H, upscaled_W = self.upscale*H_pad, self.upscale*W_pad
        if trim_boundary:
            trim_size = 6 * self.upscale
            upscaled_H -= trim_size
            upscaled_W -= trim_size
        
        hr_img[0, :, :upscaled_H, :upscaled_W] += results[0][0, :, :upscaled_H, :upscaled_W]
        hr_img[0, :, :upscaled_H, -upscaled_W:] += results[1][0, :, :upscaled_H, -upscaled_W:]
        hr_img[0, :, -upscaled_H:, :upscaled_W] += results[2][0, :, -upscaled_H:, :upscaled_W]
        hr_img[0, :, -upscaled_H:, -upscaled_W:] += results[3][0, :, -upscaled_H:, -upscaled_W:]
        
        mask[0, :, :upscaled_H, :upscaled_W] += 1
        mask[0, :, :upscaled_H, -upscaled_W:] += 1
        mask[0, :, -upscaled_H:, :upscaled_W] += 1
        mask[0, :, -upscaled_H:, -upscaled_W:] += 1
        
        return hr_img / mask