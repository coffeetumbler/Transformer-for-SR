import numpy as np

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_



# Patch partitioning for images and linear embedding layer
class EmbeddingLayer(nn.Module):
    def __init__(self, patch_size=2, d_embed=128, input_dim=3):
        super(EmbeddingLayer, self).__init__()
        self.linear_embedding = nn.Conv2d(input_dim, d_embed, kernel_size=patch_size, stride=patch_size)
        
        nn.init.xavier_uniform_(self.linear_embedding.weight)
#         trunc_normal_(self.linear_embedding.weight, std=.02)
        nn.init.zeros_(self.linear_embedding.bias)
        
    def forward(self, img):
        """
        <input>
            img : (n_batch, input_dim, img_height, img_width)
            
        <output>
            x : (n_batch, H, W, d_embed)
        """
        return self.linear_embedding(img).permute(0, 2, 3, 1).contiguous()
    
    
    
# Patch merging and linear unembedding layer
class UnembeddingLayer(nn.Module):
    def __init__(self, patch_size=2, d_embed=128, input_dim=3):
        super(UnembeddingLayer, self).__init__()
        self.d_embed = d_embed
        self.patch_size = patch_size
        self.linear_unembedding = nn.Linear(input_dim, d_embed*(patch_size**2))
        
        nn.init.xavier_uniform_(self.linear_unembedding.weight)
#         trunc_normal_(self.linear_unembedding.weight, std=.02)
        nn.init.zeros_(self.linear_unembedding.bias)
        
    def forward(self, x):
        """
        <input>
            img : (n_batch, height, width, input_dim)
            
        <output>
            x : (n_batch, d_embed, patch_size*height, patch_size*width)
        """
        n_batch, H, W, _ = x.shape
        x = self.linear_unembedding(x).view(n_batch, H, W, self.d_embed, self.patch_size, self.patch_size).permute(0, 3, 1, 4, 2, 5)
        return x.contiguous().view(n_batch, self.d_embed, H*self.patch_size, W*self.patch_size)
    
    
    
# Reconstruction block for image restoration
class ReconstructionBlock(nn.Module):
    def __init__(self, patch_size=2, d_embed=128, hidden_dim=128*4, dropout=0.1):
        super(ReconstructionBlock, self).__init__()
        self.patch_size = patch_size
        
        self.fc_layer1 = nn.Linear(d_embed, hidden_dim)
        self.activation_layer = nn.GELU()
        self.fc_layer2 = nn.Linear(hidden_dim, 3*patch_size*patch_size)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        nn.init.xavier_uniform_(self.fc_layer1.weight)
#         trunc_normal_(self.fc_layer1.weight, std=.02)
        nn.init.zeros_(self.fc_layer1.bias)
        nn.init.xavier_uniform_(self.fc_layer2.weight)
#         trunc_normal_(self.fc_layer2.weight, std=.02)
        nn.init.zeros_(self.fc_layer2.bias)
        
    def forward(self, x):
        """
        <input>
            x : (n_batch, H, W, d_embed)
            
        <output>
            img : (n_batch, 3, img_height, img_width)
        """
        n_batch, H, W, _ = x.shape
        
        x = self.fc_layer1(x)
        x = self.dropout_layer(self.activation_layer(x))
        x = self.fc_layer2(x).view(n_batch, H, W, 3, self.patch_size, self.patch_size).permute(0, 3, 1, 4, 2, 5)
        return x.contiguous().view(n_batch, 3, H*self.patch_size, W*self.patch_size)
    
    
    
# Upsampling layer
class UpsamplingLayer(nn.Module):
    def __init__(self, upscale=2, d_embed=128):
        super(UpsamplingLayer, self).__init__()
        self.upscale = upscale
        self.d_embed = d_embed
        self.linear_layer = nn.Linear(d_embed, d_embed*upscale*upscale)
        
        nn.init.xavier_uniform_(self.linear_layer.weight)
#         trunc_normal_(self.linear_layer.weight, std=.02)
        nn.init.zeros_(self.linear_layer.bias)
        
    def forward(self, x):
        """
        <input>
            x : (n_batch, H, W, d_embed)
            
        <output>
            x : (n_batch, H*upscale, W*upscale, d_embed)
        """
        n_batch, H, W, _ = x.shape
        
        x = self.linear_layer(x).view(n_batch, H, W, self.upscale, self.upscale, self.d_embed).transpose(2, 3)
        return x.contiguous().view(n_batch, H*self.upscale, W*self.upscale, self.d_embed)