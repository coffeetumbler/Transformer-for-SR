import numpy as np

import torch
import torch.nn as nn



# Patch partitioning for images and linear embedding layer
class EmbeddingLayer(nn.Module):
    def __init__(self, patch_size=2, d_embed=128):
        super(EmbeddingLayer, self).__init__()
        self.linear_embedding = nn.Conv2d(3, d_embed, kernel_size=patch_size, stride=patch_size)
        
        nn.init.xavier_uniform_(self.linear_embedding.weight)
        nn.init.zeros_(self.linear_embedding.bias)
        
    def forward(self, img):
        """
        <input>
            img : (n_batch, 3, img_height, img_width)
            
        <output>
            x : (n_batch, H, W, d_embed)
        """
        return self.linear_embedding(img).permute(0, 2, 3, 1).contiguous()
    
    
    
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
        nn.init.zeros_(self.fc_layer1.bias)
        nn.init.xavier_uniform_(self.fc_layer2.weight)
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