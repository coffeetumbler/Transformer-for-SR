import numpy as np
import copy

import torch
import torch.nn as nn



# Make clones of a layer.
def clone_layer(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Partition windows of splited heads.
def partition_window(x, window_size, nh, nw):
    """
    <input>
        x : (n_batch, n_head, H, W, C)
        window_size : (int)
        nh : (int) num_window_height
        nw : (int) num_window_width
    
    <return>
        windows : (n_batch, n_head, num_window_height, num_window_width, window_size^2, C)
    """
    n_batch, n_head, H, W, C = x.shape
    x = x.view(n_batch, n_head, nh, window_size, nw, window_size, C)
    windows = x.transpose(3, 4).contiguous().view(n_batch, n_head, nh, nw, window_size*window_size, C)
    return windows


# Merge windows of splited heads.
def merge_window(windows, window_size):
    """
    <input>
        windows : (n_batch, n_head, num_window_height, num_window_width, window_size^2, C)
        window_size : (int)
    
    <return>
        x : (n_batch, n_head, H, W, C)
    """
    n_batch, n_head, nh, nw, N, C = windows.shape
    windows = windows.view(n_batch, n_head, nh, nw, window_size, window_size, C)
    windows = windows.transpose(3, 4).contiguous()
    return windows.view(n_batch, n_head, nh*window_size, nw*window_size, C)


# 4-class cyclic shifting
def cyclic_shift(x, shift_size):
    """
    <input>
        x : (n_batch, n_head, H, W, C)
        shift_size : (int)
    """
    n_batch, n_head, H, W, C = x.shape
    x_ = x.view(n_batch, 4, n_head // 4, H, W, C).transpose(0, 1).contiguous()
    x_1 = torch.roll(x_[1], shifts=shift_size, dims=-2)
    x_2 = torch.roll(x_[2], shifts=shift_size, dims=-3)
    x_3 = torch.roll(x_[3], shifts=(shift_size, shift_size), dims=(-2, -3))
    return torch.stack([x_[0], x_1, x_2, x_3]).transpose(0, 1).contiguous().view(n_batch, n_head, H, W, C)


# Make masking matrix for 4-class split heads.
def masking_matrix(n_head, H, W, window_size, shift_size, device='cpu'):
    """
    <input>
        n_head, H, W, window_size, shift_size : (int)
        device : torch.device
        
    <return>
        masking_heads : (1, n_head, num_window_height, num_window_width, window_size^2, window_size^2)
    """
    masking_heads = torch.zeros(4, H, W, dtype=int, device=device)
    masking_heads[[1,3], :, :shift_size] = 1
    masking_heads[[2,3], :shift_size] += 2

    masking_heads = partition_window(masking_heads.unsqueeze(0).unsqueeze(-1), window_size, H // window_size, W // window_size)
    masking_heads = masking_heads - masking_heads.transpose(-1, -2)
    masking_heads = masking_heads != 0
    
    return masking_heads.repeat_interleave(n_head // 4, dim=1)


# Indices for 2D relative position bias for windows
def relative_position_index(window_size):
    """
    <input>
        window_size : (int)
        
    <return>
        relative_coord : (window_size^4, )
    """
    axis_size = window_size * 2 - 1  # Number of possible positions along an axis
    
    # Coordinate indices along each axis
    coord_x = np.repeat(np.arange(window_size) * axis_size, window_size)
    coord_y = np.tile(np.arange(window_size), window_size)
    
    # Relative coordinate indices along each axis
    relative_x = coord_x[:, np.newaxis] - coord_x
    relative_y = coord_y[:, np.newaxis] - coord_y
    
    # Relative coordinate indices in 2D window
    relative_coord = relative_x + relative_y
    relative_coord += relative_coord[-1, 0]
    
    return relative_coord.flatten()