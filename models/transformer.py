import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_, DropPath

import utils.functions as functions



# Main transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, positional_encoding_layer, encoder_layer, n_layer, mask):
        super(TransformerEncoder, self).__init__()
        self.n_layer = n_layer
        self.encoder_layers = functions.clone_layer(encoder_layer, n_layer)
        self.register_buffer('mask', mask)
        
        self.n_head = encoder_layer.attention_layer.n_head
        self.n_class = encoder_layer.attention_layer.n_class
        self.window_size = encoder_layer.attention_layer.query_config['window_size']
            
        self.positional_encoding = True if positional_encoding_layer is not None else False
        if self.positional_encoding:
            self.positional_encoding_layer = positional_encoding_layer
        
    def forward(self, x, load_mask=True, padding_mask=None):
        """
        <input>
            x : (n_batch, H, W, d_embed)
            load_mask : Use self.mask as mask input if True
        """
        if self.positional_encoding:
            x = self.positional_encoding_layer(x)

        if load_mask:
            mask = self.mask
        else:
            _, H, W, _ = x.shape
            mask = functions.masking_matrix(self.n_head, H, W, self.window_size, self.window_size//2, n_class=self.n_class).to(x.device)
            if padding_mask != None:
                mask = torch.logical_or(mask, padding_mask)
            
        for layer in self.encoder_layers:
            x = layer(x, mask)

        return x
    
#     def flops(self):
#         return self.n_layer * self.encoder_layers[0].flops()
    
    
# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, feed_forward_layer, norm_layer, dropout=0.1, path_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.norm_layers = functions.clone_layer(norm_layer, 2)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.drop_path = DropPath(path_dropout) if path_dropout > 0 else nn.Identity()
        
        for p in self.attention_layer.parameters():
            if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
                trunc_normal_(p, std=.02)
            else:
                nn.init.zeros_(p)
        for p in self.feed_forward_layer.parameters():
            if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
                trunc_normal_(p, std=.02)
            else:
                nn.init.zeros_(p)
        for layer in self.norm_layers:
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        
    def forward(self, x, mask):
        """
        <input>
            x : (n_batch, H, W, d_embed)
            mask : (1, n_head, nh, nw, query_window_size^2, key_window_size^2)
        """
        out = self.norm_layers[0](x)  # Layer norm first
        out = self.attention_layer(out, mask)
        x = self.drop_path(self.dropout_layer(out)) + x
        
        out = self.norm_layers[1](x)
        out = self.feed_forward_layer(out)
        return self.drop_path(self.dropout_layer(out)) + x
    
    
#     def flops(self):
#         flops = 0
#         N = self.attention_layer.query_height ** 2
#         d_embed = self.attention_layer.d_embed
#         # norm layers
#         flops += 2 * N * d_embed
#         # attention module
#         flops += self.attention_layer.flops()
#         # ff layer
#         flops += self.feed_forward_layer.flops(N)
#         return flops
    
    
    
# Multi-head attention layer
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_embed, key_d_embed, n_head,
                 query_height, query_width, query_window_size,
                 key_height, key_width, key_window_size,
                 n_class=4, relative_position_embedding=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_embed = d_embed
        self.key_d_embed = key_d_embed
        
        self.query_height = query_height
        self.key_height = key_height
        self.query_width = query_width
        self.key_width = key_width
        
        self.n_head = n_head
        self.n_class = n_class
        self.d_k = d_embed // n_head
        self.scale = 1 / np.sqrt(self.d_k)
        
        # Linear layers
        self.query_fc_layer = nn.Linear(d_embed, d_embed)
        self.key_value_fc_layers = nn.Linear(key_d_embed, 2*d_embed)
        self.output_fc_layer = nn.Linear(d_embed, d_embed)
        self.softmax = nn.Softmax(dim=-1)
        
        # Configs
        self.query_config = {'window_size' : query_window_size,
                             'nh' : query_height // query_window_size,
                             'nw' : query_width // query_window_size,
                             'shift_size' : query_window_size // 2,
                             'window_size_sq' : query_window_size * query_window_size}
        
        self.key_config = {'window_size' : key_window_size,
                           'nh' : key_height // key_window_size,
                           'nw' : key_width // key_window_size,
                           'shift_size' : key_window_size // 2,
                           'window_size_sq' : key_window_size * key_window_size}
        
#         # Masking matrix
#         mask = functions.masking_matrix(n_head, query_height, query_width, query_window_size, self.query_config['shift_size'],
#                                         key_height, key_width, key_window_size, self.key_config['shift_size'])
#         self.register_buffer('mask', mask)
        
        self.relative_position_embedding = relative_position_embedding
        if relative_position_embedding:
            # Table of 2D relative position embedding
            qk_ratio = query_window_size // key_window_size
            self.relative_position_embedding_table = nn.Parameter(torch.zeros((2*query_window_size-qk_ratio)**2, n_head))
            trunc_normal_(self.relative_position_embedding_table, std=.02)
            
            # Set 2D relative position embedding index.
            self.relative_position_index = functions.relative_position_index(query_window_size, key_window_size)

    def forward(self, x, z, mask):
        """
        <input>
            x : (n_batch, H, W, d_embed), input query
            z : (n_batch, _H, _W, d_embed), encoder output
            mask : (1, n_head, nh, nw, query_window_size^2, key_window_size^2)
            
        <output>
            attention : (n_batch, H, W, d_embed)
        """
        n_batch, H, W, _ = x.shape
        _, _H, _W, _ = z.shape
        
        # Apply linear layers and split heads.
        query = self.query_fc_layer(x).view(n_batch, H, W, self.n_head, self.d_k).contiguous().permute(0, 3, 1, 2, 4)
        combined_values = self.key_value_fc_layers(z).view(n_batch, _H, _W, 2, self.n_head, self.d_k).contiguous()
        combined_values = combined_values.permute(3, 0, 4, 1, 2, 5)
        
        # Q : (n_batch, n_head, H, W, d_k)
        # K, V : (n_batch, n_head, _H, _W, d_k)
        key, value = combined_values[0], combined_values[1]
        
        # Shift features in 4-class way.
        query = functions.cyclic_shift(query, self.query_config['shift_size'], self.n_class)
        key = functions.cyclic_shift(key, self.key_config['shift_size'], self.n_class)
        value = functions.cyclic_shift(value, self.key_config['shift_size'], self.n_class)

        # Partition windows.
        # Q, K, V : (n_batch, n_head, nh, nw, window_size^2, d_k)
#         query = functions.partition_window(query, self.query_config['window_size'], self.query_config['nh'], self.query_config['nw'])
        query = functions.partition_window(query, self.query_config['window_size'])
#         key = functions.partition_window(key, self.key_config['window_size'], self.key_config['nh'], self.key_config['nw'])
        key = functions.partition_window(key, self.key_config['window_size'])
#         value = functions.partition_window(value, self.key_config['window_size'], self.key_config['nh'], self.key_config['nw'])
        value = functions.partition_window(value, self.key_config['window_size'])
        
        # Compute attention score.
        x = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        
        # Add relative position embedding.
        if self.relative_position_embedding:
            position_embedding = self.relative_position_embedding_table[self.relative_position_index].view(
                self.query_config['window_size_sq'], self.key_config['window_size_sq'], -1)
            position_embedding = position_embedding.permute(2, 0, 1).contiguous()[None, :, None, None, ...]
            x = x + position_embedding
        
        # Add masking matrix.
        x.masked_fill_(mask, -1e9)

        # Compute attention probability and values.
        x = self.softmax(x)
        x = torch.matmul(x, value)
        
        # Merge windows.
        x = functions.merge_window(x, self.query_config['window_size'])  # (n_batch, n_head, H, W, d_k)
        
        # Shift features reversely.
        x = functions.cyclic_shift(x, -self.query_config['shift_size'], self.n_class)
        
        # Concatenate heads and output features.
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(n_batch, H, W, -1)
        return self.output_fc_layer(x)
    
#     def flops(self):
#         flops = 0
#         # query and output
#         flops += 2 * self.query_height * self.query_width * (self.d_embed ** 2)
#         # key and value
#         flops += 2 * self.key_height * self.key_width * (self.d_embed ** 2)
#         # window-wise attention
#         flops += 2 * self.query_config['window_size_sq'] * self.key_config['window_size_sq']\
#                    * self.query_config['nh'] * self.query_config['nw'] * self.d_embed
#         return flops
    
    
# Multi-head SELF-attention layer
class MultiHeadSelfAttentionLayer(MultiHeadAttentionLayer):
    def __init__(self, d_embed, n_head, height, width, window_size, n_class=4, relative_position_embedding=True):
        super(MultiHeadSelfAttentionLayer, self).__init__(d_embed, d_embed, n_head,
                                                          height, width, window_size,
                                                          height, width, window_size,
                                                          n_class, relative_position_embedding)

    def forward(self, x, mask):
        """
        <input>
            x : (n_batch, H, W, d_embed)
            mask : (1, n_head, nh, nw, window_size^2)
            
        <output>
            attention : (n_batch, H, W, d_embed)
        """
        return super(MultiHeadSelfAttentionLayer, self).forward(x, x, mask)

    
    
# Position-wise FF layer
class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_embed, d_ff, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.d_embed = d_embed
        self.d_ff = d_ff
        
        self.first_fc_layer = nn.Linear(d_embed, d_ff)
        self.second_fc_layer = nn.Linear(d_ff, d_embed)
        self.activation_layer = nn.GELU()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.first_fc_layer(x)
        x = self.dropout_layer(self.activation_layer(x))
        return self.second_fc_layer(x)
    
#     def flops(self, N):
#         return 2 * N * self.d_embed * self.d_ff



# Depth-wise convolutional FF layer
class ConvolutionalFeedForwardLayer(nn.Module):
    def __init__(self, d_embed, d_ff, dropout=0.1):
        super(ConvolutionalFeedForwardLayer, self).__init__()
        self.d_embed = d_embed
        self.d_ff = d_ff
        
        self.first_fc_layer = nn.Conv2d(d_embed, d_ff, kernel_size=1, stride=1, padding=0)
        self.second_fc_layer = nn.Conv2d(d_ff, d_embed, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv_layer = nn.Conv2d(d_ff, d_ff, kernel_size=3, stride=1, padding=1, groups=d_ff)
        self.activation_layer = nn.GELU()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        <input>
            x : (n_batch, H, W, d_embed)
        """
        x = self.first_fc_layer(x.permute(0, 3, 1, 2))
        x = self.dropout_layer(self.activation_layer(x))
        x = self.depthwise_conv_layer(x)
        x = self.dropout_layer(self.activation_layer(x))
        return self.second_fc_layer(x).permute(0, 2, 3, 1)
    
    
    
# Sinusoidal positional encoding
# Deprecated now
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len=512, dropout=0.1):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        denominators = torch.exp(torch.arange(0, d_embed, 2) * (np.log(0.0001) / d_embed)).unsqueeze(0)
        encoding_matrix = torch.matmul(positions, denominators)
        
        encoding = torch.empty(1, max_seq_len, d_embed)
        encoding[0, :, 0::2] = torch.sin(encoding_matrix)
        encoding[0, :, 1::2] = torch.cos(encoding_matrix[:, :(d_embed//2)])

        self.register_buffer('encoding', encoding)
        
    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_embed) == (*, max_seq_len, d_embed) (default)
        """
        return self.dropout_layer(x + self.encoding)
    
    
# Absolute position embedding
# Deprecated now
class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, d_embed, max_seq_len=512, dropout=0.1):
        super(AbsolutePositionEmbedding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_embed))
        trunc_normal_(self.embedding, std=.02)
        
    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_embed) == (*, max_seq_len, d_embed) (default)
        """
        return self.dropout_layer(x + self.embedding)
    
    

# Get a transformer encoder with its parameters.
def get_transformer_encoder(d_embed=128,
                            positional_encoding=None,
                            relative_position_embedding=True,
                            convolutional_ff=False,
                            n_layer=12,
                            n_head=4,
                            n_class=4,
                            d_ff=128*4,
                            n_patch=24,
                            window_size=4,
                            dropout=0,
                            path_dropout=0):
    
    if positional_encoding == 'Sinusoidal' or positional_encoding == 'sinusoidal' or positional_encoding == 'sin':
        positional_encoding_layer = SinusoidalPositionalEncoding(d_embed, max_seq_len, dropout)
    elif positional_encoding == 'Absolute' or positional_encoding =='absolute' or positional_encoding == 'abs':
        positional_encoding_layer = AbsolutePositionEmbedding(d_embed, max_seq_len, dropout)
    elif positional_encoding == None or positional_encoding == 'None':
        positional_encoding_layer = None
        
    # Masking matrix
    mask = functions.masking_matrix(n_head, n_patch, n_patch, window_size, window_size//2, n_class=n_class)
    
    attention_layer = MultiHeadSelfAttentionLayer(d_embed, n_head, n_patch, n_patch, window_size, n_class, relative_position_embedding)
    if convolutional_ff:
        feed_forward_layer = ConvolutionalFeedForwardLayer(d_embed, d_ff, dropout)
    else:
        feed_forward_layer = PositionWiseFeedForwardLayer(d_embed, d_ff, dropout)
    norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
    
    encoder_layer = EncoderLayer(attention_layer, feed_forward_layer, norm_layer, dropout, path_dropout)
    return TransformerEncoder(positional_encoding_layer, encoder_layer, n_layer, mask)


#####################################################################################################


# Main transformer decoder
class TransformerDecoder(nn.Module):
    def __init__(self, positional_encoding_layer, decoder_layer, n_layer, self_mask, cross_mask):
        super(TransformerDecoder, self).__init__()
        self.n_layer = n_layer
        self.decoder_layers = functions.clone_layer(decoder_layer, n_layer)
        self.register_buffer('self_mask', self_mask)
        self.register_buffer('cross_mask', cross_mask)
        
        self.n_head = decoder_layer.attention_layer.n_head
        self.n_class = decoder_layer.attention_layer.n_class
        self.query_window_size = decoder_layer.attention_layer.query_config['window_size']
        self.key_window_size = decoder_layer.attention_layer.key_config['window_size']
            
        self.positional_encoding = True if positional_encoding_layer is not None else False
        if self.positional_encoding:
            self.positional_encoding_layer = positional_encoding_layer
        
    def forward(self, x, z, load_mask=True, padding_mask=None):
        """
        <input>
            x : (n_batch, H, W, d_embed), input query
            z : (n_batch, _H, _W, d_embed), encoder output
            load_mask : Use self.mask as mask input if True
        """
        if self.positional_encoding:
            x = self.positional_encoding_layer(x)
        
        if load_mask:
            self_mask = self.self_mask
            cross_mask = self.cross_mask
        else:
            _, H, W, _ = x.shape
            _, _H, _W, _ = z.shape
            self_mask = functions.masking_matrix(self.n_head, H, W, self.query_window_size, self.query_window_size//2, n_class=self.n_class).to(x.device)
            cross_mask = functions.masking_matrix(self.n_head, H, W, self.query_window_size, self.query_window_size//2,
                                                  _H, _W, self.key_window_size, self.key_window_size//2, n_class=self.n_class).to(z.device)
            if padding_mask != None:
                self_mask = torch.logical_or(self_mask, padding_mask[0])
                cross_mask = torch.logical_or(cross_mask, padding_mask[1])
            
        for layer in self.decoder_layers:
            x = layer(x, z, self_mask, cross_mask)

        return x
    
    
    
# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, self_attention_layer, attention_layer, feed_forward_layer, norm_layer, key_norm_layer,
                 dropout=0.1, path_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention_layer = self_attention_layer
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.norm_layers = functions.clone_layer(norm_layer, 3)
        self.key_norm_layer = key_norm_layer
        
        self.dropout_layer = nn.Dropout(p=dropout)
        self.drop_path = DropPath(path_dropout) if path_dropout > 0 else nn.Identity()
        
        for p in self.self_attention_layer.parameters():
            if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
                trunc_normal_(p, std=.02)
            else:
                nn.init.zeros_(p)
        for p in self.attention_layer.parameters():
            if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
                trunc_normal_(p, std=.02)
            else:
                nn.init.zeros_(p)
        for p in self.feed_forward_layer.parameters():
            if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
                trunc_normal_(p, std=.02)
            else:
                nn.init.zeros_(p)
                
        for layer in self.norm_layers:
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.ones_(self.key_norm_layer.weight)
        nn.init.zeros_(self.key_norm_layer.bias)
        
    def forward(self, x, z, self_mask, cross_mask):
        """
        <input>
            x : (n_batch, H, W, d_embed), input query
            z : (n_batch, _H, _W, d_embed), encoder output
            mask : (1, n_head, nh, nw, query_window_size^2, key_window_size^2)
        """
        # Self-attention module
        out = self.norm_layers[0](x)  # Layer norm first
        out = self.self_attention_layer(out, self_mask)
        x = self.drop_path(self.dropout_layer(out)) + x
        
        # Cross-attention module
        out = self.norm_layers[1](x)
        out = self.attention_layer(out, self.key_norm_layer(z), cross_mask)  # Layer norm for z is applied.
        x = self.drop_path(self.dropout_layer(out)) + x
        
        out = self.norm_layers[2](x)
        out = self.feed_forward_layer(out)
        return self.drop_path(self.dropout_layer(out)) + x
    
    
    
# Get a transformer decoder with its parameters.
def get_transformer_decoder(d_embed=128,
                            key_d_embed=256,
                            positional_encoding=None,
                            relative_position_embedding=True,
                            convolutional_ff=False,
                            n_layer=12,
                            n_head=4,
                            n_class=4,
                            d_ff=128*4,
                            query_n_patch=48,
                            query_window_size=8,
                            key_n_patch=24,
                            key_window_size=4,
                            dropout=0,
                            path_dropout=0):
    
    if positional_encoding == 'Sinusoidal' or positional_encoding == 'sinusoidal' or positional_encoding == 'sin':
        positional_encoding_layer = SinusoidalPositionalEncoding(d_embed, max_seq_len, dropout)
    elif positional_encoding == 'Absolute' or positional_encoding =='absolute' or positional_encoding == 'abs':
        positional_encoding_layer = AbsolutePositionEmbedding(d_embed, max_seq_len, dropout)
    elif positional_encoding == None or positional_encoding == 'None':
        positional_encoding_layer = None
        
    # Masking matrix
    self_mask = functions.masking_matrix(n_head, query_n_patch, query_n_patch, query_window_size, query_window_size//2, n_class=n_class)
    cross_mask = functions.masking_matrix(n_head, query_n_patch, query_n_patch, query_window_size, query_window_size//2,
                                          key_n_patch, key_n_patch, key_window_size, key_window_size//2, n_class=n_class)
    
    self_attention_layer = MultiHeadSelfAttentionLayer(d_embed, n_head,
                                                       query_n_patch, query_n_patch, query_window_size,
                                                       n_class, relative_position_embedding)
    attention_layer = MultiHeadAttentionLayer(d_embed, key_d_embed, n_head,
                                              query_n_patch, query_n_patch, query_window_size,
                                              key_n_patch, key_n_patch, key_window_size,
                                              n_class, relative_position_embedding)
    
    if convolutional_ff:
        feed_forward_layer = ConvolutionalFeedForwardLayer(d_embed, d_ff, dropout)
    else:
        feed_forward_layer = PositionWiseFeedForwardLayer(d_embed, d_ff, dropout)
        
    norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
    key_norm_layer = nn.LayerNorm(key_d_embed, eps=1e-6)
    decoder_layer = DecoderLayer(self_attention_layer, attention_layer, feed_forward_layer,
                                 norm_layer, key_norm_layer, dropout, path_dropout)
    
    return TransformerDecoder(positional_encoding_layer, decoder_layer, n_layer, self_mask, cross_mask)
