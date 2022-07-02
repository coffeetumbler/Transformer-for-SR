import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

import utils.functions as functions



# Main transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, positional_encoding_layer, encoder_layer, n_layer):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = functions.clone_layer(encoder_layer, n_layer)
            
        self.positional_encoding = True if positional_encoding_layer is not None else False
        if self.positional_encoding:
            self.positional_encoding_layer = positional_encoding_layer
        
    def forward(self, x):
        """
        <input>
            x : (n_batch, H, W, d_embed)
        """
        if self.positional_encoding:
            out = self.positional_encoding_layer(x)
        else:
            out = x

        for layer in self.encoder_layers:
            out = layer(out)

        return out
    
    
# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, feed_forward_layer, norm_layer, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.norm_layers = functions.clone_layer(norm_layer, 2)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        for p in self.attention_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        for p in self.feed_forward_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        
    def forward(self, x):
        """
        <input>
            x : (n_batch, H, W, d_embed)
        """
        out1 = self.norm_layers[0](x)  # Layer norm first
        out1 = self.attention_layer(out1)
        out1 = self.dropout_layer(out1) + x
        
        out2 = self.norm_layers[1](out1)
        out2 = self.feed_forward_layer(out2)
        return self.dropout_layer(out2) + out1
    
    
    
# Multi-head attention layer
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_embed, n_head,
                 query_height, query_width, query_window_size,
                 key_height, key_width, key_window_size,
                 relative_position_embedding=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_head = n_head
        self.d_k = d_embed // n_head
        self.scale = 1 / np.sqrt(self.d_k)
        
        # Linear layers
        self.query_fc_layer = nn.Linear(d_embed, d_embed)
        self.key_value_fc_layers = nn.Linear(d_embed, 2*d_embed)
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
        
        # Masking matrix
        mask = functions.masking_matrix(n_head, query_height, query_width, query_window_size, self.query_config['shift_size'],
                                        key_height, key_width, key_window_size, self.key_config['shift_size'])
        self.register_buffer('mask', mask)
        
        self.relative_position_embedding = relative_position_embedding
        if relative_position_embedding:
            # Table of 2D relative position embedding
            qk_ratio = query_window_size // key_window_size
            self.relative_position_embedding_table = nn.Parameter(torch.zeros((2*query_window_size-qk_ratio)**2, n_head))
            trunc_normal_(self.relative_position_embedding_table, std=.02)
            
            # Set 2D relative position embedding index.
            self.relative_position_index = functions.relative_position_index(query_window_size, key_window_size)

    def forward(self, x, z):
        """
        <input>
            x : (n_batch, H, W, d_embed), input query
            z : (n_batch, _H, _W, d_embed), encoder output
            
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
        query = functions.cyclic_shift(query, self.query_config['shift_size'])
        key = functions.cyclic_shift(key, self.key_config['shift_size'])
        value = functions.cyclic_shift(value, self.key_config['shift_size'])

        # Partition windows.
        # Q, K, V : (n_batch, n_head, nh, nw, window_size^2, d_k)
        query = functions.partition_window(query, self.query_config['window_size'], self.query_config['nh'], self.query_config['nw'])
        key = functions.partition_window(key, self.key_config['window_size'], self.key_config['nh'], self.key_config['nw'])
        value = functions.partition_window(value, self.key_config['window_size'], self.key_config['nh'], self.key_config['nw'])
        
        # Compute attention score.
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        
        # Add relative position embedding.
        if self.relative_position_embedding:
            position_embedding = self.relative_position_embedding_table[self.relative_position_index].view(
                self.query_config['window_size_sq'], self.key_config['window_size_sq'], -1)
            position_embedding = position_embedding.permute(2, 0, 1).contiguous()[None, :, None, None, ...]
            scores = scores + position_embedding
        
        # Add masking matrix.
        scores.masked_fill_(self.mask, -1e9)

        # Compute attention probability and values.
        scores = self.softmax(scores)
        attention = torch.matmul(scores, value)
        
        # Merge windows.
        attention = functions.merge_window(attention, self.query_config['window_size'])  # (n_batch, n_head, H, W, d_k)
        
        # Shift features reversely.
        attention = functions.cyclic_shift(attention, -self.query_config['shift_size'])
        
        # Concatenate heads and output features.
        attention = attention.permute(0, 2, 3, 1, 4).contiguous().view(n_batch, H, W, -1)
        return self.output_fc_layer(attention)
    
    
# Multi-head SELF-attention layer
class MultiHeadSelfAttentionLayer(MultiHeadAttentionLayer):
    def __init__(self, d_embed, n_head, height, width, window_size, relative_position_embedding=True):
        super(MultiHeadSelfAttentionLayer, self).__init__(d_embed, n_head,
                                                          height, width, window_size,
                                                          height, width, window_size,
                                                          relative_position_embedding)

    def forward(self, x):
        """
        <input>
            x : (n_batch, H, W, d_embed)
            
        <output>
            attention : (n_batch, H, W, d_embed)
        """
        return super(MultiHeadSelfAttentionLayer, self).forward(x, x)

    
    
# Position-wise FF layer
class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_embed, d_ff, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.first_fc_layer = nn.Linear(d_embed, d_ff)
        self.second_fc_layer = nn.Linear(d_ff, d_embed)
        self.activation_layer = nn.GELU()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.first_fc_layer(x)
        out = self.dropout_layer(self.activation_layer(out))
        return self.second_fc_layer(out)
    
    
    
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
                            n_layer=12,
                            n_head=4,
                            d_ff=512,
                            n_patch=28,
                            window_size=4,
                            dropout=0.1):
    
    if positional_encoding == 'Sinusoidal' or positional_encoding == 'sinusoidal' or positional_encoding == 'sin':
        positional_encoding_layer = SinusoidalPositionalEncoding(d_embed, max_seq_len, dropout)
    elif positional_encoding == 'Absolute' or positional_encoding =='absolute' or positional_encoding == 'abs':
        positional_encoding_layer = AbsolutePositionEmbedding(d_embed, max_seq_len, dropout)
    elif positional_encoding == None or positional_encoding == 'None':
        positional_encoding_layer = None
    
    attention_layer = MultiHeadSelfAttentionLayer(d_embed, n_head, n_patch, n_patch, window_size, relative_position_embedding)
    feed_forward_layer = PositionWiseFeedForwardLayer(d_embed, d_ff, dropout)
    norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
    encoder_layer = EncoderLayer(attention_layer, feed_forward_layer, norm_layer, dropout)
    
    return TransformerEncoder(positional_encoding_layer, encoder_layer, n_layer)


#####################################################################################################


# Main transformer decoder
class TransformerDecoder(nn.Module):
    def __init__(self, positional_encoding_layer, decoder_layer, n_layer):
        super(TransformerDecoder, self).__init__()
        self.decoder_layers = functions.clone_layer(decoder_layer, n_layer)
            
        self.positional_encoding = True if positional_encoding_layer is not None else False
        if self.positional_encoding:
            self.positional_encoding_layer = positional_encoding_layer
        
    def forward(self, x, z):
        """
        <input>
            x : (n_batch, H, W, d_embed), input query
            z : (n_batch, _H, _W, d_embed), encoder output
        """
        if self.positional_encoding:
            out = self.positional_encoding_layer(x)
        else:
            out = x

        for layer in self.decoder_layers:
            out = layer(out, z)

        return out
    
    
    
# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, self_attention_layer, attention_layer, feed_forward_layer, norm_layer, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention_layer = self_attention_layer
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.norm_layers = functions.clone_layer(norm_layer, 3)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        for p in self.self_attention_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        for p in self.attention_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        for p in self.feed_forward_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        
    def forward(self, x, z):
        """
        <input>
            x : (n_batch, H, W, d_embed), input query
            z : (n_batch, _H, _W, d_embed), encoder output
        """
        # Self-attention module
        out1 = self.norm_layers[0](x)  # Layer norm first
        out1 = self.self_attention_layer(out1)
        out1 = self.dropout_layer(out1) + x
        
        # Attention module
        out2 = self.norm_layers[1](out1)
        out2 = self.attention_layer(out2, z)
        out2 = self.dropout_layer(out2) + out1
        
        out3 = self.norm_layers[2](out2)
        out3 = self.feed_forward_layer(out3)
        return self.dropout_layer(out3) + out2