from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(torch.arange(position)[:, None], torch.arange(d_model)[None, :], d_model)
    
    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[None, ...]
    return pos_encoding.float()


def create_padding_mask(seq):
    if len(seq.shape) < 3:  # Tokenized version
        seq = (seq == 0).float()
    elif seq.shape[-1] > 1:  # Continuous version (look at last bit)
        seq = (seq[..., -1] == 1).float()
    
    # Add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, None, None, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    # Create a lower triangular mask and invert it to get an upper triangular mask with no diagonal.
    mask = 1 - torch.tril(torch.ones((size, size), dtype=torch.float32))
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
    
    # Scale matmul_qk
    dk = float(k.shape[-1])
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    
    # Add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # Softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)
    
    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    
    return output, attention_weights



