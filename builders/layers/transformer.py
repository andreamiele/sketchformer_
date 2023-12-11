import torch
import torch.nn as nn
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

class SelfAttnV1(nn.Module):
    def __init__(self, units=None):
        super(SelfAttnV1, self).__init__()
        self.units = units

    def build(self, input_shape):
        assert len(input_shape) == 3
        fdim = input_shape[-1]
        if self.units is None:
            self.units = fdim

        self.W = nn.Parameter(torch.normal(0, 1, size=(fdim, self.units)))
        self.b = nn.Parameter(torch.zeros(self.units))
        self.V = nn.Parameter(torch.rand(self.units, 1))

    def forward(self, x):
        ui = torch.tanh(torch.matmul(x, self.W) + self.b)  # [B, T, L]
        ai = F.softmax(torch.matmul(ui, self.V), dim=1)  # [B, T, 1]
        o = torch.sum(x * ai, dim=1)
        return o, ai

class SelfAttnV2(nn.Module):
    def __init__(self, units=None):
        super(SelfAttnV2, self).__init__()
        self.units = units

    def build(self, input_shape):
        assert len(input_shape) == 3
        fdim = input_shape[-1]
        if self.units:
            self.embedding_layer = nn.Linear(fdim, self.units)

        self.W = nn.Parameter(torch.normal(0, 1, size=(fdim, fdim)))
        self.b = nn.Parameter(torch.zeros(fdim))
        self.V = nn.Parameter(torch.rand(fdim, 1))

    def forward(self, x):
        ui = torch.tanh(torch.matmul(x, self.W) + self.b)  # [B, T, L]
        ai = F.softmax(torch.matmul(ui, self.V), dim=1)  # [B, T, 1]
        o = torch.sum(x * ai, dim=1)  # [B, T, L]
        if self.units:
            o = self.embedding_layer(o)
        return o, ai

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding=1000, rate=0.1,
                 use_continuous_input=False):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        if use_continuous_input:
            self.embedding = nn.Linear(input_vocab_size, d_model)
        else:
            self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, training, mask):
        seq_len = x.size(1)
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, ...]
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding=1000, rate=0.1,
                 use_continuous_input=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        if use_continuous_input:
            self.embedding = nn.Linear(target_vocab_size, d_model)
        else:
            self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = x.size(1)
        attention_weights = {}
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, ...]
        x = self.dropout(x)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        return x, attention_weights

class DenseExpander(nn.Module):
    def __init__(self, seq_len, feat_dim_out=0):
        super(DenseExpander, self).__init__()
        self.seq_len = seq_len
        self.feat_dim_out = feat_dim_out

    def forward(self, x):
        if self.feat_dim_out:
            x = self.embedding_layer(x)
        x = x.unsqueeze(2)
        x = self.expand_layer(x)
        x = x.permute(0, 2, 1)
        return x
