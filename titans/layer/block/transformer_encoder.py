from torch import nn
from colossalai import nn as col_nn

from titans.layer.attention import TransformerMultiHeadAttention
from titans.layer.mlp import TransformerMLP
from .utils import get_clones


class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.selfAttn = TransformerMultiHeadAttention(hidden_size, dim_feedforward, nhead, dropout)
        self.feedForward = TransformerMLP(hidden_size, dim_feedforward, dropout)

        self.norm_1 = col_nn.LayerNorm(hidden_size)
        self.norm_2 = col_nn.LayerNorm(hidden_size)
        self.dropout_1 = col_nn.Dropout(dropout)
        self.dropout_2 = col_nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.selfAttn(x1, x1, x1))
        x2 = self.norm_2(x)
        out = x + self.dropout_2(self.feedForward(x2))
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src if pos is None else (src + pos)
        output = output.transpose(0, 1)

        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output
