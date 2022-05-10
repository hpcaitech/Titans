from torch import nn
from colossalai import nn as col_nn

from titans.layer.attention import TransformerMultiHeadAttention
from .utils import get_clones


class TransformerDecoderLayer(nn.Module):

    def __init__(self, hidden_size, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.selfAttn = TransformerMultiHeadAttention(hidden_size, dim_feedforward, nhead, dropout)

        self.linear_1 = col_nn.Linear(hidden_size, dim_feedforward)
        self.linear_2 = col_nn.Linear(dim_feedforward, hidden_size)
        self.norm_1 = col_nn.LayerNorm(hidden_size)
        self.norm_2 = col_nn.LayerNorm(hidden_size)
        self.norm_3 = col_nn.LayerNorm(hidden_size)
        self.dropout_1 = col_nn.Dropout(dropout)
        self.dropout_2 = col_nn.Dropout(dropout)
        self.dropout_3 = col_nn.Dropout(dropout)
        self.dropout_4 = col_nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos, query_pos):
        tgt = tgt.transpose(0, 1)
        query_pos = query_pos.transpose(0, 1)
        pos = pos.transpose(0, 1)

        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.selfAttn(q, k, tgt)

        tgt = tgt + self.dropout_1(tgt2)
        tgt = self.norm_1(tgt)
        tgt2 = self.selfAttn(q, self.with_pos_embed(memory, pos), memory)
        tgt = tgt + self.dropout_2(tgt2)
        tgt = self.norm_2(tgt)
        tgt2 = self.linear_2(self.dropout_3(F.relu(self.linear_1(tgt))))
        tgt = tgt + self.dropout_4(tgt2)
        tgt = self.norm_3(tgt)
        return tgt


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, pos, query_pos):
        intermediate = []

        for layer in self.layers:
            tgt = layer(tgt, memory, pos=pos, query_pos=query_pos).transpose(0, 1)

            if self.return_intermediate:
                intermediate.append(self.norm(tgt))

        return torch.stack(intermediate)
