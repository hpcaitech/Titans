import math

import torch
from torch import nn
from colossalai import nn as col_nn


class TransformerSelfAttention(nn.Module):

    def __init__(
        self,
        dropout,
    ):
        super(SelfAttention, self).__init__()
        self.dropout = col_nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = torch.softmax(scores, dim=2)
        return torch.bmm(self.dropout(self.attention_weights), values)


class TransformerMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_hiddens, num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = SelfAttention(dropout)
        self.W_q = col_nn.Linear(d_model, num_hiddens, bias=bias)
        self.W_k = col_nn.Linear(d_model, num_hiddens, bias=bias)
        self.W_v = col_nn.Linear(d_model, num_hiddens, bias=bias)
        self.W_o = col_nn.Linear(num_hiddens, d_model, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
