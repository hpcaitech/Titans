import math

import torch
from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import _init_rules


class ViTSelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 attention_dropout: float,
                 dropout: float,
                 bias: bool = True,
                 dtype: dtype = None,
                 init_method: str = 'torch'):
        super().__init__()
        self.attention_head_size = dim // num_heads
        self.query_key_value = col_nn.Linear(dim,
                                             3 * dim,
                                             dtype=dtype,
                                             bias=bias,
                                             **_init_rules[init_method]['transformer'])
        self.attention_dropout = col_nn.Dropout(attention_dropout)
        self.dense = col_nn.Linear(dim, dim, dtype=dtype, bias=True, **_init_rules[init_method]['transformer'])
        self.dropout = col_nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        qkv = self.query_key_value(x)
        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = all_head_size // self.attention_head_size
        new_qkv_shape = qkv.shape[:-1] + \
            (num_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        qkv = qkv.permute((0, 2, 1, 3))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / math.sqrt(self.attention_head_size)
        x = self.softmax(x)
        x = self.attention_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size,)
        x = x.reshape(new_context_layer_shape)

        x = self.dense(x)
        x = self.dropout(x)

        return x
