import math

import torch
from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import init_rules
from titans.decorator import no_support
# This part need to work together with the col_nn.Linear (row, col) in order to better parallelize.

@no_support(['sp'])
class DeTrCrossAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 attention_dropout: float,
                 dropout: float,
                 bias: bool = True,
                 dtype: dtype = None,
                 init_method: str = 'torch'):
        super().__init__()
        self.attention_head_size = hidden_size // num_heads
        self.query = col_nn.Linear1D_Col(hidden_size,
                                    hidden_size,
                                    dtype=dtype,
                                    bias=bias,
                                    )
        self.key_value = col_nn.Linear1D_Col(hidden_size,
                                        2 * hidden_size,
                                        dtype=dtype,
                                        bias=bias,
                                        )
        self.attention_dropout = col_nn.Dropout(attention_dropout)
        self.dense = col_nn.Linear1D_Row(hidden_size, hidden_size, dtype=dtype, bias=True)
        self.dropout = col_nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, memory):
        q = self.query(x)
        kv = self.key_value(memory)
        all_head_size = kv.shape[-1] // 2
        num_attention_heads = all_head_size // self.attention_head_size

        new_q_shape = q.shape[:-1] + (num_attention_heads, self.attention_head_size)
        q = q.view(new_q_shape)
        q = q.permute((0, 2, 1, 3))
        q = q.permute((2, 3, 0, 1)) # ?

        new_kv_shape = kv.shape[:-1] + (num_attention_heads, 2 * self.attention_head_size)
        kv = kv.view(new_kv_shape)
        kv = kv.permute((0, 2, 1, 3))
        k, v = torch.chunk(kv, 2, dim=-1)
        k = k.permute((2, 3, 0, 1)) # ?
        v = v.permute((2, 3, 0, 1)) # ?

        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / math.sqrt(self.attention_head_size)
        x = self.softmax(x)
        x = self.attention_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size,)
        x = x.reshape(new_context_layer_shape)
        x = x.transpose(0, 1)

        x = self.dense(x)
        x = self.dropout(x)

        return x
