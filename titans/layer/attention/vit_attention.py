import math

import torch
from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import init_rules
from titans.decorator import no_support


@no_support(['sp'])
class ViTSelfAttention(nn.Module):

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
        self.query_key_value = col_nn.Linear(hidden_size,
                                             3 * hidden_size,
                                             dtype=dtype,
                                             bias=bias,
                                             **init_rules[init_method]['transformer'])
        self.attention_dropout = col_nn.Dropout(attention_dropout)
        self.dense = col_nn.Linear(hidden_size, hidden_size, dtype=dtype, bias=True, **init_rules[init_method]['transformer'])
        self.dropout = col_nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # the size of x is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        # the size of qkv is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE*3)
        qkv = self.query_key_value(x)
        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = all_head_size // self.attention_head_size
        new_qkv_shape = qkv.shape[:-1] + \
            (num_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        qkv = qkv.permute((0, 2, 1, 3))
        # the size of q is (BATCH_SZIE, NUM_HEADS, SEQ_LEN, HIDDEN_SIZE//NUM_HEADS)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # the size of x is (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / math.sqrt(self.attention_head_size)
        x = self.softmax(x)
        x = self.attention_dropout(x)

        # the size of x after matmul is (BATCH_SZIE, NUM_HEADS, SEQ_LEN, HIDDEN_SIZE//NUM_HEADS)
        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size,)
        # the size of x after reshape is (BATCH_SZIE, SEQ_LEN, HIDDEN_SIZE)
        x = x.reshape(new_context_layer_shape)
        # the size of x after dense is (BATCH_SZIE, SEQ_LEN, HIDDEN_SIZE)
        x = self.dense(x)
        x = self.dropout(x)

        return x
