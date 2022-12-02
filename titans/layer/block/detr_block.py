import math
from typing import Callable

import torch
from colossalai import nn as col_nn
from colossalai.nn.layer.utils import CheckpointModule
from torch import dtype, nn

from titans.layer.attention import DeTrAttention
from titans.layer.mlp import ViTMLP
from titans.decorator import support_tp_pp_only


@support_tp_pp_only()
class DeTrEncoder(CheckpointModule):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: int,
                 activation: Callable,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch'):
        super().__init__(checkpoint)
        self.norm1 = col_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)
        self.attn = DeTrAttention(hidden_size=hidden_size,
                                     num_heads=num_heads,
                                     attention_dropout=attention_dropout,
                                     dropout=dropout,
                                     bias=bias,
                                     dtype=dtype,
                                     init_method=init_method)
        self.drop_path = col_nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = col_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)
        self.mlp = ViTMLP(hidden_size=hidden_size,
                          mlp_ratio=mlp_ratio,
                          activation=activation,
                          dropout=dropout,
                          dtype=dtype,
                          bias=bias,
                          init_method=init_method)

    def _forward(self, x, attn_mask=None, key_padding_mask=None):
        # input dimension [b,s,h]
        x = x.transpose(0,1)
        x = x + self.drop_path(self.norm1(self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x.transpose(0,1)


@support_tp_pp_only()
class DeTrDecoder(CheckpointModule):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: int,
                 activation: Callable,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch'):
        super().__init__(checkpoint)
        self.norm1 = col_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)
        self.norm2 = col_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)
        self.norm3 = col_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)

        self.attn1 = DeTrAttention(hidden_size=hidden_size,
                                     num_heads=num_heads,
                                     attention_dropout=attention_dropout,
                                     dropout=dropout,
                                     bias=bias,
                                     dtype=dtype,
                                     init_method=init_method)

        self.attn2 = DeTrAttention(hidden_size=hidden_size,
                                     num_heads=num_heads,
                                     attention_dropout=attention_dropout,
                                     dropout=dropout,
                                     bias=bias,
                                     dtype=dtype,
                                     init_method=init_method)

        self.drop_path = col_nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.mlp = ViTMLP(hidden_size=hidden_size,
                          mlp_ratio=mlp_ratio,
                          activation=activation,
                          dropout=dropout,
                          dtype=dtype,
                          bias=bias,
                          init_method=init_method)

    def _forward(self, x, memory, self_attn_mask=None, self_attn_key_padding_mask=None, multihead_attn_mask=None, multihead_attn_key_padding_mask=None):
        # input dimension [b,s,h] [q,s,h]
        x = x.transpose(0,1)
        memory = memory.transpose(0,1)
        x = x + self.drop_path(self.norm1(self.attn1(x, x, x, attn_mask=self_attn_mask, key_padding_mask=self_attn_key_padding_mask)))
        x = x + self.drop_path(self.norm2(self.attn2(x, memory, memory, attn_mask=multihead_attn_mask, key_padding_mask=multihead_attn_key_padding_mask)))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x.transpose(0,1)
