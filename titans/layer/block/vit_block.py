import math
from typing import Callable

import torch
from colossalai import nn as col_nn
from colossalai.nn.layer.utils import CheckpointModule
from torch import dtype, nn

from titans.layer.attention import ViTSelfAttention
from titans.layer.mlp import ViTMLP
from titans.decorator import support_tp_pp_only


@support_tp_pp_only()
class ViTBlock(CheckpointModule):

    def __init__(self,
                 dim: int,
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
        self.norm1 = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.attn = ViTSelfAttention(dim=dim,
                                     num_heads=num_heads,
                                     attention_dropout=attention_dropout,
                                     dropout=dropout,
                                     bias=bias,
                                     dtype=dtype,
                                     init_method=init_method)
        self.drop_path = col_nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.mlp = ViTMLP(dim=dim,
                          mlp_ratio=mlp_ratio,
                          activation=activation,
                          dropout=dropout,
                          dtype=dtype,
                          bias=bias,
                          init_method=init_method)

    def _forward(self, x):
        # the size of x is (BATCH_SZIE, SEQ_LEN, HIDDEN_SIZE)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # the size of x after attn is (BATCH_SZIE, SEQ_LEN, HIDDEN_SIZE)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # the size of x after mlp is (BATCH_SZIE, SEQ_LEN, HIDDEN_SIZE)
        return x
