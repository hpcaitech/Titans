from typing import Callable

from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import init_rules


class ViTMLP(nn.Module):

    def __init__(self,
                 dim: int,
                 mlp_ratio: int,
                 activation: Callable,
                 dropout: float,
                 dtype: dtype = None,
                 bias: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        self.dense_1 = col_nn.Linear(dim,
                                     mlp_ratio * dim,
                                     dtype=dtype,
                                     bias=bias,
                                     **init_rules[init_method]['transformer'])
        self.activation = activation
        self.dropout_1 = col_nn.Dropout(dropout)
        self.dense_2 = col_nn.Linear(mlp_ratio * dim,
                                     dim,
                                     dtype=dtype,
                                     bias=bias,
                                     **init_rules[init_method]['transformer'])
        self.dropout_2 = col_nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        return x
