from torch import dtype, nn
from typing import Callable
from colossalai import nn as col_nn


class TransformerMLP(nn.Module):

    def __init__(self,
                 dim: int,
                 mlp_ratio: float,
                 activation: Callable,
                 dropout: float,
                 dtype: dtype = None,
                 bias: bool = True):
        super().__init__()
        intermediate_dim = int(dim * mlp_ratio)
        self.dense_1 = col_nn.Linear(dim, intermediate_dim, dtype=dtype, bias=bias)
        self.activation = activation
        self.dense_2 = col_nn.Linear(intermediate_dim, dim, dtype=dtype, bias=bias)
        self.dropout = col_nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x