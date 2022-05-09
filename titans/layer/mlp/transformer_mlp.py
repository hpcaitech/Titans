import torch.nn as nn
import torch.nn.functional as F

from colossalai import nn as col_nn
from typing import Callable
from torch import Tensor
from torch import dtype


class TransformerMLP(nn.Module):
    """
    The MLP module in the Transformer Architecture.

    Args:
        hidden_size (int): the dimension of the linear layer.
        mlp_ratio (int): the multiplication factor of the linear dimension, default is 4.
        act_func (Callable): the activation function, default is None which will use GeLU.
        dropout_prob (float): the probability of dropout, default is 0.
        dtype (torch.dtype): the data type for model parameters, default is None.
        bias (bool): whether the linear layers have bias, default is True.
    """

    def __init__(self,
                 hidden_size: int,
                 mlp_ratio: int = 4,
                 act_func: Callable = None,
                 dropout_prob: float = 0.0,
                 dtype: dtype = None,
                 bias: bool = True):
        super().__init__()
        intermediate_dim = int(hidden_size * mlp_ratio)

        # int linear layers
        self.linear_1 = col_nn.Linear(hidden_size, intermediate_dim, dtype=dtype, bias=bias)
        self.linear_2 = col_nn.Linear(intermediate_dim, hidden_size, dtype=dtype, bias=bias)

        # int activation function
        if act_func:
            self.act_func = act_func
        else:
            self.act_func = F.gelu

        # init dropout
        if dropout_prob > 0:
            self.dropout = col_nn.Dropout(dropout_prob)
        else:
            self.dropout = None

    def forward(self, x: Tensor) -> Tensor:
        intermediate_activate = self.linear_1(x)
        intermediate_activate = self.act_func(intermediate_activate)

        output = self.linear_2(intermediate_activate)

        if self.dropout:
            output = self.dropout(output)

        return output
