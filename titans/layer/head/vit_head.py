from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import _init_rules


class ViTHead(nn.Module):

    def __init__(self,
                 dim: int,
                 num_classes: int,
                 representation_size: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        if representation_size:
            self.representation = col_nn.Linear(dim,
                                                representation_size,
                                                bias=bias,
                                                dtype=dtype,
                                                **_init_rules[init_method]['head'])
        else:
            self.representation = None
            representation_size = dim

        self.dense = col_nn.Classifier(representation_size,
                                       num_classes,
                                       dtype=dtype,
                                       bias=bias,
                                       **_init_rules[init_method]['head'])

    def forward(self, x):
        x = x[:, 0]
        if self.representation is not None:
            x = self.representation(x)
        x = self.dense(x)
        return x
