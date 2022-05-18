from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import init_rules


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
                                                **init_rules[init_method]['head'])
        else:
            self.representation = None
            representation_size = dim

        self.dense = col_nn.Classifier(representation_size,
                                       num_classes,
                                       dtype=dtype,
                                       bias=bias,
                                       **init_rules[init_method]['head'])

    def forward(self, x):
        # the size of x is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        x = x[:, 0]
        # the size of x is (BATCH_SIZE, HIDDEN_SIZE)
        if self.representation is not None:
            x = self.representation(x)
            # the size of x after representation is (BATCH_SIZE, REPRESENTATION_SIZE)
        x = self.dense(x)
        # the size of x after dense is (BATCH_SIZE, NUM_CLASSES)
        return x
