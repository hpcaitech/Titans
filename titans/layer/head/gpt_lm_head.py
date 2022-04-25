from torch import dtype, nn

from colossalai import nn as col_nn


class GPTLMHead(nn.Module):

    def __init__(self,
                 dim: int,
                 vocab_size: int,
                 word_embeeding_weight: nn.Parameter = None,
                 bias: bool = False,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.dense = col_nn.Classifier(dim, vocab_size, word_embeeding_weight, bias=bias, dtype=dtype)

    @property
    def weight(self):
        return self.dense.weight

    def forward(self, x):
        x = self.dense(x)
        return x
