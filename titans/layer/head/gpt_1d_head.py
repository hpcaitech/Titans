import torch
from torch import nn as nn, Tensor, distributed as dist
from torch.nn import functional as F
import torch.nn.init as init
from colossalai.context import ParallelMode
from colossalai.nn.layer.base_layer import ParallelLayer
from torch.nn.parameter import Parameter

from colossalai.nn.layer.parallel_1d._utils import reduce_grad
from colossalai.nn.layer.parallel_1d.layers import Linear1D_Row
from titans.layer.embedding import VocabParallelEmbedding1D, HiddenParallelEmbedding1D


class VocabParallelGPTLMHead1D(ParallelLayer):
    """
    Language model head that shares the same parameters with the embedding matrix.
    """

    def __init__(self, embed=None, vocab_size=None, dtype=None, embed_dim=None):
        super().__init__()
        if embed is not None:
            self.head = embed
        else:
            self.head = VocabParallelEmbedding1D(vocab_size, embed_dim, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = reduce_grad(x, ParallelMode.PARALLEL_1D)
        x = F.linear(x, self.head.weight)
        return x


class HiddenParallelGPTLMHead1D(ParallelLayer):
    """
    Language model head that shares the same parameters with the embedding matrix.
    """

    def __init__(
        self,
        embed=None,
        embed_dim=None,
        vocab_size=None,
        dtype=None,
    ):
        super().__init__()
        if embed is not None:
            self.head = embed
            self.synced_embed = True
        else:
            # self.embedding = HiddenParallelEmbedding1D(vocab_size, hidden_size, dtype, padding_idx)
            # (hidden_size/q, vocab_size)
            self.synced_embed = False
            self.head = Linear1D_Row(in_features=embed_dim,
                                     out_features=vocab_size,
                                     bias=False,
                                     dtype=dtype,
                                     parallel_input=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.synced_embed:
            x = F.linear(x, self.head.weight)
        else:
            x = self.head(x)

        return x
