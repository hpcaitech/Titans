import torch
from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import _init_rules


class ViTEmbedding(nn.Module):
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embedding_dim: int,
                 dropout: float,
                 dtype: dtype = None,
                 flatten: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        self.patch_embed = col_nn.PatchEmbedding(img_size,
                                                 patch_size,
                                                 in_chans,
                                                 embedding_dim,
                                                 dtype=dtype,
                                                 flatten=flatten,
                                                 **_init_rules[init_method]['embed'])
        self.dropout = col_nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.dropout(x)
        return x

