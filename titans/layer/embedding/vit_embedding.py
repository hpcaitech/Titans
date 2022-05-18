import torch
from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import init_rules


class ViTEmbedding(nn.Module):
    """
    Construct the patch embeddings.

    Args:
        img_size(int): The size of images.
        patch_size(int): The size of patches.
        in_chans(int): The size of input channels.
        embedding_dim(int): The embedding size of patches.
        dropout(float): The ratio used to construct dropout modules, which indicates the percentage of parameters should be casted to zero.
        dtype (:class:`torch.dtype`): The dtype of parameters, defaults to None.
        flatten(bool): If set to ``False``, the patches will not be flatten, defaults to ``True``.
        init_method(str): The initializing method used in layers, defaults to `torch`.
    """

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
                                                 **init_rules[init_method]['embed'])
        self.dropout = col_nn.Dropout(dropout)

    def forward(self, x):
        # the size of x before embed is (BATCH_SIZE, IN_CHAN, IMAGE_SIZE, IMAGE_SIZE)
        # the size of x after embedding is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        x = self.patch_embed(x)
        x = self.dropout(x)
        return x
