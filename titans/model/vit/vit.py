import math
from typing import Callable

import torch
from colossalai import nn as col_nn
from colossalai.nn.layer.utils import CheckpointModule
from torch import dtype, nn

from titans.layer.embedding import ViTEmbedding
from titans.layer.head import ViTHead
from titans.layer.block import ViTBlock
from titans.decorator import no_support

__all__ = [
    'VisionTransformer',
    'vit_lite_depth7_patch4_32',
    'vit_tiny_patch4_32',
    'vit_tiny_patch16_224',
    'vit_tiny_patch16_384',
    'vit_small_patch16_224',
    'vit_small_patch16_384',
    'vit_small_patch32_224',
    'vit_small_patch32_384',
    'vit_base_patch16_224',
    'vit_base_patch16_384',
    'vit_base_patch32_224',
    'vit_base_patch32_384',
    'vit_large_patch16_224',
    'vit_large_patch16_384',
    'vit_large_patch32_224',
    'vit_large_patch32_384',
]


@no_support(['sp', 'moe'])
class VisionTransformer(nn.Module):
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.

    Args:
        img_size(int): The size of images, defaults to 224.
        patch_size(int): The size of patches, defaults to 16.
        in_chans(int): The size of input channels, defaults to 3.
        num_classes(int): The number of target classes, defaults to 1000.
        depth(int): The number of transformer layers, defaults to 12.
        num_heads(int): The number of heads in transformer blocks, defaults to 12.
        dim(int): Hidden size of the transformer blocks, defaults to 768.
        mlp_ratio(int): The ratio used in mlp layer, defaults to 4.
        attention_dropout(float): The ratio used to construct attention dropout modules, which indicates the percentage of parameters should be casted to zero, defaults to 0.1.
        dropout(float): The ratio used to construct dropout modules, which indicates the percentage of parameters should be casted to zero, defaults to 0.1.
        drop_path(float): The ratio used to construct drop_path modules, which indicates the percentage of branches should be casted to zero, defaults to 0..
        layernorm_epsilon(float): The argument used to construct layernorm modules, defaults to 1e-6.
        activation(Callable): The activation function used in model, defaults to nn.functional.gelu.
        representation_size(int): The size of representation in head layer, defaults to None.
        dtype (:class:`torch.dtype`): The dtype of parameters, defaults to None.
        bias (bool): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        checkpoint(bool): If set to "True", checkpoint feature will be activated to save memory, defaults to ``False``.
        init_method(str): The initializing method used in layers, defaults to `torch`.
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 depth: int = 12,
                 num_heads: int = 12,
                 dim: int = 768,
                 mlp_ratio: int = 4,
                 attention_dropout: float = 0.,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 activation: Callable = nn.functional.gelu,
                 representation_size: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch'):
        super().__init__()

        self.embed = ViTEmbedding(img_size=img_size,
                                  patch_size=patch_size,
                                  in_chans=in_chans,
                                  embedding_dim=dim,
                                  dropout=dropout,
                                  dtype=dtype,
                                  init_method=init_method)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                drop_path=dpr[i],
                activation=activation,
                dtype=dtype,
                bias=bias,
                checkpoint=checkpoint,
                init_method=init_method,
            ) for i in range(depth)
        ])

        self.norm = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)

        self.head = ViTHead(dim=dim,
                            num_classes=num_classes,
                            representation_size=representation_size,
                            dtype=dtype,
                            bias=bias,
                            init_method=init_method)

    def forward(self, x):
        # the size of x is (BATCH_SIZE, IN_CHAN, IMAGE_SIZE, IMAGE_SIZE)
        x = self.embed(x)
        # the size of x after embed layer is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        for block in self.blocks:
            x = block(x)
        # the size of x after block is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        x = self.head(self.norm(x))
        # the size of x is (BATCH_SIZE, NUM_CLASSES)
        return x


def _create_vit_model(**model_kwargs):
    model = VisionTransformer(**model_kwargs)
    return model


def vit_lite_depth7_patch4_32(**kwargs):
    model_kwargs = dict(img_size=32, patch_size=4, dim=256, depth=7, num_heads=4, mlp_ratio=2, num_classes=10, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_tiny_patch4_32(**kwargs):
    model_kwargs = dict(img_size=32, patch_size=4, dim=512, depth=6, num_heads=8, mlp_ratio=1, num_classes=10, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_tiny_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, dim=192, depth=12, num_heads=3, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_tiny_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, dim=192, depth=12, num_heads=3, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_small_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_small_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_small_patch32_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=32, dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_small_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_base_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_base_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_base_patch32_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=32, dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_base_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_large_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_large_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_large_patch32_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=32, dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_large_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)
