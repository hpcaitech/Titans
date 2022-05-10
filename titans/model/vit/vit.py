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

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 depth: int = 12,
                 num_heads: int = 12,
                 hidden_size: int = 768,
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
                                  embedding_dim=hidden_size,
                                  dropout=dropout,
                                  dtype=dtype,
                                  init_method=init_method)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            ViTBlock(
                hidden_size=hidden_size,
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

        self.norm = col_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)

        self.head = ViTHead(hidden_size=hidden_size,
                            num_classes=num_classes,
                            representation_size=representation_size,
                            dtype=dtype,
                            bias=bias,
                            init_method=init_method)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(self.norm(x))
        return x


def _create_vit_model(**model_kwargs):
    model = VisionTransformer(**model_kwargs)
    return model


def vit_lite_depth7_patch4_32(**kwargs):
    model_kwargs = dict(img_size=32, patch_size=4, hidden_size=256, depth=7, num_heads=4, mlp_ratio=2, num_classes=10, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_tiny_patch4_32(**kwargs):
    model_kwargs = dict(img_size=32, patch_size=4, hidden_size=512, depth=6, num_heads=8, mlp_ratio=1, num_classes=10, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_tiny_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, hidden_size=192, depth=12, num_heads=3, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_tiny_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, hidden_size=192, depth=12, num_heads=3, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_small_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, hidden_size=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_small_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, hidden_size=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_small_patch32_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=32, hidden_size=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_small_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, hidden_size=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_base_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_base_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_base_patch32_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=32, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_base_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_large_patch16_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=16, hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_large_patch16_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=16, hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_large_patch32_224(**kwargs):
    model_kwargs = dict(img_size=224, patch_size=32, hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)


def vit_large_patch32_384(**kwargs):
    model_kwargs = dict(img_size=384, patch_size=32, hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
    return _create_vit_model(**model_kwargs)
