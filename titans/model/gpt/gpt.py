import math
from typing import Callable

import torch
from colossalai import nn as col_nn
from colossalai.builder.pipeline import partition_uniform
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.layer.utils import CheckpointModule, divide
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
from colossalai.utils import get_current_device
from torch import dtype, nn

from titans.layer.embedding import GPTEmbedding
from titans.layer.head import GPTLMHead
from titans.layer.block import GPTBlock
from titans.loss.lm_loss import GPTLMLoss
from titans.decorator import no_support

__all__ = ['GPT', 'GPTLMLoss', 'gpt2_small', 'gpt2_medium', 'gpt2_large', 'gpt2_xl', 'gpt2_8B', 'gpt3']


@no_support(['sp', 'moe'])
class GPT(nn.Module):

    def __init__(self,
                 vocab_size: int = 50304,
                 max_position_embeddings: int = 1024,
                 dim: int = 768,
                 num_heads: int = 12,
                 depth: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 embedding_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 layernorm_epsilon: float = 1e-5,
                 activation: Callable = nn.functional.gelu,
                 padding_idx: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 fuse_scale_mask_softmax: bool = False,
                 checkpoint: bool = False,
                 activation_offload: bool = False) -> None:
        super().__init__()
        self.embed = GPTEmbedding(embedding_dim=dim,
                                  vocab_size=vocab_size,
                                  max_position_embeddings=max_position_embeddings,
                                  padding_idx=padding_idx,
                                  dropout=embedding_dropout,
                                  dtype=dtype)
        self.blocks = nn.ModuleList([
            GPTBlock(dim=dim,
                     num_heads=num_heads,
                     mlp_ratio=mlp_ratio,
                     activation=activation,
                     attention_dropout=attention_dropout,
                     dropout=dropout,
                     layernorm_epsilon=layernorm_epsilon,
                     dtype=dtype,
                     bias=bias,
                     apply_post_layernorm=apply_post_layernorm,
                     fuse_scale_mask_softmax=fuse_scale_mask_softmax,
                     checkpoint=checkpoint,
                     activation_offload=activation_offload) for _ in range(depth)
        ])

        self.norm = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)

        self.head = GPTLMHead(
            dim=dim,
            vocab_size=vocab_size,
            embedding_layer=self.embed,
            # word_embeeding_weight=self.embed.word_embedding_weight,
            dtype=dtype)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # Adapted from huggingface
        if attention_mask is not None:
            batch_size = input_ids.shape[0]
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = col_nn.partition_batch(attention_mask)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=x.dtype)    # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        for block in self.blocks:
            x, attention_mask = block(x, attention_mask)

        x = self.head(self.norm(x))

        return x


def _create_gpt_model(**model_kwargs):
    model = GPT(**model_kwargs)
    return model


def gpt2_small(**kwargs):
    model_kwargs = dict(dim=768, depth=12, num_heads=12, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_medium(**kwargs):
    model_kwargs = dict(dim=1024, depth=24, num_heads=8, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_large(**kwargs):
    model_kwargs = dict(dim=1536, depth=36, num_heads=12, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_xl(**kwargs):
    model_kwargs = dict(dim=1600, depth=48, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_2B(**kwargs):
    model_kwargs = dict(dim=2048, depth=40, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_3B(**kwargs):
    model_kwargs = dict(dim=2304, depth=48, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_4B(**kwargs):
    model_kwargs = dict(dim=2304, depth=64, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_6B(**kwargs):
    model_kwargs = dict(dim=4096, depth=30, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_8B(**kwargs):
    model_kwargs = dict(dim=3072, depth=72, num_heads=24, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_12B(**kwargs):
    model_kwargs = dict(dim=4096, depth=60, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_15B(**kwargs):
    model_kwargs = dict(dim=4096, depth=78, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_18B(**kwargs):
    model_kwargs = dict(dim=4096, depth=90, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_20B(**kwargs):
    model_kwargs = dict(dim=8192, depth=25, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_24B(**kwargs):
    model_kwargs = dict(dim=8192, depth=30, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_28B(**kwargs):
    model_kwargs = dict(dim=8192, depth=35, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_32B(**kwargs):
    model_kwargs = dict(dim=8192, depth=40, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_36B(**kwargs):
    model_kwargs = dict(dim=8192, depth=45, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt2_36B(**kwargs):
    model_kwargs = dict(dim=8192, depth=50, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


def gpt3(**kwargs):
    model_kwargs = dict(dim=12288, depth=96, num_heads=96, **kwargs)
    return _create_gpt_model(**model_kwargs)
