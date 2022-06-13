from colossalai.context.parallel_mode import ParallelMode
from typing import Callable
import math
from torch import dtype
import torch.nn as nn
import torch
from colossalai import nn as col_nn
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
from colossalai.core import global_context as gpc
import inspect
from colossalai.pipeline.utils import partition_uniform
from colossalai import kernel
from colossalai.logging import get_dist_logger
from titans.decorator import support_tp_pp_only
from titans.layer.block import DeepNetBlock
from titans.layer.embedding import GPTEmbedding
from titans.layer.head import GPTLMHead
from titans.layer.block import GPTBlock
from titans.loss.lm_loss import GPTLMLoss

__all__ = ['DeepNet', 'deepnet_small']


@support_tp_pp_only()
class DeepNet(nn.Module):
    """The decoder-only DeepNet model is modified from the GPT model.

    Args:
        vocab_size(int): The size of dictionary, defaults to 50304.
        max_position_embeddings(int): The max value of positional embeddings, defaults to 1024.
        dim(int): Hidden size of the transformer blocks, defaults to 768.
        num_heads(int): The number of heads in transformer blocks, defaults to 12.
        depth(int): The number of transformer layers, defaults to 12.
        mlp_ratio(float): The ratio used in mlp layer, defaults to 4.0.
        dropout(float): The ratio used to construct dropout modules, which indicates the percentage of parameters should be casted to zero, defaults to 0.1.
        embedding_dropout(float): The ratio used to construct embedding dropout modules, which indicates the percentage of parameters should be casted to zero, defaults to 0.1.
        attention_dropout(float): The ratio used to construct attention dropout modules, which indicates the percentage of parameters should be casted to zero, defaults to 0.1.
        layernorm_epsilon(float): The argument used to construct layernorm modules, defaults to 1e-5.
        activation(Callable): The activation function used in model, defaults to nn.functional.gelu.
        padding_idx(int): The length to be padded for each batch, defaults to None.
        dtype (:class:`torch.dtype`): The dtype of parameters, defaults to None.
        bias (bool): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        fuse_scale_mask_softmax(bool): If set to "True", FuseScaleMaskSoftmax will be used in self-attention layer, defaults to ``False``.
        checkpoint(bool): If set to "True", checkpoint feature will be activated to save memory, defaults to ``False``.
        activation_offload(bool): If set to "True", offload feature will be activated during checkpointing, defaults to ``False``.    
    """

    def __init__(self,
                 vocab_size: int = 50304,
                 max_position_embeddings: int = 1024,
                 hidden_size: int = 768,
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
                 fuse_scale_mask_softmax: bool = False,
                 checkpoint: bool = False,
                 activation_offload: bool = False) -> None:
        super().__init__()
        self.embed = GPTEmbedding(embedding_dim=hidden_size,
                                  vocab_size=vocab_size,
                                  max_position_embeddings=max_position_embeddings,
                                  padding_idx=padding_idx,
                                  dropout=embedding_dropout,
                                  dtype=dtype)
        alpha = math.sqrt(2 * depth)
        self.blocks = nn.ModuleList([
            DeepNetBlock(hidden_size=hidden_size,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         activation=activation,
                         attention_dropout=attention_dropout,
                         dropout=dropout,
                         alpha=alpha,
                         layernorm_epsilon=layernorm_epsilon,
                         dtype=dtype,
                         bias=bias,
                         fuse_scale_mask_softmax=fuse_scale_mask_softmax,
                         checkpoint=checkpoint,
                         activation_offload=activation_offload) for _ in range(depth)
        ])

        self.norm = col_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)

        self.head = GPTLMHead(hidden_size=hidden_size, vocab_size=vocab_size, embedding_layer=self.embed, dtype=dtype)

    def forward(self, input_ids, attention_mask=None):

        # the size of input_ids is (BATCH_SIZE, SEQ_LEN)
        x = self.embed(input_ids)
        # the size of x after embed layer is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

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

        # the size of x in blocks is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        for block in self.blocks:
            x, attention_mask = block(x, attention_mask)

        x = self.head(self.norm(x))
        # the size of x is (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        return x


def _create_deepnet_model(**model_kwargs):
    model = DeepNet(**model_kwargs)
    return model


def deepnet_small(**kwargs):
    model_kwargs = dict(dim=768, depth=12, num_heads=12, **kwargs)
    return _create_deepnet_model(**model_kwargs)
