from typing import Callable

from torch import dtype

from colossalai import nn as col_nn
from colossalai.nn.layer.utils import CheckpointModule
from colossalai import nn as col_nn
from colossalai.nn.layer import MoeModule

from titans.layer.attention import GPTSelfAttention

from titans.decorator import support_tp_pp_only
from titans.layer.mlp import TransformerMLP


@support_tp_pp_only()
class GPTBlock(CheckpointModule):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float,
                 activation: Callable,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 layernorm_epsilon: float = 1e-5,
                 dtype: dtype = None,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 fuse_scale_mask_softmax: bool = False,
                 checkpoint: bool = False,
                 activation_offload: bool = False):
        super().__init__(checkpoint, activation_offload)
        self.apply_post_layernorm = apply_post_layernorm
        self.norm1 = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.attn = GPTSelfAttention(dim=dim,
                                     num_heads=num_heads,
                                     attention_dropout=attention_dropout,
                                     dropout=dropout,
                                     bias=bias,
                                     fuse_scale_mask_softmax=fuse_scale_mask_softmax,
                                     dtype=dtype)
        self.norm2 = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.mlp = TransformerMLP(hidden_size=dim,
                                  mlp_ratio=mlp_ratio,
                                  act_func=activation,
                                  dropout_prob=dropout,
                                  dtype=dtype,
                                  bias=bias)

    def _forward(self, x, attention_mask=None):
        if attention_mask is not None and attention_mask.dtype != x.dtype:
            attention_mask = attention_mask.to(x.dtype)
        if not self.apply_post_layernorm:
            residual = x
        x = self.norm1(x)
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.attn(x, attention_mask)

        if not self.apply_post_layernorm:
            residual = x
        x = self.norm2(x)
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.mlp(x)

        return x, attention_mask


class MOEGPTBlock(CheckpointModule):

    def __init__(self,
                 num_experts: int,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float,
                 activation: Callable,
                 capacity_factor_train: float = 1.0,
                 capacity_factor_eval: float = 1.0,
                 use_residual: bool = False,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 layernorm_epsilon: float = 1e-5,
                 dtype: dtype = None,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 fuse_scale_mask_softmax: bool = False,
                 checkpoint: bool = False):
        super().__init__(checkpoint)
        self.apply_post_layernorm = apply_post_layernorm
        self.norm1 = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.attn = GPTSelfAttention(dim=dim,
                                     num_heads=num_heads,
                                     attention_dropout=attention_dropout,
                                     dropout=dropout,
                                     bias=bias,
                                     fuse_scale_mask_softmax=fuse_scale_mask_softmax,
                                     dtype=dtype)
        self.norm2 = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)

        mpl_factory_dict = dict(dim=dim,
                                mlp_ratio=mlp_ratio,
                                activation=activation,
                                dropout=dropout,
                                dtype=dtype,
                                bias=bias)

        self.mlp = MoeModule(dim_model=dim,
                             num_experts=num_experts,
                             top_k=1,
                             capacity_factor_train=capacity_factor_train,
                             capacity_factor_eval=capacity_factor_eval,
                             noisy_policy='Jitter',
                             expert_cls=TransformerMLP,
                             **mpl_factory_dict)

    def _forward(self, x, attention_mask=None):
        if not self.apply_post_layernorm:
            residual = x
        x = self.norm1(x)
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.attn(x, attention_mask)

        if not self.apply_post_layernorm:
            residual = x
        x = self.norm2(x)
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.mlp(x)

        return x, attention_mask
