from torch import nn

from colossalai import kernel
from titan.layer.attention import GPTSelfAttention1D, FusedGPTSelfAttention1D
from .transformer_1d import GenericTransformerLayer1D


class GPTTransformerLayer1D(GenericTransformerLayer1D):

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 act_func: str = 'gelu',
                 mlp_ratio: float = 4,
                 attention_dropout_prob: float = 0,
                 hidden_dropout_prob: float = 0,
                 dtype=None,
                 checkpoint: bool = False,
                 max_position_embeddings: int = 1024,
                 layer_norm_epsilon: float = 0.00001,
                 apply_post_layer_norm: bool = False):
        attention = GPTSelfAttention1D
        layer_norm = nn.LayerNorm
        super().__init__(hidden_size,
                         num_attention_heads,
                         act_func=act_func,
                         mlp_ratio=mlp_ratio,
                         attention_dropout_prob=attention_dropout_prob,
                         hidden_dropout_prob=hidden_dropout_prob,
                         dtype=dtype,
                         checkpoint=checkpoint,
                         max_position_embeddings=max_position_embeddings,
                         layer_norm_epsilon=layer_norm_epsilon,
                         apply_post_layer_norm=apply_post_layer_norm,
                         attention=attention,
                         layer_norm=layer_norm)


class FusedGPTTransformerLayer1D(GenericTransformerLayer1D):

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 act_func: str = 'gelu',
                 mlp_ratio: float = 4,
                 attention_dropout_prob: float = 0,
                 hidden_dropout_prob: float = 0,
                 dtype=None,
                 checkpoint: bool = False,
                 max_position_embeddings: int = 1024,
                 layer_norm_epsilon: float = 0.00001,
                 apply_post_layer_norm: bool = False):
        attention = FusedGPTSelfAttention1D
        layer_norm = kernel.LayerNorm
        super().__init__(hidden_size,
                         num_attention_heads,
                         act_func=act_func,
                         mlp_ratio=mlp_ratio,
                         attention_dropout_prob=attention_dropout_prob,
                         hidden_dropout_prob=hidden_dropout_prob,
                         dtype=dtype,
                         checkpoint=checkpoint,
                         max_position_embeddings=max_position_embeddings,
                         layer_norm_epsilon=layer_norm_epsilon,
                         apply_post_layer_norm=apply_post_layer_norm,
                         attention=attention,
                         layer_norm=layer_norm)
