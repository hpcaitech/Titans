from colossalai.utils.activation_checkpoint import checkpoint    #TODO:There are two checkpoints importing from different files.
import torch
from torch import nn, Tensor
from colossalai.core import global_context as gpc
from colossalai.nn.layer.utils import divide
from colossalai.utils import checkpoint
from colossalai.nn.layer import Linear1D_Col, Linear1D_Row
from colossalai.nn.layer.base_layer import ParallelLayer
from colossalai import nn as col_nn


class GenericSelfAttention1D(ParallelLayer):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout_prob: float,
        hidden_dropout_prob: float,
        dtype=None,
        checkpoint: bool = False,
        max_position_embeddings=1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads, gpc.tensor_parallel_size)
        self.hidden_size_per_partition = divide(hidden_size, gpc.tensor_parallel_size)
        self.checkpoint = checkpoint
        self.query_key_value = Linear1D_Col(
            hidden_size,
            3 * hidden_size,
            dtype=dtype,
        )
        self.attention_dropout = col_nn.Dropout(attention_dropout_prob)
        self.dense = Linear1D_Row(
            hidden_size,
            hidden_size,
            dtype=dtype,
            parallel_input=True,
        )
        self.dropout = col_nn.Dropout(hidden_dropout_prob)

    def softmax_forward(self, attention_scores, attention_mask, query_layer, key_layer):
        raise NotImplementedError

    def _forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
        query_key_value = self.query_key_value(hidden_states)
        new_qkv_shape = query_key_value.shape[:-1] + \
            (self.num_attention_heads_per_partition, 3 * self.attention_head_size)
        query_key_value = query_key_value.view(new_qkv_shape)
        query_key_value = query_key_value.permute((0, 2, 1, 3))
        query_layer, key_layer, value_layer = torch.chunk(query_key_value, 3, dim=-1)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = self.softmax_forward(attention_scores, attention_mask, query_layer, key_layer)

        attention_scores = attention_scores.type(value_layer.dtype)

        attention_probs = self.attention_dropout(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        output = self.dense(context_layer)
        output = self.dropout(output)

        return output

    def _checkpoint_forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
        return checkpoint(self._forward, hidden_states, attention_mask)

    def forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
        if self.checkpoint:
            return self._checkpoint_forward(hidden_states, attention_mask)
        else:
            return self._forward(hidden_states, attention_mask)
