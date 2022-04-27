from colossalai.utils.activation_checkpoint import checkpoint    #TODO:There are two checkpoints importing from different files.
from torch import nn, Tensor
from colossalai.nn.layer.utils import ACT2FN
from colossalai.utils import checkpoint
from colossalai.nn.layer import Linear1D_Col, Linear1D_Row
from colossalai.nn.layer.base_layer import ParallelLayer
from colossalai import nn as col_nn


class MLP1D(ParallelLayer):

    def __init__(
        self,
        in_features: int,
        mlp_ratio: float,
        act_func: str = 'gelu',
        dropout_prob: float = 0.,
        dtype=None,
        checkpoint: bool = False,
        skip_bias_add: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.mlp_ratio = mlp_ratio
        self.checkpoint = checkpoint
        self.skip_bias_add = skip_bias_add

        self.act = ACT2FN[act_func]
        skip_dense_1_add_bias = False

        # Project to mlp_ratio * h.
        self.dense_1 = Linear1D_Col(
            self.in_features,
            int(self.mlp_ratio * self.in_features),
            dtype=dtype,
            gather_output=False,
            skip_bias_add=skip_dense_1_add_bias,
        )

        # Project back to h.
        self.dense_2 = Linear1D_Row(
            int(self.mlp_ratio * self.in_features),
            self.in_features,
            dtype=dtype,
            parallel_input=True,
        )

        self.dropout = col_nn.Dropout(dropout_prob)

    def _forward(self, hidden_states: Tensor) -> Tensor:
        intermediate_output = self.dense_1(hidden_states)
        intermediate_output = self.act(intermediate_output)

        output = self.dense_2(intermediate_output)
        output = self.dropout(output)
        return output

    def _checkpoint_forward(self, hidden_states: Tensor) -> Tensor:
        return checkpoint(self._forward, hidden_states)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.checkpoint:
            return self._checkpoint_forward(hidden_states)
        else:
            return self._forward(hidden_states)
