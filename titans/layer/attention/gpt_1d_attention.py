import math
import torch
from torch import Tensor, nn
from colossalai import kernel
from colossalai.kernel.cuda_native.scaled_softmax import AttnMaskType
from colossalai.utils import checkpoint
from colossalai.utils.activation_checkpoint import checkpoint

from .attention_1d import GenericSelfAttention1D


class GPTSelfAttention1D(GenericSelfAttention1D):

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout_prob: float,
                 hidden_dropout_prob: float,
                 dtype=None,
                 checkpoint: bool = False,
                 max_position_embeddings=1024):
        super().__init__(hidden_size,
                         num_attention_heads,
                         attention_dropout_prob,
                         hidden_dropout_prob,
                         dtype=dtype,
                         checkpoint=checkpoint,
                         max_position_embeddings=max_position_embeddings)
        self.softmax = nn.Softmax(dim=-1)
        max_positions = max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions),
                                  dtype=torch.uint8)).view(1, 1, max_positions, max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

    def softmax_forward(self, attention_scores, attention_mask, query_layer, key_layer):
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # causal mask
        query_length, key_length = query_layer.size(-2), key_layer.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length].bool()
        attention_scores = torch.where(causal_mask, attention_scores, self.masked_bias.to(attention_scores))
        if attention_mask is not None:
            # Apply the attention mask
            attention_scores = attention_scores + attention_mask
        attention_scores = self.softmax(attention_scores)
        return attention_scores


class FusedGPTSelfAttention1D(GenericSelfAttention1D):

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout_prob: float,
                 hidden_dropout_prob: float,
                 dtype=None,
                 checkpoint: bool = False,
                 max_position_embeddings=1024):
        super().__init__(hidden_size,
                         num_attention_heads,
                         attention_dropout_prob,
                         hidden_dropout_prob,
                         dtype=dtype,
                         checkpoint=checkpoint,
                         max_position_embeddings=max_position_embeddings)
        self.softmax = kernel.FusedScaleMaskSoftmax(input_in_fp16=True,
                                                    input_in_bf16=False,
                                                    attn_mask_type=AttnMaskType.causal,
                                                    scaled_masked_softmax_fusion=True,
                                                    mask_func=None,
                                                    softmax_in_fp32=True,
                                                    scale=math.sqrt(self.attention_head_size))

    def softmax_forward(self, attention_scores, attention_mask, query_layer, key_layer):
        return self.softmax(attention_scores, attention_mask)
