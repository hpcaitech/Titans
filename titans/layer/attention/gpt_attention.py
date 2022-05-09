import math

import torch
from torch import dtype, nn

from colossalai import nn as col_nn
from colossalai.nn.layer.utils import divide
from colossalai.utils import get_current_device
from titans.decorator import no_support


@no_support(['sp'])
class GPTSelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 attention_dropout: float,
                 dropout: float,
                 bias: bool = True,
                 fuse_scale_mask_softmax: bool = False,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.fuse_scale_mask_softmax = fuse_scale_mask_softmax
        self.attention_head_size = divide(dim, num_heads)
        self.query_key_value = col_nn.Linear(dim, 3 * dim, dtype=dtype, bias=bias)
        if fuse_scale_mask_softmax:
            from colossalai.kernel import FusedScaleMaskSoftmax
            from colossalai.kernel.cuda_native.scaled_softmax import \
                AttnMaskType
            self.softmax = FusedScaleMaskSoftmax(input_in_fp16=True,
                                                 input_in_bf16=False,
                                                 attn_mask_type=AttnMaskType.causal,
                                                 scaled_masked_softmax_fusion=True,
                                                 mask_func=None,
                                                 softmax_in_fp32=True,
                                                 scale=math.sqrt(self.attention_head_size))
        else:
            self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = col_nn.Dropout(attention_dropout)
        self.dense = col_nn.Linear(dim, dim, dtype=dtype, bias=True)
        self.dropout = col_nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        qkv = self.query_key_value(x)
        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = divide(all_head_size, self.attention_head_size)
        new_qkv_shape = qkv.shape[:-1] + \
            (num_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        qkv = qkv.permute((0, 2, 1, 3))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        x = torch.matmul(q, k.transpose(-1, -2))

        if self.fuse_scale_mask_softmax:
            x = self.softmax(x, attention_mask)
        else:
            x = x / math.sqrt(self.attention_head_size)
            # causal mask
            q_len, k_len = q.size(-2), k.size(-2)
            causal_mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.uint8,
                                                device=get_current_device())).view(1, 1, q_len, k_len).bool()
            x = torch.where(causal_mask, x, torch.tensor(-1e4, dtype=x.dtype, device=get_current_device()))
            if attention_mask is not None:
                x = x + attention_mask
            x = self.softmax(x)

        x = self.attention_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size,)
        x = x.reshape(new_context_layer_shape)

        x = self.dense(x)
        x = self.dropout(x)

        return x
