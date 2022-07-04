import math

import torch
from torch import dtype, nn

from colossalai import nn as col_nn
from ..init_rules import init_rules
from titans.decorator import no_support


@no_support(['sp'])
class DeTrAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 attention_dropout: float,
                 dropout: float,
                 bias: bool = True,
                 dtype: dtype = None,
                 init_method: str = 'torch'):
        super().__init__()
        self.attention_head_size = hidden_size // num_heads
        self.query = col_nn.Linear1D_Col(hidden_size,
                                    hidden_size,
                                    dtype=dtype,
                                    bias=bias,
                                    **init_rules[init_method]['transformer'])
        self.key = col_nn.Linear1D_Col(hidden_size,
                                        hidden_size,
                                        dtype=dtype,
                                        bias=bias,
                                        **init_rules[init_method]['transformer'])
        self.value = col_nn.Linear1D_Col(hidden_size,
                                        hidden_size,
                                        dtype=dtype,
                                        bias=bias,
                                        **init_rules[init_method]['transformer'])
        self.attention_dropout = col_nn.Dropout(attention_dropout)
        self.dense = col_nn.Linear1D_Row(hidden_size, hidden_size, dtype=dtype, bias=True)
        self.dropout = col_nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        # bsz, tgt_len, all_head_size = q.shape
        # _, src_len, _ = k.shape

        # num_attention_heads = all_head_size // self.attention_head_size

        # if key_padding_mask is not None:
        #     assert key_padding_mask.shape == (bsz, src_len), \
        #         f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        #     key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
        #         expand(-1, num_attention_heads, -1, -1).reshape(bsz * num_attention_heads, 1, src_len)
        #     if attn_mask is None:
        #         attn_mask = key_padding_mask
        #     elif attn_mask.dtype == torch.bool:
        #         attn_mask = attn_mask.logical_or(key_padding_mask)
        #     else:
        #         attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # # convert mask to float
        # if attn_mask is not None and attn_mask.dtype == torch.bool:
        #     new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        #     new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        #     attn_mask = new_attn_mask


        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        all_head_size = q.shape[-1]
        num_attention_heads = all_head_size // self.attention_head_size

        new_q_shape = q.shape[:-1] + (num_attention_heads, self.attention_head_size)
        q = q.view(new_q_shape)
        q = q.permute((0, 2, 1, 3))

        new_k_shape = k.shape[:-1] + (num_attention_heads, self.attention_head_size)
        k = k.view(new_k_shape)
        k = k.permute((0, 2, 1, 3))

        new_v_shape = v.shape[:-1] + (num_attention_heads, self.attention_head_size)
        v = v.view(new_v_shape)
        v = v.permute((0, 2, 1, 3))

        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / math.sqrt(self.attention_head_size)

        # if attn_mask is not None:
        #     x += attn_mask

        x = self.softmax(x)
        x = self.attention_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size,)
        # the size of x after reshape is (BATCH_SZIE, SEQ_LEN, HIDDEN_SIZE)
        x = x.reshape(new_context_layer_shape)
        # the size of x after dense is (BATCH_SZIE, SEQ_LEN, HIDDEN_SIZE)
        x = self.dense(x)
        x = self.dropout(x)

        return x
