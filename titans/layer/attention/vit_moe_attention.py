import math

import torch
import torch.nn as nn

from colossalai.utils import get_current_device


class SelfAttentionForMoe(nn.Module):
    """Standard ViT self attention.
    """

    def __init__(self,
                 hidden_size: int,
                 n_heads: int,
                 d_kv: int,
                 attention_drop: float = 0,
                 drop_rate: float = 0,
                 bias: bool = True,
                 dropout1=None,
                 dropout2=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.scale = 1.0 / math.sqrt(self.d_kv)

        self.dense1 = nn.Linear(hidden_size, 3 * n_heads * d_kv, bias, device=get_current_device())
        self.softmax = nn.Softmax(dim=-1)
        self.atten_drop = nn.Dropout(attention_drop) if dropout1 is None else dropout1
        self.dense2 = nn.Linear(n_heads * d_kv, hidden_size, device=get_current_device())
        self.dropout = nn.Dropout(drop_rate) if dropout2 is None else dropout2

    def forward(self, x):
        qkv = self.dense1(x)
        new_shape = qkv.shape[:2] + (3, self.n_heads, self.d_kv)
        qkv = qkv.view(*new_shape)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[:]

        x = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        x = self.atten_drop(self.softmax(x))

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_shape = x.shape[:2] + (self.n_heads * self.d_kv,)
        x = x.reshape(*new_shape)
        x = self.dense2(x)
        x = self.dropout(x)

        return x
