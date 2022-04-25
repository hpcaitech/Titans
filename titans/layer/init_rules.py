import math

from colossalai import nn as col_nn
from torch import nn

init_rules = dict(
    torch=dict(
        embed=dict(
            weight_initializer=col_nn.init.kaiming_uniform_(a=math.sqrt(5)),
            bias_initializer=col_nn.init.xavier_uniform_(a=1, scale=1),
            position_embed_initializer=col_nn.init.zeros_(),
        ),
        transformer=dict(
            weight_initializer=col_nn.init.kaiming_uniform_(a=math.sqrt(5)),
            bias_initializer=col_nn.init.xavier_uniform_(a=1, scale=1),
        ),
        head=dict(
            weight_initializer=col_nn.init.kaiming_uniform_(a=math.sqrt(5)),
            bias_initializer=col_nn.init.xavier_uniform_(a=1, scale=1),
        ),
    ),
    jax=dict(
        embed=dict(
            weight_initializer=col_nn.init.lecun_normal_(),
            bias_initializer=col_nn.init.zeros_(),
            position_embed_initializer=col_nn.init.trunc_normal_(std=.02),
        ),
        transformer=dict(
            weight_initializer=col_nn.init.xavier_uniform_(),
            bias_initializer=col_nn.init.normal_(std=1e-6),
        ),
        head=dict(
            weight_initializer=col_nn.init.zeros_(),
            bias_initializer=col_nn.init.zeros_(),
        ),
    ),
)