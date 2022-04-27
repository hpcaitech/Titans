import torch.nn as nn

from colossalai.utils import get_current_device


class MLPForMoe(nn.Module):
    """FFN composed with two linear layers, also called MLP.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 activation=None,
                 drop_rate: float = 0,
                 bias: bool = True,
                 dropout1=None,
                 dropout2=None):
        super().__init__()
        dense1 = nn.Linear(d_model, d_ff, bias, device=get_current_device())
        act = nn.GELU() if activation is None else activation
        dense2 = nn.Linear(d_ff, d_model, bias, device=get_current_device())
        drop1 = nn.Dropout(drop_rate) if dropout1 is None else dropout1
        drop2 = nn.Dropout(drop_rate) if dropout2 is None else dropout2

        self.ffn = nn.Sequential(dense1, act, drop1, dense2, drop2)

    def forward(self, x):
        return self.ffn(x)