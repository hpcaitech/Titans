import torch.nn.functional as F
from torch import nn

from colossalai import nn as col_nn


class TransformerMLP(nn.Module):

    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.linear_1 = col_nn.Linear(d_model, dim_feedforward)
        self.ff_drop = col_nn.Dropout(dropout)
        self.linear_2 = col_nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        x = self.ff_drop(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
