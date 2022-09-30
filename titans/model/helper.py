import torch
import torch.nn as nn
from colossalai.nn.layer import WrappedDropPath as DropPath
from colossalai.nn.layer.utils import CheckpointModule


class TransformerLayer(CheckpointModule):
    """Transformer layer builder.
    """

    def __init__(self,
                 att: nn.Module,
                 ffn: nn.Module,
                 norm1: nn.Module,
                 norm2: nn.Module,
                 droppath=None,
                 droppath_rate: float = 0,
                 checkpoint: bool = False):
        super().__init__(checkpoint=checkpoint)
        self.att = att
        self.ffn = ffn
        self.norm1 = norm1
        self.norm2 = norm2
        self.droppath = DropPath(droppath_rate) if droppath is None else droppath

    def _forward(self, x, y):
        x1 = x + self.droppath(self.att(self.norm1(x)))
        x2 = self.ffn(self.norm2(x1))

        if isinstance(x2, tuple):
            x, z = x2
            y = y + z
        else:
            x = x2

        x = x1 + self.droppath(x)
        return x, y
