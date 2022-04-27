from torch.nn.modules.loss import *
from torch.nn.modules.loss import _Loss


class MLM_loss(_Loss):

    def __init__(self, reduction: bool = True, *args, **kwargs):
        super().__init__()

    def itm_mlm_loss(self, output):
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def forward(self, *args):
        return self.itm_mlm_loss(*args)
