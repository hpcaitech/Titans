import colossalai
import pytest
import torch.nn as nn
import torch.multiprocessing as mp

from colossalai.utils import free_port
from functools import partial
from titans.decorator import no_support

CONFIG = dict(parallel=dict(tensor=dict(mode='1d', size=2)))


@no_support('tp')
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(x)


def run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port)
    try:
        net = Net()
    except Exception as e:
        assert isinstance(e, AssertionError)


def test_no_support():
    world_size = 2
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_no_support()
