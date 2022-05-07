import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from titans.model.gpt import GPT
from titans.utils import split_data_for_tensor_parallel
from colossalai.utils import free_port
from colossalai.nn.layer.utils import divide
from colossalai import nn as col_nn
from functools import partial
from colossalai.global_variables import tensor_parallel_env as tp_env

BATCH_SIZE = 4
SEQ_LENGHT = 16
HIDDEN_SIZE = 32
NUM_HEADS = 4
VOCAB_SIZE = 50304


def run_gpt(data, hidden_size, num_heads):

    #build model
    model = GPT(dim=hidden_size, num_heads=num_heads).cuda()

    # forward
    out = model(data)

    # backward
    out.mean().backward()


def run_dist(rank, world_size, port, config):
    colossalai.launch(config=config, rank=rank, world_size=world_size, port=port, host='localhost')

    if tp_env.mode == 'sequence':
        tp_env.mode = None

    data = torch.rand(BATCH_SIZE, SEQ_LENGHT) * VOCAB_SIZE
    data = data.int().cuda()
    run_gpt(data, HIDDEN_SIZE, NUM_HEADS)


@pytest.mark.parametrize('parallel_config', [(4, 'sequence'), (4, '1d'), (4, '2d'), (4, '2.5d'), (8, '2.5d'),
                                             (8, '3d')])
def test_gpt(parallel_config):
    world_size, tp_mode = parallel_config
    port = free_port()

    config = dict(parallel=dict(tensor=dict(size=world_size, mode=tp_mode)))

    if tp_mode == '2.5d':
        config['parallel']['tensor']['depth'] = world_size // 4

    run_func = partial(run_dist, world_size=world_size, port=port, config=config)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_gpt((4, '2d'))
