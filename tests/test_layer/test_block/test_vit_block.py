import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from titans.layer.block import ViTBlock
from titans.utils import split_data_for_tensor_parallel
from colossalai.utils import free_port
from colossalai.nn.layer.utils import divide
from colossalai import nn as col_nn
from functools import partial
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.testing import rerun_if_address_is_in_use
from tests.utils import run_with_parallel_config

BATCH_SIZE = 4
SEQ_LENGTH = 16
NUM_HEADS = 4
HIDDEN_SIZE = 32


def run_vit_block(data, hidden_size, num_heads):

    #build model
    model = ViTBlock(dim=hidden_size, num_heads=num_heads, mlp_ratio=4, activation=F.gelu).cuda()

    # forward
    out = model(data)

    # backward
    out.mean().backward()


def run_dist(rank, world_size, port, config):
    colossalai.launch(config=config, rank=rank, world_size=world_size, port=port, host='localhost')

    if tp_env.mode == 'sequence':
        tp_env.mode = None

    data = torch.rand(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE).cuda()
    data = split_data_for_tensor_parallel(data)
    run_vit_block(data, HIDDEN_SIZE, NUM_HEADS)


@pytest.mark.parametrize('parallel_config', [(4, '1d'), (4, '2d'), (4, '2.5d'), (8, '2.5d'), (8, '3d')])
@rerun_if_address_is_in_use()
def test_vit_block(parallel_config):
    run_with_parallel_config(*parallel_config, run_func=run_dist)
