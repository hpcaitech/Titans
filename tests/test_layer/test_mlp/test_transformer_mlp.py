import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from titans.layer.mlp import TransformerMLP, ViTMLP
from titans.utils import split_data_for_tensor_parallel
from colossalai.utils import free_port
from functools import partial
from colossalai.global_variables import tensor_parallel_env as tp_env

BATCH_SIZE = 4
SEQ_LENGTH = 16
HIDDEN_SIZE = 32


def run_transformer_mlp(data, hidden_size):

    #build model
    model = TransformerMLP(hidden_size=hidden_size, mlp_ratio=4).cuda()

    # forward
    out = model(data)

    # backward
    out.mean().backward()


def run_vit_mlp(data, hidden_size):

    #build model
    model = ViTMLP(dim=hidden_size, mlp_ratio=4, activation=F.gelu, dropout=0.0).cuda()

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
    run_transformer_mlp(data, HIDDEN_SIZE)
    run_vit_mlp(data, HIDDEN_SIZE)


@pytest.mark.parametrize('parallel_config', [(4, '1d'), (4, '2d'), (4, '2.5d'), (8, '2.5d'), (8, '3d')])
def test_transformer_mlp(parallel_config):
    world_size, tp_mode = parallel_config
    port = free_port()

    config = dict(parallel=dict(tensor=dict(size=world_size, mode=tp_mode)))

    if tp_mode == '2.5d':
        config['parallel']['tensor']['depth'] = world_size // 4

    run_func = partial(run_dist, world_size=world_size, port=port, config=config)
    mp.spawn(run_func, nprocs=world_size)
