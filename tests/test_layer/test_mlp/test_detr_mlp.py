import colossalai
import pytest
import torch
import torch.nn.functional as F

from titans.layer.mlp import TransformerMLP, ViTMLP, DetrMLP
from titans.utils import split_data_for_tensor_parallel
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.testing import rerun_if_address_is_in_use
from tests.utils import run_with_parallel_config

BATCH_SIZE = 4
SEQ_LENGTH = 16
HIDDEN_SIZE = 32


def run_detr_mlp(data, hidden_size):

    #build model
    model = DetrMLP(input_dim=hidden_size, hidden_size=4*hidden_size, output_dim=hidden_size, num_layers=1).cuda()

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
    run_detr_mlp(data, HIDDEN_SIZE)


@pytest.mark.parametrize('parallel_config', [(4, '1d'), (4, '2d'), (4, '2.5d'), (8, '2.5d'), (8, '3d')])
@rerun_if_address_is_in_use()
def test_transformer_mlp(parallel_config):
    run_with_parallel_config(*parallel_config, run_func=run_dist)
