import colossalai
import pytest
import torch

from titans.layer.head import ViTHead
from titans.utils import split_data_for_tensor_parallel
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.testing import rerun_if_address_is_in_use
from tests.utils import run_with_parallel_config

BATCH_SIZE = 4
MIDDLE_DIM = 80
NUM_CLASSES = 10
HIDDEN_SIZE = 32


def run_vit_head(data, hidden_size, num_classes):

    #build model
    model = ViTHead(dim=hidden_size, num_classes=num_classes).cuda()

    # forward
    out = model(data)

    # backward
    out.mean().backward()


def run_dist(rank, world_size, port, config):
    colossalai.launch(config=config, rank=rank, world_size=world_size, port=port, host='localhost')

    if tp_env.mode == 'sequence':
        tp_env.mode = None

    data = torch.rand(BATCH_SIZE, MIDDLE_DIM, HIDDEN_SIZE).cuda()
    data = split_data_for_tensor_parallel(data)
    run_vit_head(data, HIDDEN_SIZE, NUM_CLASSES)


@pytest.mark.parametrize('parallel_config', [(4, '1d'), (4, '2d'), (4, '2.5d'), (8, '2.5d'), (8, '3d')])
@rerun_if_address_is_in_use()
def test_vit_head(parallel_config):
    run_with_parallel_config(*parallel_config, run_func=run_dist)
