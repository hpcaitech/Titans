import colossalai
import pytest
import torch

from titans.model.vit import VisionTransformer
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.testing import rerun_if_address_is_in_use
from tests.utils import run_with_parallel_config

BATCH_SIZE = 4
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_HEADS = 4
IN_CHANS = 3
HIDDEN_SIZE = 32


def run_vit(data, img_size, patch_size, in_chans, hidden_size, num_heads):

    #build model
    model = VisionTransformer(img_size=img_size,
                              patch_size=patch_size,
                              in_chans=in_chans,
                              hidden_size=hidden_size,
                              num_heads=num_heads).cuda()

    # forward
    out = model(data)

    # backward
    out.mean().backward()


def run_dist(rank, world_size, port, config):
    colossalai.launch(config=config, rank=rank, world_size=world_size, port=port, host='localhost')

    if tp_env.mode == 'sequence':
        tp_env.mode = None

    data = torch.rand(BATCH_SIZE, IN_CHANS, IMAGE_SIZE, IMAGE_SIZE).cuda()
    run_vit(data, IMAGE_SIZE, PATCH_SIZE, IN_CHANS, HIDDEN_SIZE, NUM_HEADS)


@pytest.mark.parametrize('parallel_config', [(4, '1d'), (4, '2d'), (4, '2.5d'), (8, '2.5d'), (8, '3d')])
@rerun_if_address_is_in_use()
def test_vit(parallel_config):
    run_with_parallel_config(*parallel_config, run_func=run_dist)
