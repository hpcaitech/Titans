import colossalai
import pytest
import torch

from titans.layer.embedding import ViTEmbedding
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.testing import rerun_if_address_is_in_use
from tests.utils import run_with_parallel_config

BATCH_SIZE = 4
IMAGE_SIZE = 224
PATCH_SIZE = 16
IN_CHANS = 3
HIDDEN_SIZE = 32


def run_vit_embed(data, img_size, patch_size, in_chans, hidden_size):

    #build model
    model = ViTEmbedding(img_size=img_size,
                         patch_size=patch_size,
                         in_chans=in_chans,
                         embedding_dim=hidden_size,
                         dropout=0.0).cuda()

    # forward
    out = model(data)

    # backward
    out.mean().backward()


def run_dist(rank, world_size, port, config):
    colossalai.launch(config=config, rank=rank, world_size=world_size, port=port, host='localhost')

    if tp_env.mode == 'sequence':
        tp_env.mode = None

    data = torch.rand(BATCH_SIZE, IN_CHANS, IMAGE_SIZE, IMAGE_SIZE).cuda()
    run_vit_embed(data, IMAGE_SIZE, PATCH_SIZE, IN_CHANS, HIDDEN_SIZE)


@pytest.mark.parametrize('parallel_config', [(4, '1d'), (4, '2d'), (4, '2.5d'), (8, '2.5d'), (8, '3d')])
@rerun_if_address_is_in_use()
def test_vit_embedding(parallel_config):
    run_with_parallel_config(*parallel_config, run_func=run_dist)
