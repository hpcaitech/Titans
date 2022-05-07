import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from titans.model.vit import VisionTransformer
from titans.utils import split_data_for_tensor_parallel
from colossalai.utils import free_port
from colossalai.nn.layer.utils import divide
from colossalai import nn as col_nn
from functools import partial
from colossalai.global_variables import tensor_parallel_env as tp_env

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
                              dim=hidden_size,
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


@pytest.mark.parametrize('parallel_config', [(4, 'sequence'), (4, '1d'), (4, '2d'), (4, '2.5d'), (8, '2.5d'),
                                             (8, '3d')])
def test_vit_embedding(parallel_config):
    world_size, tp_mode = parallel_config
    port = free_port()

    config = dict(parallel=dict(tensor=dict(size=world_size, mode=tp_mode)))

    if tp_mode == '2.5d':
        config['parallel']['tensor']['depth'] = world_size // 4

    run_func = partial(run_dist, world_size=world_size, port=port, config=config)
    mp.spawn(run_func, nprocs=world_size)
