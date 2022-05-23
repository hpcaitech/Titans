import colossalai
import pytest
import torch

from titans.model.detr import DeTr
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.testing import rerun_if_address_is_in_use
from tests.utils import run_with_parallel_config

BATCH_SIZE = 1
HEIGHT = 800
WIDTH = 1200
PATCH_SIZE = 16
NUM_HEADS = 4
IN_CHANS = 3
HIDDEN_SIZE = 256
NUM_ENCODER_LAYER = 6
NUM_DECODER_LAYER = 6


def run_detr(data, img_size, patch_size, in_chans, hidden_size, num_heads, num_encoder_layer, num_decoder_layer):

    #build model
    model = DeTr(img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_encoder_layer=num_encoder_layer,
                    num_decoder_layer=num_decoder_layer).cuda()

    # forward
    out = model(data)

    # backward
    out.mean().backward()


def run_dist(rank, world_size, port, config):
    colossalai.launch(config=config, rank=rank, world_size=world_size, port=port, host='localhost')

    if tp_env.mode == 'sequence':
        tp_env.mode = None

    data = torch.rand(BATCH_SIZE, IN_CHANS, HEIGHT, WIDTH).cuda()
    run_detr(data, 224, PATCH_SIZE, IN_CHANS, HIDDEN_SIZE, NUM_HEADS, NUM_ENCODER_LAYER, NUM_DECODER_LAYER)


@pytest.mark.parametrize('parallel_config', [(2, '1d')])
@rerun_if_address_is_in_use()
def test_detr(parallel_config):
    run_with_parallel_config(*parallel_config, run_func=run_dist)
