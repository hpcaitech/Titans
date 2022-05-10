import colossalai
import pytest
import torch

from titans.model.moe import MOEGPT, ViTMoE, Widenet
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.testing import rerun_if_address_is_in_use
from tests.utils import run_with_moe_config

NUM_EXPERTS = 64
BATCH_SIZE = 4
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_HEADS = 4
IN_CHANS = 3
HIDDEN_SIZE = 32

SEQ_LENGHT = 16
VOCAB_SIZE = 50304

def run_moe_gpt(data, num_experts, hidden_size, num_heads):

    #build model
    model = MOEGPT(num_experts=num_experts,
                    hidden_size = hidden_size,
                    num_heads=num_heads).cuda()

    # forward
    out = model(data)

    # backward
    out.mean().backward()


def run_vit_moe(data, num_experts, img_size, patch_size, in_chans, hidden_size, num_heads):

    #build model
    model = ViTMoE(num_experts=num_experts,
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    hidden_size=hidden_size,
                    num_heads=num_heads).cuda()

    # forward
    out = model(data)

    # backward
    out.mean().backward()


def run_widenet(data, num_experts, img_size, patch_size, in_chans, hidden_size, num_heads):

    #build model
    model = Widenet(num_experts=num_experts,
                    img_size=img_size,
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

    language_data = torch.rand(BATCH_SIZE, SEQ_LENGHT) * VOCAB_SIZE
    language_data = language_data.int().cuda()
    run_moe_gpt(language_data, NUM_EXPERTS, HIDDEN_SIZE, NUM_HEADS)

    image_data = torch.rand(BATCH_SIZE, IN_CHANS, IMAGE_SIZE, IMAGE_SIZE).cuda()
    run_vit_moe(image_data, NUM_EXPERTS, IMAGE_SIZE, PATCH_SIZE, IN_CHANS, HIDDEN_SIZE, NUM_HEADS)
    run_widenet(image_data, NUM_EXPERTS, IMAGE_SIZE, PATCH_SIZE, IN_CHANS, HIDDEN_SIZE, NUM_HEADS)


@rerun_if_address_is_in_use()
def test_moe():
    run_with_moe_config(4, run_func=run_dist)
