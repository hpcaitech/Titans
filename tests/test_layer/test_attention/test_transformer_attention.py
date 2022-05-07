import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from titans.layer.attention import TransformerSelfAttention, GPTSelfAttention, ViTSelfAttention
from titans.utils import split_data_for_tensor_parallel
from colossalai.utils import free_port
from colossalai.nn.layer.utils import divide
from colossalai import nn as col_nn
from functools import partial
from colossalai.global_variables import tensor_parallel_env as tp_env

BATCH_SIZE = 4
SEQ_LENGTH = 16
NUM_HEADS = 4
HIDDEN_SIZE = 32


def run_transformer_attention(data, hidden_size, num_heads):

    #build model
    model = TransformerSelfAttention(dropout=0.0).cuda()

    #process data
    query_key_value = col_nn.Linear(hidden_size, 3 * hidden_size)
    qkv = query_key_value(data)
    all_head_size = qkv.shape[-1] // 3
    attention_head_size = divide(hidden_size, num_heads)
    num_attention_heads = divide(all_head_size, attention_head_size)
    new_qkv_shape = qkv.shape[:-1] + \
        (num_attention_heads, 3 * attention_head_size)
    qkv = qkv.view(new_qkv_shape)
    qkv = qkv.permute((0, 2, 1, 3))
    q, k, v = torch.chunk(qkv, 3, dim=-1)

    # forward
    out = model(q, k, v)

    # backward
    out.mean().backward()


def run_gpt_attention(data, hidden_size, num_heads):

    #build model
    model = GPTSelfAttention(dim=hidden_size, num_heads=num_heads, attention_dropout=0.0, dropout=0.0).cuda()

    # forward
    out = model(data)

    # backward
    out.mean().backward()


def run_vit_attention(data, hidden_size, num_heads):

    #build model
    model = ViTSelfAttention(dim=hidden_size, num_heads=num_heads, attention_dropout=0.0, dropout=0.0).cuda()

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
    run_gpt_attention(data, HIDDEN_SIZE, NUM_HEADS)
    run_vit_attention(data, HIDDEN_SIZE, NUM_HEADS)
    run_transformer_attention(data, HIDDEN_SIZE, NUM_HEADS)


@pytest.mark.parametrize('parallel_config', [(4, 'sequence'), (4, '1d'), (4, '2d'), (4, '2.5d'), (8, '2.5d'),
                                             (8, '3d')])
def test_transformer_attention(parallel_config):
    world_size, tp_mode = parallel_config
    port = free_port()

    config = dict(parallel=dict(tensor=dict(size=world_size, mode=tp_mode)))

    if tp_mode == '2.5d':
        config['parallel']['tensor']['depth'] = world_size // 4

    run_func = partial(run_dist, world_size=world_size, port=port, config=config)
    mp.spawn(run_func, nprocs=world_size)
