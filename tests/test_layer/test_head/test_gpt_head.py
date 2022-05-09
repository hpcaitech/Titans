import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from titans.layer.embedding import GPTEmbedding
from titans.layer.head import GPTLMHead
from titans.utils import split_data_for_tensor_parallel
from colossalai.utils import free_port
from colossalai.nn.layer.utils import divide
from colossalai import nn as col_nn
from functools import partial
from colossalai.global_variables import tensor_parallel_env as tp_env

BATCH_SIZE = 4
SEQ_LENGTH = 256
VOCAB_SIZE = 50304
HIDDEN_SIZE = 32


def run_gpt_head(data, hidden_size, vocab_size):

    #build model
    embedding_layer = GPTEmbedding(embedding_dim=hidden_size, vocab_size=vocab_size,
                                   max_position_embeddings=1024).cuda()
    model = GPTLMHead(dim=hidden_size, vocab_size=vocab_size, embedding_layer=embedding_layer).cuda()

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
    run_gpt_head(data, HIDDEN_SIZE, VOCAB_SIZE)


@pytest.mark.parametrize('parallel_config', [(4, '1d'), (4, '2d'), (4, '2.5d'), (8, '2.5d'), (8, '3d')])
def test_gpt_head(parallel_config):
    world_size, tp_mode = parallel_config
    port = free_port()

    config = dict(parallel=dict(tensor=dict(size=world_size, mode=tp_mode)))

    if tp_mode == '2.5d':
        config['parallel']['tensor']['depth'] = world_size // 4

    run_func = partial(run_dist, world_size=world_size, port=port, config=config)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_gpt_head((4, '1d'))
