import colossalai
import torch
import torch.nn.functional as F

from titans.layer.mlp import MLPForMoe
from titans.utils import split_data_for_tensor_parallel
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.testing import rerun_if_address_is_in_use
from tests.utils import run_with_moe_config

BATCH_SIZE = 4
SEQ_LENGTH = 16
HIDDEN_SIZE = 32
D_FF = 4 * 32


def run_moe_mlp(data, hidden_size, d_ff):

    #build model
    model = MLPForMoe(hidden_size=hidden_size, d_ff=d_ff).cuda()

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
    run_moe_mlp(data, HIDDEN_SIZE, D_FF)


@rerun_if_address_is_in_use()
def test_moe_mlp():
    run_with_moe_config(4, run_func=run_dist)
