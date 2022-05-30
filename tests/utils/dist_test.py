import torch.multiprocessing as mp
from colossalai.utils import free_port
from functools import partial


def run_with_parallel_config(world_size, parallel_mode, run_func):
    """
    A wrapper function to reuse the same code snippet in layer/model testing.

    Args:
        world_size (int): the number of processes to launch
        parallel_mode (str): the parallelism method used
        run_func (Callable): the function to launch multiple processes, must have world_size, port and config as arguments.
    """

    port = free_port()

    config = dict(parallel=dict(tensor=dict(size=world_size, mode=parallel_mode)))

    if parallel_mode == '2.5d':
        config['parallel']['tensor']['depth'] = world_size // 4

    run_func = partial(run_func, world_size=world_size, port=port, config=config)
    mp.spawn(run_func, nprocs=world_size)


def run_with_moe_config(world_size, run_func):
    """
    A wrapper function to reuse the same code snippet in layer/model testing.

    Args:
        world_size (int): the number of processes to launch
        run_func (Callable): the function to launch multiple processes, must have world_size, port and config as arguments.
    """

    port = free_port()

    config = dict()

    run_func = partial(run_func, world_size=world_size, port=port, config=config)
    mp.spawn(run_func, nprocs=world_size)