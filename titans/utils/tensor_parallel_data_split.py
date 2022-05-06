import torch
from torch import Tensor
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.nn.layer.parallel_3d._utils import get_parallel_mode_from_env
from colossalai.constants import INPUT_GROUP_3D, OUTPUT_GROUP_3D, WEIGHT_GROUP_3D


def split_data_2d(x: Tensor) -> Tensor:
    """
    2D tensor parallel requries splitting the data in the first dimension and last dimension
    """
    j = gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)
    i = gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)
    x = torch.chunk(x, tp_env.summa_dim, dim=0)[i]
    x = torch.chunk(x, tp_env.summa_dim, dim=-1)[j]
    return x


def split_data_2p5d(x: Tensor) -> Tensor:
    """
    2.5D tensor parallel requries splitting the data in the first dimension and last dimension just like 2D
    """
    i = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
    j = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
    x = torch.chunk(x, tp_env.tesseract_dim, dim=0)[i]
    x = torch.chunk(x, tp_env.tesseract_dim, dim=-1)[j]
    return x


def split_data_3d(x: Tensor) -> Tensor:
    """
    2.5D tensor parallel requries splitting the data in the first dimension twice and last dimension once
    """
    input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
    weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
    output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)

    j = gpc.get_local_rank(input_parallel_mode)
    i = gpc.get_local_rank(weight_parallel_mode)
    k = gpc.get_local_rank(output_parallel_mode)

    x = torch.chunk(x, tp_env.depth_3d, dim=0)[i]
    x = torch.chunk(x, tp_env.depth_3d, dim=-1)[k]
    x = torch.chunk(x, tp_env.depth_3d, dim=0)[j]
    return x


def split_data_for_tensor_parallel(x: Tensor) -> Tensor:
    """
    Split the data based on the tensor parallel environment
    """

    if tp_env.mode == '2d':
        return split_data_2d(x)
    elif tp_env.mode == '2.5d':
        return split_data_2p5d(x)
    elif tp_env.mode == '3d':
        return split_data_3d(x)
    else:
        return x
