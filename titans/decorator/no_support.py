from typing import List, Union, Callable
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode

SUPPORTED_MODES = ['tp', 'pp', 'sp', 'moe']


def no_support(modes: Union[str, List[str]]):
    """
    A decorator to indicate the forbidden parallel modes for the module.
    
    Args:
        modes (Union[str, List[str]]): the mode can only be tp (tensor parallel), 
            pp (pipeline parallel), sp (sequence parallel), and moe (mixture-of-experts).

    Usage:
        # if this model does not support tensor parallel version
        @no_support('tp')
        class SomeModule(torch.nn.Module):
            ...

        # if this model does not support tp and pp
        @no_support(['tp', 'pp'])
        class SomeModule(torch.nn.Module):
            ...
    """

    if isinstance(modes, str):
        assert modes in SUPPORTED_MODES, f'expected modes to be none, tp, pp, sp or moe, but got {modes}'
        modes = [modes]
    elif isinstance(modes, (tuple, list)):
        for mode in modes:
            assert mode in SUPPORTED_MODES, f'expected modes to be none, tp, pp, sp or moe, but got {mode}'
    else:
        raise TypeError(f'expected modes to be of type str or list, but got {type(modes)}')

    def _wrap_callable(callable_: Callable):
        assert hasattr(callable_, '__init__'), 'the wrapped callable must be a class'
        origin_init = callable_.__init__
        class_name = callable_.__class__.__name__

        def new_init(*args, **kwargs):
            if tp_env.mode != None:
                assert 'tp' not in modes, f'{class_name} does not support tensor parallel implementation'

            if MOE_CONTEXT.is_initialized:
                assert 'moe' not in modes, f'{class_name} does not support MOE implementation'

            if gpc.is_initialized(ParallelMode.PIPELINE) and gpc.get_world_size(ParallelMode.PIPELINE) > 1:
                assert 'pp' not in modes, f'{class_name} does not support pipeline parallel implementation'

            if gpc.is_initialized(ParallelMode.SEQUENCE) and gpc.get_world_size(ParallelMode.SEQUENCE):
                assert 'sp' not in modes, f'{class_name} does not support sequence parallel implementation'

            origin_init(*args, **kwargs)

        callable_.__init__ = new_init

        return callable_

    return _wrap_callable


def support_tp_pp_only():
    return no_support(['moe', 'sp'])


def support_sp_pp_only():
    return no_support(['moe', 'tp'])


def support_moe_only():
    return no_support(['tp', 'sp', 'pp'])


def no_parallel_support():
    return no_support(['tp', 'pp', 'sp', 'moe'])
