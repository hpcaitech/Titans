import colossalai
import os
import pytest
import torch.multiprocessing as mp
from colossalai.context.parallel_mode import ParallelMode
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from functools import partial

try:
    from titans.dataloader.bert import get_bert_pretrain_data_loader
except:
    # to bypass pytest
    get_bert_pretrain_data_loader = None


CONFIG = dict(
    parallel=dict(
        tensor=dict(size=2, mode='1d')
    )
)
DATA_PATH = os.environ['PARQUET_PATH']

def load_data(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, port=port, host='localhost')

    dataloader = get_bert_pretrain_data_loader(
        path=DATA_PATH,
        vocab_file='bert-large-uncased-vocab',
        local_rank=rank,
        process_group=gpc.get_group(ParallelMode.DATA),
        data_loader_kwargs={
            'batch_size': 16,
            # 'num_workers': 4,
            # 'persistent_workers': True,
            # 'pin_memory': True,
        },
    )

    for _ in dataloader:
        break
    
    gpc.destroy()


@pytest.mark.skip('This test should be manually invoked as the dataset is too large')
def test_bert_pretrain_dataloader():
    world_size = 4
    port = free_port()
    run_func = partial(load_data, world_size=world_size, port=port)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_bert_pretrain_dataloader()
