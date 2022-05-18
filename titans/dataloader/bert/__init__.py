try:
    from .bert_pretrain import get_bert_pretrain_data_loader
except ImportError:
    raise ImportError('lddl is required for BERT pretraining but not found, '
    'you can install lddl by pip install git+https://github.com/NVIDIA/DeepLearningExamples.git#subdirectory=Tools/lddl')

__all__ = ['get_bert_pretrain_data_loader']