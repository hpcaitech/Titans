"""
This file is adapated from 
"""

import logging
import os
import torch
import transformers
import torch.distributed as dist

try:
  from lddl.utils import get_all_parquets_under, get_all_bin_ids, get_file_paths_for_bin_id
  from lddl.torch.dataloader import DataLoader
  from .parquet_dataset import ParquetDataset
  from lddl.torch.bert import _decode_record_batch, _to_encoded_inputs, _mask_tokens, BertPretrainBinned
  from lddl.torch.log import DatasetLogger
  from lddl.torch.utils import get_node_rank, get_nproc_per_node
except ImportError:
  raise ImportError('lddl is required for BERT pretraining but not found, '
    'you can install lddl by pip install git+https://github.com/NVIDIA/DeepLearningExamples.git#subdirectory=Tools/lddl')


class BertPretrainDataset(ParquetDataset):

  def _decode_record_batch(self, b):
    return _decode_record_batch(b)


def get_bert_pretrain_data_loader(
    path,
    shuffle_buffer_size=16384,
    shuffle_buffer_warmup_factor=16,
    tokenizer_class=transformers.BertTokenizerFast,
    vocab_file=None,
    tokenizer_kwargs={},
    data_loader_class=DataLoader,
    data_loader_kwargs={},
    mlm_probability=0.15,
    base_seed=12345,
    log_dir=None,
    log_level=logging.INFO,
    return_raw_samples=False,
    start_epoch=0,
    sequence_length_alignment=8,
    ignore_index=-1,
    process_group=None,
):
  """Gets a PyTorch DataLoader for the BERT pretraining task.

  The LDDL DataLoader can be used in the same way as a normal PyTorch
  DataLoader. The 'persistent_workers' attribute will always be enabled.

  The LDDL DataLoader streams samples from disk into memory, and uses a shuffle
  buffer to perform shuffling: at each iteration, a random sample from the
  shuffle buffer is popped, and a new sample is pushed into the shuffle buffer
  at this vacant location.

  Args:
    path: A string of the path pointing to the directory that contains the
      pretraining dataset in the format of balanced parquet shards.
    local_rank: The local rank ID (on this node) of the current pretraining
      process.
    shuffle_buffer_size: The size of the shuffle buffer.
    shuffle_buffer_warmup_factor: At the beginning, the shuffle buffer is empty.
      Therefore, in order to fill the shuffle buffer, at each iteration, more
      samples need to be pushed into the shuffle buffer than being popped out
      of. This factor indicates how many samples is pushed into the shuffle
      buffer per 1 sample being popped out of the shuffle buffer, until the
      shuffle buffer is full.
    tokenizer_class: The HuggingFace tokenizer class for BERT pretraining.
    vocab_file: The path to a vocab file, or the name of a pretrained model
      registered on huggingface.co (e.g., 'bert-large-uncased') of which the
      vocab file is downloaded.
    tokenizer_kwargs: The arguments to the tokenizer class.
    data_loader_class: The class of the DataLoader.
    data_loader_kwargs: The arguments to the DataLoader class.
    mlm_probability: The probability for masking tokens in the masked language
      modeling task (in BERT pretraining).
    base_seed: A base seed value on which other seeds used in the DataLoader are
      based.
    log_dir: The path to a directory to store the logs from the LDDL DataLoader.
    log_level: The logging verbose level.
    return_raw_samples: If True, returns the raw string pairs instead of token
      indices.
    start_epoch: The epoch number to start from. An epoch is defined as going
      through every sample in a dataset once.
    sequence_length_alignment: To get the input tensors of token indices, each
      sequence in a batch will only be padded to the longest sequence in this
      batch. However, certain hardware features might prefer the shapes of the
      input tensors to meet certain conditions. For example, it's better for the
      Tensor Core on NVIDIA GPUs if the dimensions of the input tensors are
      divisible by 8. Therefore, this argument is an alignment factor such that
      the sequences in a batch will be padded to the first sequence length
      larger than the longest sequence in this batch and also divisible by this
      alignment factor.
    ignore_index: The label value for the unmasked tokens in the language
      modeling task (in BERT pretraining).

  Returns:
    A PyTorch DataLoader that, in each iteration, yield:
    - If return_raw_samples is False, a dict of 5 key-value pairs which are the
      necessary input for BERT pretraining:
      {
        'input_ids': a torch.Tensor of size [batch_size, sequence_length],
        'token_type_ids': a torch.Tensor of size [batch_size, sequence_length],
        'attention_mask': a torch.Tensor of size [batch_size, sequence_length],
        'labels': a torch.Tensor of size [batch_size, sequence_length],
        'next_sentence_labels': a torch.Tensor of size [batch_size],
      }
    - If return_raw_samples is True, a list of the following lists:
      [
        strings of the first sequences in the sequence pairs,
        strings of the second sequences in the sequence pairs,
        bools that indicate whether the second sequences are the next sequences
          for the first sequences,
        numpy.ndarrays of positions of the masked tokens for the masked language
          modeling task (only exists if static masking is enabled),
        strings of space-seperated labels of the masked tokens for the masked
          language modeling task (only exists if static masking is enabled),
      ]

  Examples:
    train_dataloader = lddl.torch.get_bert_pretrain_data_loader(
      input_dir,
      local_rank=local_rank,
      vocab_file=vocab_file,
      data_loader_kwargs={
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
      },
      log_level=logging.WARNING,
      start_epoch=start_epoch,
    )

    for epoch in range(start_epoch, start_epoch + epochs):
      for i, batch in enumerate(train_dataloader):
        prediction_scores, seq_relationship_score = model(
          input_ids=batch['input_ids'].to(device),
          token_type_ids=batch['token_type_ids'].to(device),
          attention_mask=batch['attention_mask'].to(device),
      )
      loss = criterion(
          prediction_scores,
          seq_relationship_score,
          batch['labels'].to(device),
          batch['next_sentence_labels'].to(device),
      )
      ...
  """
  assert isinstance(path, str)
  assert isinstance(shuffle_buffer_size, int) and shuffle_buffer_size > 0
  assert (isinstance(shuffle_buffer_warmup_factor, int) and
          shuffle_buffer_warmup_factor > 0)
  assert tokenizer_class in {
      transformers.BertTokenizerFast, transformers.BertTokenizer
  }
  assert isinstance(vocab_file, str)
  assert isinstance(tokenizer_kwargs, dict)
  assert data_loader_class in {DataLoader}
  assert isinstance(data_loader_kwargs, dict)
  assert isinstance(mlm_probability, (int, float)) and 0 <= mlm_probability <= 1
  assert isinstance(base_seed, int)
  assert log_dir is None or isinstance(log_dir, str)
  assert log_level in {
      logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING,
      logging.ERROR, logging.CRITICAL
  }
  assert isinstance(return_raw_samples, bool)
  assert isinstance(start_epoch, int)

  local_rank = dist.get_rank(process_group)

  if os.path.isfile(vocab_file):
    tokenizer = tokenizer_class(vocab_file, **tokenizer_kwargs)
  else:
    tokenizer = tokenizer_class.from_pretrained(vocab_file, **tokenizer_kwargs)

  def _batch_preprocess(batch):
    with torch.no_grad():
      encoded_inputs = _to_encoded_inputs(
          batch,
          tokenizer,
          sequence_length_alignment=sequence_length_alignment,
          ignore_index=ignore_index,
      )
      if 'special_tokens_mask' in encoded_inputs:  # Dynamic masking.
        special_tokens_mask = encoded_inputs.pop('special_tokens_mask', None)
        (encoded_inputs['input_ids'], encoded_inputs['labels']) = _mask_tokens(
            encoded_inputs['input_ids'],
            special_tokens_mask=special_tokens_mask,
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            ignore_index=ignore_index,
        )
    return encoded_inputs

  logger = DatasetLogger(
      log_dir=log_dir,
      node_rank=get_node_rank(nproc_per_node=get_nproc_per_node(local_rank)),
      local_rank=local_rank,
      log_level=log_level,
  )

  dataset_kwargs = {
      'shuffle_buffer_size': shuffle_buffer_size,
      'shuffle_buffer_warmup_factor': shuffle_buffer_warmup_factor,
      'base_seed': base_seed,
      'logger': logger,
      'start_epoch': start_epoch,
      'process_group': process_group
  }

  extra_collate = data_loader_kwargs.get('collate_fn', lambda x: x)
  if not return_raw_samples:
    data_loader_kwargs['collate_fn'] = lambda batch: extra_collate(
        _batch_preprocess(batch))

  # Find all the parquet file paths and figure out whether it is binned or
  # un-binned.
  all_file_paths = get_all_parquets_under(path)
  bin_ids = get_all_bin_ids(all_file_paths)
  if len(bin_ids) > 0:
    data_loader = BertPretrainBinned(
        [
            data_loader_class(
                BertPretrainDataset(
                    get_file_paths_for_bin_id(all_file_paths, bin_id),
                    **dataset_kwargs,
                ),
                **data_loader_kwargs,
            ) for bin_id in bin_ids
        ],
        base_seed=base_seed,
        start_epoch=start_epoch,
        logger=logger,
    )
  else:  # un-binned
    data_loader = data_loader_class(
        BertPretrainDataset(all_file_paths, **dataset_kwargs),
        **data_loader_kwargs,
    )

  return data_loader
