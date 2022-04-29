import torch
from torch import dtype, nn

from colossalai import nn as col_nn
from colossalai.utils import get_current_device


class GPTEmbedding(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 max_position_embeddings: int,
                 num_tokentypes: int = 0,
                 padding_idx: int = None,
                 dropout: float = 0.,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.word_embeddings = col_nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, dtype=dtype)
        self.position_embeddings = col_nn.Embedding(max_position_embeddings, embedding_dim, dtype=dtype)
        if num_tokentypes > 0:
            self.tokentype_embeddings = col_nn.Embedding(num_tokentypes, embedding_dim, dtype=dtype)
        else:
            self.tokentype_embeddings = None
        self.dropout = col_nn.Dropout(dropout)

    @property
    def word_embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, input_ids, position_ids=None, tokentype_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            bs = input_ids.size(0)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=get_current_device()).unsqueeze(0)
            position_ids = position_ids.repeat(bs, 1)
        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        if self.tokentype_embeddings is not None and tokentype_ids is not None:
            x = x + self.tokentype_embeddings(tokentype_ids)
        x = self.dropout(x)

        return x