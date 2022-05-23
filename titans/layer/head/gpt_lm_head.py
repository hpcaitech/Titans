from torch import dtype, nn

from colossalai import nn as col_nn


class GPTLMHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 embedding_layer=None,
                 bias: bool = False,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.dense = col_nn.Classifier(hidden_size, vocab_size, embedding_layer.word_embedding_weight, bias=bias, dtype=dtype)

    @property
    def weight(self):
        return self.dense.weight

    def forward(self, x):
        # the size of x before dense is (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        # the size of x after dense is (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        x = self.dense(x)
        return x
