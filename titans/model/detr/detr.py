import math
from typing import Callable

import torch
from colossalai import nn as col_nn
from colossalai.nn.layer.utils import CheckpointModule
from torch import dtype, nn
from torchvision.models import resnet50

from titans.layer.embedding import ViTEmbedding
# from titans.layer.head import DeTrHead
from titans.layer.mlp import DeTrMLP
from titans.layer.block import DeTrEncoder, DeTrDecoder
from titans.decorator import no_support

__all__ = [
    'DeTr',
    'detr_1',
]


@no_support(['sp', 'moe'])
class DeTr(nn.Module):

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 91,
                 num_encoder_layer: int = 6,
                 num_decoder_layer: int = 6,
                 num_heads: int = 12,
                 num_queries: int = 100,
                 hidden_size: int = 256,
                 mlp_ratio: int = 4,
                 attention_dropout: float = 0.,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 activation: Callable = nn.functional.gelu,
                 representation_size: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch'):
        super().__init__()

        # self.embed = ViTEmbedding(img_size=img_size,
        #                           patch_size=patch_size,
        #                           in_chans=in_chans,
        #                           embedding_dim=hidden_size,
        #                           dropout=dropout,
        #                           dtype=dtype,
        #                           init_method=init_method)

        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_size, 1)

        # stochastic depth decay rule
        dpr1 = [x.item() for x in torch.linspace(0, drop_path, num_encoder_layer)]
        self.blocks1 = nn.ModuleList([
            DeTrEncoder(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                drop_path=dpr1[i],
                activation=activation,
                dtype=dtype,
                bias=bias,
                checkpoint=checkpoint,
                init_method=init_method,
            ) for i in range(num_encoder_layer)
        ])

        dpr2 = [x.item() for x in torch.linspace(0, drop_path, num_decoder_layer)]
        self.blocks2 = nn.ModuleList([
            DeTrDecoder(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                drop_path=dpr2[i],
                activation=activation,
                dtype=dtype,
                bias=bias,
                checkpoint=checkpoint,
                init_method=init_method,
            ) for i in range(num_decoder_layer)
        ])

        self.norm = col_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)
        
        self.class_embed = nn.Linear(hidden_size, num_classes + 1)
        self.bbox_embed = DeTrMLP(hidden_size, hidden_size, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_size)

        self.query_pos = nn.Parameter(torch.rand(100, hidden_size))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_size // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_size // 2))

    def forward(self, x):
        x = self.backbone(x)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        for block in self.blocks1:
            memory = block(pos + h.flatten(2).permute(2, 0, 1))
        print('memory',memory.size())
        print('self.query_pos.unsqueeze(1)',self.query_pos.unsqueeze(1).size())
        for block in self.blocks2:
            x = block(self.query_pos.unsqueeze(1), memory)

        x = self.norm(x)
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        # return out # not dict 
        return outputs_class # temp
 
        


def _create_detr_model(**model_kwargs):
    model = DeTr(**model_kwargs)
    return model


def detr_1(**kwargs):
    model_kwargs = dict(img_size=32, patch_size=4, hidden_size=256, depth=7, num_heads=4, mlp_ratio=2, num_classes=10, **kwargs)
    return _create_detr_model(**model_kwargs)

