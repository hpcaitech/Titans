import torch

from torch import nn
from colossalai import nn as col_nn
from titans.layer.block import TransformerEncoderLayer, TransformerEncoder, \
                               TransformerDecoderLayer, TransformerDecoder


class Transformer(nn.Module):

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = col_nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = col_nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer,
                                          num_decoder_layers,
                                          decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos=pos_embed)

        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2)
