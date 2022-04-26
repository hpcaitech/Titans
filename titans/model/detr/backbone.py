import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from util.misc import NestedTensor, is_main_process
from titans.layer.embedding import PositionEmbeddingSine
from titans.layer.batchnorm import FrozenBatchNorm2d


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # if return_interm_layers:
        #     return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
            self,
            name: str,
            train_backbone: bool,
    # return_interm_layers: bool,
            dilation: bool):
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],
                                                     pretrained=is_main_process(),
                                                     norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels)


class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def _build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    return position_embedding


def build_backbone(args):
    position_embedding = _build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.backbone, train_backbone, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
