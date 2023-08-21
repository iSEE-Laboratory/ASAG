
"""
Backbone modules.
"""

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Tuple

from util.misc import NestedTensor, is_main_process

# class FrozenBatchNorm2d(torch.nn.Module):
#     """
#     BatchNorm2d where the batch statistics and the affine parameters are fixed.

#     Copy-paste from torchvision.misc.ops with added eps before rqsrt,
#     without which any other models than torchvision.models.resnet[18,34,50,101]
#     produce nans.
#     """

#     def __init__(self, n):
#         super(FrozenBatchNorm2d, self).__init__()
#         self.register_buffer("weight", torch.ones(n))
#         self.register_buffer("bias", torch.zeros(n))
#         self.register_buffer("running_mean", torch.zeros(n))
#         self.register_buffer("running_var", torch.ones(n))

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         num_batches_tracked_key = prefix + 'num_batches_tracked'
#         if num_batches_tracked_key in state_dict:
#             del state_dict[num_batches_tracked_key]

#         super(FrozenBatchNorm2d, self)._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict,
#             missing_keys, unexpected_keys, error_msgs)

#     def forward(self, x):
#         # move reshapes to the beginning
#         # to make it fuser-friendly
#         w = self.weight.reshape(1, -1, 1, 1)
#         b = self.bias.reshape(1, -1, 1, 1)
#         rv = self.running_var.reshape(1, -1, 1, 1)
#         rm = self.running_mean.reshape(1, -1, 1, 1)
#         eps = 1e-5
#         scale = w * (rv + eps).rsqrt()
#         bias = b - rm * scale
#         return x * scale + bias



class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        xs = self.body(tensor_list)
        out = {}
        for name, x in xs.items():
            out[name] = x
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 frozenbn: bool,
                 pretrained_checkpoint: str):
        # if frozenbn:
        #     backbone = getattr(torchvision.models, name)(
        #         replace_stride_with_dilation=[False, False, dilation],
        #         pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        # else:
        #     backbone = getattr(torchvision.models, name)(
        #         replace_stride_with_dilation=[False, False, dilation], pretrained=is_main_process())
        if frozenbn:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=False, norm_layer=FrozenBatchNorm2d)
        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation], pretrained=False)
        if is_main_process() and pretrained_checkpoint != '':
            state_dict = torch.load(pretrained_checkpoint, 'cpu')
            backbone.load_state_dict(state_dict, False)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone):
        super().__init__(backbone)

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        out = list(xs.values())
        return out


def build_backbone(args):
    train_backbone = True
    return_interm_layers = args.return_interm_layers
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args.frozen_bn, args.pretrained_checkpoint)
    model = Joiner(backbone)
    model.num_channels = backbone.num_channels
    return model
