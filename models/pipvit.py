import inspect
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# local imports
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, grandparentdir)
torch.multiprocessing.set_sharing_strategy('file_system')

from feature_extractor_vit import base_architecture_to_features

class BatchNormRelu(nn.Module):
    def __init__(self, num_features):
        super(BatchNormRelu, self).__init__()
        # Assuming input has 2048 channels
        self.batch_norm = nn.BatchNorm2d(num_features)  # BatchNorm2d for 2048 feature maps
        self.relu = nn.ReLU(inplace=True)  # ReLU activation

    def forward(self, x):
        x = self.batch_norm(x)  # Apply BatchNorm2d
        x = self.relu(x)         # Apply ReLU activation
        return x


class PiPViT(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 num_features: int,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 _norm_layer: nn.Module,
                 classification_layer: nn.Module,
                 ):
        super().__init__()
        assert num_classes > 0
        self._num_features = num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier
        self.feature_norm = _norm_layer

    def forward(self, xs, inference=False):
        features = self._net(xs) # feature size torch.Size([1, 144, 768])
        # apply normalization on each feature vector
        features = features.permute(0, 2, 1)  # torch.Size([1, 768, 196])
        features = features.reshape(features.shape[0], features.shape[1], self._net.patch_embed.grid_size[0],
                      self._net.patch_embed.grid_size[1])  # torch.Size([16, 768, 12, 12]) for the pipnet it is out torch.Size([32, 2048, 28, 28])
        features = self.feature_norm(features) # torch.Size([16, 768, 12, 12])

        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)
        if inference:
            clamped_pooled = torch.where(pooled < 0.1, 0.,
                                         pooled)  # during inference, ignore all prototypes that have 0.1 similarity or lower
            out = self._classification(clamped_pooled)  # shape (bs, num_classes)
            # print(f'out {out.shape}')
            return out, proto_features, clamped_pooled
        else:
            out = self._classification(pooled)  # shape (bs, num_classes)
            return out, proto_features, pooled



# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,), requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.relu(self.weight), self.bias)


def get_network(num_classes: int, model_name, imgnet_pretrained, num_features=0, bias=False, img_size=384):
    print('Image size is set to ', img_size, flush=True)
    features = base_architecture_to_features[model_name](pretrained=imgnet_pretrained, img_size=img_size)
    first_add_on_layer_in_channels = features.num_features
    if num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        print("Number of prototypes: ", num_prototypes, flush=True)
        add_on_layers = nn.Sequential(
            nn.Softmax(dim=1),
            # softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1
        )
    else:
        num_prototypes = num_features
        print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes,
              ". Extra 1x1 conv layer added. Not recommended.", flush=True)
        add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, kernel_size=1, stride=1,
                      padding=0, bias=True),
            nn.Softmax(dim=1),
            # softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1
        )
    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool2d(output_size=(1, 1)),  # outputs (bs, ps,1,1)
        nn.Flatten()  # outputs (bs, ps)
    )

    if bias:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)
    feature_norm_layer = BatchNormRelu(num_features=num_prototypes)

    return features, add_on_layers, pool_layer, classification_layer, num_prototypes, feature_norm_layer






















