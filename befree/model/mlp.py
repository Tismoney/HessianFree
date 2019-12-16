from torch import nn
import torch.nn.functional as F
from itertools import count
from collections import OrderedDict

def get_mlp(config):
    if 'layers_size' in config:
        return MLP(config['layers_size'])
    else:
        return MLP()


def make_mlp(layers_size):
    layers = []
    last_layers = len(layers_size) - 1
    for i, in_features, out_features in zip(count(1, 1), layers_size, layers_size[1:]):
        layers.append((f'fc{i}', nn.Linear(in_features, out_features)))
        if i != last_layers:
            layers.append((f'relu{i}', nn.ReLU()))
        else:
            layers.append((f'softmax{i}', nn.LogSoftmax(dim=1)))

    return nn.Sequential(OrderedDict(layers))


class MLP(nn.Module):
    def __init__(self, layers_size = [3 * 32 * 32, 1024, 512, 512, 10]):
        super(MLP, self).__init__()
        self.net = make_mlp(layers_size)

    def forward(self, x):
        x = x.view(-1, 3 * 32*32)
        x = x.net(x)
        return x

