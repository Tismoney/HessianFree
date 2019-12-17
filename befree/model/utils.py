from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def make_cnn(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]    
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=5, padding=2)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_fc(cfg, batch_norm=False):
    layers = []
    in_featues = cfg[0]
    for v in cfg[1:]: 
        fc = nn.Linear(in_featues, v)
        if batch_norm:
            layers += [fc, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
        else:
            layers += [fc, nn.ReLU(inplace=True)]
            in_featues = v
    layers = layers[:-1]
    return nn.Sequential(*layers)

def make_net(cnn_cfg=None, fc_cfg=None):
    blocks = []
    if cnn_cfg:
        blocks += [make_cnn(cnn_cfg)]
    if fc_cfg:
        blocks += [Flatten(), make_fc(fc_cfg)]    
    return nn.Sequential(*blocks)