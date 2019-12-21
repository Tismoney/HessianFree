import torchvision.models as models
from .utils import *

def get_resnet18(config):
    return models.resnet18(num_classes=config['num_classes'])
    
def get_vgg16(config):   
    return models.vgg16(num_classes=config['num_classes'])

def get_cnn(config):
    return make_net(cnn_cfg=config['cnn'], fc_cfg=config['fc'])

def get_mlp(config):
    return make_net(cnn_cfg=None, fc_cfg=config['fc'])

def get_net(config):
    net_params = {
        'cnn_cfg': config['cnn'] if 'cnn' in config else None,
        'fc_cfg': config['fc'] if 'fc' in config else None,
        'batch_norm': config['batch_norm'] if 'batch_norm' in config else False,
        'fc_bias': config['fc_bias'] if 'fc_bias' in config else True,
        'fc_bias': config['cnn_bias'] if 'cnn_bias' in config else True
    }
    return make_net(**net_params)


