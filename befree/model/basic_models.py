import torchvision.models as models
from .utils import *

def get_resnet18(config):
    return models.resnet18()
    
def get_vgg16(config):   
    return models.vgg16()

def get_cnn(config):
    return make_net(cnn_cfg=config['cnn'], fc_cfg=config['fc'])

def get_mlp(config):
    return make_net(cnn_cfg=None, fc_cfg=config['fc'])
