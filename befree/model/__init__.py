from .basic_models import *

models_name = ['resnet18', 'vgg16']

def get_model(config):
    
    assert config.name in models_name
    
    if config.name == 'resnet18':
        model = get_resnet18(config)
    elif config.name == 'vgg16':
        model = get_vgg16(config)
    
    return model
    
    