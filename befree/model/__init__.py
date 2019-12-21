from .basic_models import *

models_name = ['resnet18', 'vgg16', 'mlp', 'cnn']

def get_model(config):
    
    assert config['name'] in models_name
    
    if config['name'] == 'resnet18':
        return get_resnet18(config)
    elif config['name'] == 'vgg16':
        return get_vgg16(config)
    elif config['name'] == 'mlp':
        return get_net(config)
    elif config['name'] == 'cnn':
        return get_net(config)

    
    