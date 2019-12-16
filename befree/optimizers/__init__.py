from .basic_optim import *

optimizer_name = ['Adam', 'SGD', 'Momentum']

def get_optimizer(params, config):
    
    assert config['name'] in optimizer_name
    
    if config['name'] == 'Adam':
        return get_adam(params, config)
    elif config['name'] == 'SGD':
        return get_sgd(params, config)
    elif config['name'] == 'Momentum':
        return get_momentum(params, config)
    
    