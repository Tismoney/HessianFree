from .basic_optim import *
from .newton_optim import *
optimizer_name = ['Adam', 'SGD', 'Momentum', 'Newton']

def get_optimizer(params, config):
    
    assert config['name'] in optimizer_name
    
    if config['name'] == 'Adam':
        return get_adam(params, config)
    elif config['name'] == 'SGD':
        return get_sgd(params, config)
    elif config['name'] == 'Momentum':
        return get_momentum(params, config)
    elif config['name'] == 'Newton':
        return get_newton(params, config)
    
    