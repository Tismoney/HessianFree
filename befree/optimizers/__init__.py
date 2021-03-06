from .basic_optim import *
from .newton_optim import *

from .simplified_optim import *
from .curveball_optim import *
from .hessian_free_optim import *
optimizer_name = ['Adam', 'SGD', 'Momentum', 'Newton', 'Curveball', 'SimplifiedHessian', 'HessianFree']

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
    elif config['name'] == 'SimplifiedHessian':
        return get_simplified_hessian(params, config)
    elif config['name'] == 'HessianFree':
        return get_hessian_free_optim(params, config)
    elif config['name'] == 'LBFGS':
        return get_lbfgs(params, config)
    elif config['name'] == 'Curveball':
        return get_curve_ball(params, config)
