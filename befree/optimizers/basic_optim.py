from torch.optim import Adam, SGD, LBFGS

def get_adam(params, config):
    adam_params = ['lr', 'betas', 'eps', 'weight_decay']
    adam_params = {p: config[p] for p in adam_params if p in config}
    return Adam(params, **adam_params)
    
def get_momentum(params, config):
    momentum_params = ['lr', 'momentum', 'weight_decay', 'nesterov']
    momentum_params = {p: config[p] for p in momentum_params if p in config}
    if 'momentum' not in momentum_params or ('momentum' in momentum_params and momentum_params['momentum'] == 0):
        print("The config value of momentum in Momentum optimizer is 0. The value is forced to default 0.9")
        momentum_params['momentum'] = 0.9
    return SGD(params, **momentum_params)

def get_sgd(params, config):
    sgd_params = ['lr', 'weight_decay'] #momentum is always 0
    sgd_params = {p: config[p] for p in sgd_params if p in config}
    return SGD(params, **sgd_params)

def get_lbfgs(params, config):
    lbfgs_params = ['lr', 'max_iter', 'max_eval', 'tolerance_grad', 'tolerance_change', 'history_size']
    lbfgs_params = {p: config[p] for p in lbfgs_params if p in config}
    return LBFGS(params, **lbfgs_params)