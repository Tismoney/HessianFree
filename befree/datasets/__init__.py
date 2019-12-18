from .cifar import *
from .mnist import *
from .rosen import *

datasets_names = ['mnist', 'cifar10', 'rosen']

def get_dataset(config):

    assert config['name'] in datasets_names

    if config['name'] == 'mnist':
        return get_mnist(config)
    elif config['name'] == 'cifar10':
        return get_cifar(config)
    elif config['name'] == 'rosen':
        return get_rosen(config)

