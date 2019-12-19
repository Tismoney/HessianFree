from ..model import get_model
from ..datasets import get_dataset
from ..optimizers import get_optimizer
from ..config import get_config
from .train import train

from torch import nn
from torch.nn import functional as F

import numpy as np

def fit_model(config, use_gpu=True, print_model=False, print_test_epoch=50, save_test=True):
    config = get_config(config)
    train_loader, test_loader = get_dataset(config['dataset'])
    model = get_model(config['model'])
    optim = get_optimizer(model.parameters(), config['optimizer'])
    criterion = get_loss(config['loss'])
    metrics = get_metrics(config['metrics'])
    
    
    print('-'*20)
    print(config['optimizer']['name'])
    if print_model: print(model)
    stats, times_per_iter = train(model, train_loader, test_loader,
                  optim, criterion, metrics,
                  config['optimizer']['num_epochs'], use_gpu=use_gpu,
                  print_test_epoch = print_test_epoch, save_test=save_test)
    
    print(f'Times per iter {np.mean(times_per_iter) * 1e3 :.3f} ms ± {np.std(times_per_iter) * 1e6 :.3f} µs')
    return (config['optimizer']['name'], stats)


def get_loss(config):
    if config['name'] == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif config['name'] == 'MSE':
        return nn.MSELoss()


def torch_accuracy(predictions, targets):
    predictions = predictions.argmax(1, keepdim=True)
    targets = targets.view_as(predictions)
    return (targets == predictions).sum().double() / targets.size(0)

def get_metrics(config):
    metrics = []
    if 'accuracy' in config['names']:
        metrics.append(('accuracy', torch_accuracy))
    return dict(metrics)