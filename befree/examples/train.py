import torch
from torch.nn import functional as F
from ..optimizers import LBFGS, CurveBall, SimplifiedHessian, HessianFree
import numpy as np
from time import time

def test(model, test_loader, criterion, metrics, use_gpu=True):
    with torch.no_grad():
        model.eval()
        if use_gpu and torch.cuda.is_available():
            model = model.cuda()
        
        stats = {key: [] for key in ['test.loss'] + ['test.' + k for k in metrics.keys()]}
        
        for batch_idx, (inputs, targets) in enumerate(test_loader):
                if use_gpu and torch.cuda.is_available():           
                    inputs, targets = inputs.cuda(), targets.cuda()
                
                predictions = model(inputs)
                loss = criterion(predictions, targets).item()
                stats['test.loss'].append(loss)
                
                for name, func in metrics.items():
                    res = func(predictions, targets).item()
                    stats['test.' + name].append(res)
                
        for key in stats.keys():
            stats[key] = np.mean(stats[key])
        
        return stats


def train(model, train_loader, test_loader, optimizer,
          criterion, metrics, epoch, use_gpu=True,
          print_test_epoch=50, save_test=True):
    '''
        Params:
            model: pytorch model
            train_loader: train dataset
            optimizer: torch.optim.Optimizer
            criterion: loss function
            metrics: dict of functions
            epoch: int, num of epoch
    '''
    model.train()
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
    keys = ['train.loss'] + ['train.' + k for k in metrics.keys()]
    if save_test:
        keys += ['test.loss'] + ['test.' + k for k in metrics.keys()]
    stats = {key: [] for key in keys}
    times_per_iter = []
    num_iter = 0
    for epoch_i in range(1, epoch + 1):
        start = time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            num_iter += 1
            if use_gpu and torch.cuda.is_available():           
                inputs, targets = inputs.cuda(), targets.cuda()
            start_iter_time = time()
            # inputs.requires_grad = True
            if isinstance(optimizer, CurveBall) or isinstance(optimizer, SimplifiedHessian):
                model_fn = lambda: model(inputs)
                loss_fn = lambda predictions: criterion(predictions, targets)
                loss, predictions = optimizer.step(model_fn, loss_fn)
            elif isinstance(optimizer, HessianFree):
                def closure():
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                    loss.backward(create_graph=True)
                    return loss, predictions
                loss = optimizer.step(closure)
                predictions = model(inputs)
                
            elif isinstance(optimizer, LBFGS):
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                    if loss.requires_grad:
                        loss.backward()
                    return loss
                loss = optimizer.step(closure)
                predictions = model(inputs)
            else: # standard optimizer
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
            
            end_iter_time = time()
            
            times_per_iter.append(end_iter_time - start_iter_time)
            stats['train.loss'].append(loss.item())
            with torch.no_grad(): 
                for name, func in metrics.items():
                    res = func(predictions, targets)
                    stats['train.' + name].append(res.item())
            
            if save_test:
                if num_iter % print_test_epoch == 0:
                    test_stats = test(model, test_loader, criterion, metrics, use_gpu=use_gpu)
                    for key, val in test_stats.items():
                        stats[key].append(val)
            

        print_stat = f"[{epoch_i}/{epoch}] epoch "
        print_stat += f"| Train Loss: {np.mean(stats['train.loss'][-25]):.3f} | "
        
        for name in metrics.keys():
            print_stat += f"{name} : {np.mean(stats['train.' + name][-25:]):.3f} | "
        end = time()
        print_stat += f"time: {end - start:.2f}s"
        print(print_stat)
    return stats, times_per_iter