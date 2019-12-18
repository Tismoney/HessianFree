import torch
from torch.nn import functional as F
from ..optimizers import LBFGS, CurveBall, SimplifiedHessian, HessianFree
import numpy as np
from time import time


def train(model, train_loader, optimizer, criterion, metrics, epoch, use_gpu=True):
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
    stats = {key: [] for key in ['loss'] + list(metrics.keys())}
    for epoch_i in range(1, epoch + 1):
        start = time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if use_gpu and torch.cuda.is_available():           
                inputs, targets = inputs.cuda(), targets.cuda()
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

            stats['loss'].append(loss.item())
            with torch.no_grad(): 
                for name, func in metrics.items():
                    res = func(predictions, targets)
                    stats[name].append(res.item())
            
            if isinstance(optimizer, SimplifiedHessian):
                print_stat = f"[{epoch_i}/{epoch}] epoch | [{batch_idx}] batch | Loss: {np.mean(stats['loss'][-1]):.3f} | "
                for name in metrics.keys():
                    print_stat += f"{name} : {np.mean(stats[name][-1]):.3f} | "
                end = time()
                print_stat += f"time: {end - start:.2f}s"
                print(print_stat)

        print_stat = f"[{epoch_i}/{epoch}] epoch | Loss: {np.mean(stats['loss'][-25]):.3f} | "
        for name in metrics.keys():
            print_stat += f"{name} : {np.mean(stats[name][-25:]):.3f} | "
        end = time()
        print_stat += f"time: {end - start:.2f}s"
        print(print_stat)
    return stats