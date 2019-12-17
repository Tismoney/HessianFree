import torch
from torch.nn import functional as F
from ..optimizers.basic_optim import LBFGS
import numpy as np
from time import time

## Will be deleted in the future
class CustomOptimizer: 
    None

def train(model, train_loader, optimizer, criterion, metrics, epoch):
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
    stats = {key: [] for key in ['loss'] + list(metrics.keys())}
    for epoch_i in range(1, epoch + 1):
        start = time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if torch.cuda.is_available():           
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs.requires_grad = True

            if isinstance(optimizer, CustomOptimizer):
                (loss, predictions) = optimizer.step(model, criterion)
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

        print_stat = f"[{epoch_i}/{epoch}] epoch | Loss: {np.mean(stats['loss'][-25]):.3f} | "
        for name in metrics.keys():
            print_stat += f"{name} : {np.mean(stats[name][-25:]):.3f} | "
        end = time()
        print_stat += f" time: {end - start:.2f}s"
        print(print_stat)
    return stats