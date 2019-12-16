
import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import grad
import numpy as np

class Newton(Optimizer):


  def __init__(self, params, lr=1):
    
    defaults = dict(lr=lr)
    super().__init__(params, defaults)

  def eval_hessian(self, loss_grad, params):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1

    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = grad(g_vector[idx], params, create_graph=True, retain_graph=True, allow_unused=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian.cpu().data


  def step(self, model_predict, loss_func):
    ##only one param group is supported
    if len(self.param_groups) != 1:
        raise Exception("only one group is allowed")

    group = self.param_groups[0]
    params = group['params']
    lr = group["lr"]
    predictions = model_predict()
    loss = loss_func(predictions)
    loss.retain_grad()
    loss.backward(retain_graph=True, create_graph=True)
    grads = []
    for param in params:
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    hessian = self.eval_hessian(grads, params)
    updatae = hessian.inverse() @ grads
    with torch.no_grad():
        prev_size = 0
        for i, param in enumerate(params):
            size = param.size()
            len_ = param.view(-1).size(0)
            params[i] -= updatae[prev_size: len_].reshape(size)
            prev_size += len_
    predictions = model_predict()   
    loss = loss_func(predictions)
    return loss.item()



def get_newton(params, config):
    sgd_params = ['lr'] 
    sgd_params = {p: config[p] for p in sgd_params if p in config}
    return Newton(params, **sgd_params)