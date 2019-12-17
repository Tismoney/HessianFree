
import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import grad
import numpy as np
from torch_cg import CG

class SimplifiedHessian(Optimizer):

    def __init__(self, params, lr=1):

        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def fmad(self, predictions, parameters, zs):
        v = torch.zeros_like(predictions, requires_grad=True)
        g = torch.autograd.grad(predictions, parameters, grad_outputs=v, create_graph=True)
        output = torch.autograd.grad(g, v, grad_outputs=zs)
        return output

    def step(self, model_predict, loss_func):
        ##only one param group is supported
        if len(self.param_groups) != 1:
            raise Exception("only one group is allowed")

        group = self.param_groups[0]
        params = group['params']
        lr = group["lr"]
        state = self.state
        for p in params:
            if p not in state:
                state[p] = {'z': torch.zeros_like(p)}

        predictions = model_predict()
        loss = loss_func(predictions)

        zs = [state[p]['z'] for p in params]
        (Jz,) = self.fmad(predictions, params, zs)

        (Jl,) = grad(loss, predictions, create_graph=True)
        Jl_d = Jl.detach()

        (Hl_Jz,) = grad(Jl, predictions, grad_outputs=Jz, retain_graph=True)
        delta_zs = grad(predictions, params, grad_outputs=(Hl_Jz + Jl_d), retain_graph=True)
        R = 10
        for i in range(R):
            (Jz,) = self.fmad(predictions, params, zs)
            (Hl_Jz,) = grad(Jl, predictions, grad_outputs=Jz, retain_graph=True)

        with torch.no_grad():
            for (p, z, dz) in zip(params, zs, delta_zs):
                p.data.add_(z)  # update parameter

        predictions = model_predict()
        loss = loss_func(predictions)
        return loss.item()


def get_simplified_hessian(params, config):
    newton_params = ['lr']
    newton_params = {p: config[p] for p in newton_params if p in config}
    return SimplifiedHessian(params, **newton_params)