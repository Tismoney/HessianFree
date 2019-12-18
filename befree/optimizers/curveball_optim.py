import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import grad
import numpy as np

class CurveBall(Optimizer):

  def __init__(self, params, lr=None, momentum=None, lambd=1.0):
      defaults = dict(lr=lr, momentum=momentum, lambd=lambd)
      super().__init__(params, defaults)

  def fmad(self, predictions, parameters, zs):
    v = torch.ones_like(predictions, requires_grad=True)
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
    momentum = group["momentum"]
    lambd = group["lambd"]

    state = self.state
    for p in params:
      if p not in state:
        state[p] = {'z': torch.zeros_like(p)}
    
    predictions = model_predict()
    loss = loss_func(predictions)

    zs = [state[p]['z'] for p in params]
    (Jz,) = self.fmad(predictions,  params, zs)
    (Jl,) = grad(loss, predictions, create_graph=True)
    Jl_d = Jl.detach()
    (Hl_Jz,) = grad(Jl, predictions, grad_outputs=Jz, retain_graph=True)
    delta_zs = grad(predictions, params, grad_outputs=(Hl_Jz + Jl_d), retain_graph=True)

    for (z, dz) in zip(zs, delta_zs):
        dz.data.add_(lambd, z)

    # autoparams
    # compute J^T * delta_zs
    (Jdeltaz,) = self.fmad(predictions, params, delta_zs)  # equivalent but slower

    # project result by loss hessian (using 2nd-order gradients)
    (Hl_Jdeltaz,) = grad(Jl, predictions, grad_outputs=Jdeltaz)

    # solve 2x2 linear system: [rho, -beta]^T = [a11, a12; a12, a22]^-1 [b1, b2]^T.
    # accumulate components of dot-product from all parameters, by first aggregating them into a vector.
    z_vec = torch.cat([z.flatten() for z in zs])
    dz_vec = torch.cat([dz.flatten() for dz in delta_zs])

    a11 = lambd * (dz_vec * dz_vec).sum() + (Jdeltaz * Hl_Jdeltaz).sum()
    a12 = lambd * (dz_vec * z_vec).sum() + (Jz * Hl_Jdeltaz).sum()
    a22 = lambd * (z_vec * z_vec).sum() + (Jz * Hl_Jz).sum()

    b1 = (Jl_d * Jdeltaz).sum()
    b2 = (Jl_d * Jz).sum()

    # item() implicitly moves to the CPU
    A = torch.tensor([[a11.item(), a12.item()], [a12.item(), a22.item()]])
    b = torch.tensor([[b1.item()], [b2.item()]])
    auto_params = A.pinverse() @ b

    lr = auto_params[0].item()
    momentum = -auto_params[1].item()

    for (p, z, dz) in zip(params, zs, delta_zs):
        z.data.mul_(momentum).add_(-lr, dz)  # update state
        p.data.add_(z)  # update parameter

    predictions = model_predict()
    loss = loss_func(predictions)
    return loss, predictions



def get_curve_ball(params, config):
    curve_params = ["lr", "momentum", "lambd"]
    curve_params = {p: config[p] for p in curve_params if p in config}
    return CurveBall(params, **curve_params)