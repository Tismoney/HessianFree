import torch
from torch.autograd import grad
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch.optim.optimizer import Optimizer

from befree.utils.cg import CG


class SimplifiedHessian(Optimizer):

    def __init__(self, params, lr=None, momentum=None, lambd=10.0):

        defaults = dict(lr=lr, momentum=momentum, lambd=lambd)
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
        # lr = group["lr"]
        # momentum = group["momentum"]
        # lambd = group["lambd"]
        state = self.state
        for p in params:
            if p not in state:
                state[p] = {'z': torch.zeros_like(p)}

        predictions = model_predict()
        loss = loss_func(predictions)

        (Jl,) = grad(loss, predictions, create_graph=True)
        Jl_d = Jl.detach()
        Jl_reshaped = Jl.reshape(1, 1, -1)
        z0 = Jl_d.reshape(1, 1, -1).neg()
        R = 10

        def A_bmm(x):
            (Hl_Jz,) = grad(Jl, predictions, grad_outputs=x[0, 0], retain_graph=True)
            return Hl_Jz.reshape(1, 1, -1)

        for i in range(R):
            cg = CG(A_bmm, maxiter=len(predictions) * 2, verbose=False)
            z0 = cg(Jl_reshaped.neg(), z0)
            residual = grad(Jl, predictions, grad_outputs=z0[0, 0], retain_graph=True)[0] + Jl
            if torch.norm(residual) > 1e2:
                z0 = torch.zeros_like(Jl_reshaped)

        print('residiual:', grad(Jl, predictions, grad_outputs=z0[0, 0], retain_graph=True)[0] + Jl)
        print('----------------------')
        z0 = z0[0, 0]

        flat_params = parameters_to_vector(params)
        vector_to_parameters(flat_params + z0, params)
        predictions = model_predict()
        loss = loss_func(predictions)
        return loss, predictions


def get_simplified_hessian(params, config):
    simpl_params = ["lr", "momentum", "lambd"]
    simpl_params = {p: config[p] for p in simpl_params if p in config}
    return SimplifiedHessian(params, **simpl_params)
