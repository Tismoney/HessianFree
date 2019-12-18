import torch
from torch.autograd import grad
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch.optim.optimizer import Optimizer

from befree.utils.cg import CG


class SimplifiedHessian(Optimizer):

    def __init__(self, params, lr=None, momentum=None, lambd=10.0):

        defaults = dict(lr=lr, momentum=momentum, lambd=lambd)
        super(SimplifiedHessian, self).__init__(params, defaults)

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
        flat_params = parameters_to_vector(params)

        J = grad(loss, params, create_graph=True)
        J = parameters_to_vector(J)
        J_d = J.detach()
        J_reshaped = J.reshape(1, 1, -1)
        
        z0 = J_d.reshape(1, 1, -1).neg()
        R = 10

        def A_bmm(x):
            x_ = x.view_as(J)
            (Hl_Jz,) = grad(J, predictions, grad_outputs=x_, retain_graph=True)
            delta_zs = grad(predictions, params, grad_outputs=Hl_Jz, retain_graph=True)  
            return parameters_to_vector(delta_zs)

        for i in range(R):
            cg = CG(A_bmm, maxiter=len(flat_params) * 2, verbose=False)
            z0 = cg(J_reshaped.neg(), z0)
            residual = A_bmm(z0) + J
            print('residual norm: ', torch.norm(residual))
            if torch.norm(residual) > 1e2 and i != R - 1:
                z0 = torch.zeros_like(J_reshaped)

        print('residual norm: ', torch.norm(residual))
        z0 = z0[0, 0]
        vector_to_parameters(flat_params + z0, params)
        predictions = model_predict()
        loss = loss_func(predictions)
        return loss, predictions


def get_simplified_hessian(params, config):
    simpl_params = ["lr", "momentum", "lambd"]
    simpl_params = {p: config[p] for p in simpl_params if p in config}
    return SimplifiedHessian(params, **simpl_params)
