import torch
import torch.nn as nn
import numpy as np
import math


class An(nn.Module):

    def __init__(self, num_linear, input_features, output_features):
        super(An, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(num_linear, input_features, output_features))
        self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = torch.matmul(x.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            out += self.bias
        return out


class A(nn.Module):
    def __init__(self, dim):
        super(A, self).__init__()
        self.layer_dim = dim
        self.layers = [An(dim[0], dim[i + 1], dim[i + 2]) for i in range(len(dim) - 2)]
        self.weights = [layer.weight for layer in self.layers]

    def forward(self, x):
        for layer in self.layers:
            x = layer(torch.sigmoid(x))
        return x


class NN_Dag(nn.Module):
    def __init__(self, dims):
        super(NN_Dag, self).__init__()
        d = dims[0]
        self.dims = dims
        self.a1 = nn.Linear(d, d * dims[1])
        self.non_linear = False
        if len(dims) > 2:
            self.non_linear = True
            self.an = A(dims)

    def forward(self, x):
        x = self.a1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        if self.non_linear:
            x = self.an(x)
        return x.squeeze(dim=2)

    def h_func(self):
        d = self.dims[0]
        a1_weight = self.a1.weight.view(d, -1, d)
        A = torch.sum(a1_weight ** 2, dim=1).t()
        h = torch.trace(torch.matrix_exp(A)) - d
        return h

    def l2_reg(self):
        reg = torch.sum(self.a1.weight ** 2)
        if self.non_linear:
            for ai in self.an.weights:
                reg += torch.sum(ai ** 2)
        return reg

    @torch.no_grad()
    def get_W(self):
        d = self.dims[0]
        a1_weight = self.a1.weight.view(d, -1, d)
        A = torch.sum(a1_weight ** 2, dim=1).t()
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()
        return W


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def rho_h_update(model, X, lambda2, rho, alpha, h, rho_max):
    h_new = None
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=1500)
    X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            primal_obj = loss + penalty + l2_reg
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.5 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def func(X, lambda2: float = 0.01, max_iter: int = 100, h_tol: float = 1e-6, rho_max: float = 1e6, w_threshold: float = 0.3,
         extra_layer_dim=None):
    if extra_layer_dim is None:
        extra_layer_dim = []
    d=X.shape[1]
    model = NN_Dag(dims=[d]+extra_layer_dim +[1])
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = rho_h_update(model, X, lambda2, rho, alpha, h, rho_max)
        W_est = model.get_W()
        if h <= h_tol or rho >= rho_max or np.isnan(W_est).any():
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

X=None
W_est = func(X,extra_layer_dim = [10])#在extra_layer_dim里设置额外的神经网络结构
