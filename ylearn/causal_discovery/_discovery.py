"""
See (https://arxiv.org/pdf/1803.01422.pdf) for reference.
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ylearn.utils import drop_none, set_random_state, logging
from ._base import BaseDiscovery

_is_cuda_available = torch.cuda.is_available()

logger = logging.get_logger(__name__)


class L(nn.Module):
    def __init__(self, num_linear, input_features, output_features, dtype=None, device=None):
        super().__init__()

        options = drop_none(dtype=dtype, device=device)

        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.empty((num_linear, input_features, output_features), **options))
        self.bias = nn.Parameter(torch.empty((num_linear, output_features), **options))
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
    def __init__(self, dim, dtype=None, device=None):
        super(A, self).__init__()
        self.layer_dim = dim
        self.layers = [L(dim[0], dim[i + 1], dim[i + 2], dtype=dtype, device=device) for i in range(len(dim) - 2)]
        self.weights = [layer.weight for layer in self.layers]

    def forward(self, x):
        for layer in self.layers:
            x = layer(torch.sigmoid(x))
        return x


class DagNet(nn.Module):
    def __init__(self, dims, dtype=None, device=None):
        super().__init__()
        d = dims[0]
        self.dims = dims
        self.a1 = nn.Linear(d, d * dims[1], **drop_none(dtype=dtype, device=device))
        self.non_linear = False
        if len(dims) > 2:
            self.non_linear = True
            self.an = A(dims, dtype=dtype, device=device)

    def forward(self, x):
        x = self.a1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        if self.non_linear:
            x = self.an(x)
        return x.squeeze(dim=2)

    def h_func(self):
        d = self.dims[0]
        a1_weight = self.a1.weight.view(d, -1, d)
        s = torch.sum(a1_weight ** 2, dim=1).t()
        h = torch.trace(torch.matrix_exp(s)) - d
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

    @torch.no_grad()
    def is_weight_nan(self):
        # d = self.dims[0]
        # a1_weight = self.a1.weight.view(d, -1, d)
        # A = torch.sum(a1_weight ** 2, dim=1).t()
        # W = torch.sqrt(A)
        # W = W.cpu().detach().numpy()
        return torch.isnan(self.a1.weight).any().cpu().detach().numpy()


class CausalDiscovery(BaseDiscovery):
    def __init__(self, hidden_layer_dim=None,
                 lambdaa: float = 0.01, h_tol: float = 1e-6, rho_max: float = 1e6,
                 scale='auto',
                 device=None, dtype=None, random_state=None):
        self.hidden_layer_dim = hidden_layer_dim
        self.lambdaa = lambdaa
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.scale = scale
        self.device = device
        self.dtype = dtype
        self.random_state = random_state

    @staticmethod
    def _squared_loss(output, target):
        n = target.shape[0]
        loss = 0.5 / n * torch.sum((output - target) ** 2)
        return loss

    def _rho_h_update(self, model, optimizer, X, rho, alpha, h):
        h_new = None
        # X_torch = torch.from_numpy(X)
        # X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        while rho < self.rho_max:
            def closure():
                optimizer.zero_grad()
                X_hat = model(X)
                loss = self._squared_loss(X_hat, X)
                h_val = model.h_func()
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                l2_reg = 0.5 * self.lambdaa * model.l2_reg()
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

    def _get_scaler(self):
        if self.scale is None or self.scale is False:
            return None
        elif self.scale is True or self.scale in {'auto', 'minmax'}:
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()
        elif self.scale in {'std', 'standard'}:
            from sklearn.preprocessing import StandardScaler
            return StandardScaler()
        elif hasattr(self.scale, 'fit_transform'):
            import copy
            return copy.copy(self.scale)
        else:
            raise ValueError(f'Failed to create scaler {self.scale}')

    def __call__(self, data, *, return_dict=False, threshold=0.01,
                 # optimizer='lbfgs',
                 epoch=100, lr=0.01, max_iter=1500, verbose=False,
                 **kwargs):
        assert isinstance(data, (np.ndarray, pd.DataFrame))

        set_random_state(self.random_state)

        if isinstance(data, pd.DataFrame):
            columns = data.columns.tolist()
            data = data.values
        else:
            columns = None

        scaler = self._get_scaler()
        if scaler is not None:
            data = scaler.fit_transform(data)

        device = self.device
        if device is None or device == 'auto':
            device = 'cuda' if _is_cuda_available else 'cpu'

        dtype = self.dtype
        if dtype is None:
            # dtype = torch.float32 if str(device) == 'cuda' else torch.float64
            dtype = torch.float64
        hidden = self.hidden_layer_dim if self.hidden_layer_dim is not None else []
        dims = [data.shape[1]] + hidden + [1]

        logger.info(f'learning data{data.shape} with device={device}, dtype={dtype}, dims={dims}')

        model = DagNet(dims=dims, dtype=dtype, device=device)
        data_t = torch.tensor(data, dtype=dtype, device=device)

        pbar = tqdm(total=epoch, leave=False, desc='learning..') if verbose else None
        rho, alpha, h = 1.0, 0.0, np.inf
        for i in range(epoch):
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                **drop_none(lr=lr, max_iter=max_iter)
            )
            rho, alpha, h = self._rho_h_update(model, optimizer, data_t, rho, alpha, h)
            if pbar is not None:
                pbar.update(i)
                pbar.desc = f'rho={rho}, alpha={alpha}, h={h}'
            # W_est = model.get_W()
            # if h <= self.h_tol or rho >= self.rho_max or np.isnan(W_est).any():
            #     break
            if h <= self.h_tol or rho >= self.rho_max or model.is_weight_nan():
                break

        matrix = model.get_W()

        logger.info(f'trim causal matrix to DAG, threshold={threshold}.')
        if threshold is not None:
            matrix[np.abs(matrix) < threshold] = 0
        matrix = self._to_dag(matrix)

        if columns is not None:
            matrix = pd.DataFrame(matrix, columns=columns, index=columns)

        if return_dict:
            matrix = self.matrix2dict(matrix)

        return matrix

    @staticmethod
    def _to_dag(matrix):
        for i in range(matrix.shape[0]):
            for c in range(i + 1, matrix.shape[0]):
                if abs(matrix[i, c]) < abs(matrix[c, i]):
                    matrix[i, c] = 0.
                else:
                    matrix[c, i] = 0.
            matrix[i, i] = 0.

        matrix = BaseDiscovery.trim_cycle(matrix)
        return matrix
