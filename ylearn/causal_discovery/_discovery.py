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

class Golem(nn.Module):
    def __init__(self, dims, equal_variances, lambdaa=1e-3, lambdab=1e-3, dtype=None, device=None):
        super(Golem, self).__init__()
        self.d = dims[0]
        self.n = dims[1]
        self.equal_variances = equal_variances
        self.w_est = torch.zeros(self.d, self.d, dtype=dtype, device=device)
        self.w_est = nn.Parameter(self.w_est, requires_grad=True)
        self.lambdaa = lambdaa
        self.lambdab = lambdab

    def h_func(self):
        h = torch.trace(torch.matrix_exp(self.w_est * self.w_est)) - self.d
        return h

    def _compute_L1_penalty(self):
        return torch.norm(self.w_est, p=1)

    def _compute_likelihood(self, X):
        """Compute (negative log) likelihood in the linear Gaussian case.
        """
        # print(X.dtype, self.w_est.dtype)
        if self.equal_variances:  # Assuming equal noise variances
            return 0.5 * self.d * torch.log(
                torch.square(
                    torch.norm(X - X @ self.w_est)
                )
            ) - torch.slogdet(torch.eye(self.d) - self.w_est)[1]
        else:  # Assuming non-equal noise variances
            return 0.5 * torch.sum(
                torch.log(
                    torch.sum(
                        torch.square(X - X @ self.w_est), dim=0
                    )
                )
            ) - torch.slogdet(torch.eye(self.d) - self.w_est)[1]

    def forward(self, X):  # [n, d] -> [n,d]
        self.likelihood = self._compute_likelihood(X)
        self.L1_penalty = self._compute_L1_penalty()
        self.h = self.h_func()
        self.score = self.likelihood + self.lambdaa * self.L1_penalty + self.lambdab * self.h
        return self.score, self.likelihood, self.h, self.w_est


class GolemDiscovery(BaseDiscovery):
    def __init__(self, lambdaa: float = 0.01,
                 lambdab: float = 0.01, scale='auto', equal_variances=True,
                 device=None, dtype=None, random_state=None):
        super(GolemDiscovery, self).__init__()
        self.scale = scale
        self.dtype = dtype
        self.device = device
        self.lambdaa = lambdaa
        self.lambdab = lambdab
        self.random_state = random_state

    def _update(self, model, optimizer, X):
        def closure():
            optimizer.zero_grad()
            primal_obj, _, _, _ = model(X)
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)
        with torch.no_grad():
            h_new = model.h_func().item()
            score = model.score.item()
            likelihood = model.likelihood.item()
        return h_new, score, likelihood

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

    def __call__(self, data, *, return_dict=False, threshold=0.3,
                 epoch=10000, lr=0.001, verbose=False, **kwargs):
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
            dtype = torch.float64

        dims = [data.shape[1]] + [data.shape[0]]

        logger.info(f'learning data{data.shape} with device={device}, dtype={dtype}, dims={dims}')

        model = Golem(dims=dims, equal_variances=True, lambdaa=self.lambdaa,
                      lambdab=self.lambdab, dtype=dtype, device=device)

        data_t = torch.tensor(data, dtype=dtype, device=device)
        pbar = tqdm(total=epoch, leave=False, desc='learning..') if verbose else None
        for i in range(epoch):
            optimizer = torch.optim.Adam(
                model.parameters(),
                **drop_none(lr=lr)
            )
            h_new, score, likelihood = self._update(model, optimizer, data_t)
            if pbar is not None:
                pbar.update(i)
                pbar.desc = f'loss={score}, likelihood={likelihood}, h={h_new}'

        matrix = model.w_est.detach().numpy()

        logger.info(f'trim causal matrix to DAG, threshold={threshold}.')
        if threshold is not None:
            matrix[np.abs(matrix) < threshold] = 0
            matrix[np.abs(matrix) > threshold] = 1
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


class Dagma(nn.Module):
    def __init__(self, dims, mu_init=1.0, lambdaa=0.02, beta_1=0.99, beta_2=0.999,
                 checkpoint=1000, device=None, dtype=None):
        super(Dagma, self).__init__()
        self.d = dims[0]
        self.n = dims[1]
        self.lambda_1 = lambdaa
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.w_est = torch.zeros(self.d, self.d, device=device, dtype=dtype)
        self.mu_init = mu_init
        self.checkpoint = checkpoint
        self.mu_factor = 0.1
        self.lr = 0.0003
        self.warm_iter, self.max_iter = 3e4, 6e4

    def forward(self, X):
        self.X = X
        self.X = -X.mean(axis=0, keepdims=True)
        self.cov = X.t() @ X / float(self.n)
        # dif = torch.eye(self.d) - self.w_est
        # rhs = self.cov @ dif
        # loss = 0.5 * torch.trace(dif.t() @ rhs)
        # G_loss = -rhs
        return self.cov

    def _score(self, cov):
        dif = torch.eye(self.d) - self.w_est
        rhs = cov @ dif
        loss = 0.5 * torch.trace(dif.t() @ rhs)
        G_loss = -rhs
        return loss, G_loss

    def _h(self, s=1.0):
        M = s * torch.eye(self.d) - self.w_est * self.w_est
        h = torch.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * self.w_est * torch.inverse(M).t()
        return h, G_h

    def _func(self, cov, mu, s=1.0):
        score, _ = self._score(cov)
        h, _ = self._h(s)
        obj = mu * (score + self.lambda_1 * torch.abs(self.w_est).sum()) + h
        return obj, score, h

    def _adam_update(self, grad, iter):
        self.opt_m = self.opt_m * self.beta_1 + (1 - self.beta_1) * grad
        self.opt_v = self.opt_v * self.beta_2 + (1 - self.beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - self.beta_1 ** iter)
        v_hat = self.opt_v / (1 - self.beta_2 ** iter)
        grad = m_hat / (torch.sqrt(v_hat) + 1e-8)
        return grad

    def minimize(self, cov, mu, max_iter, lr, s, tol=1e-6, pbar=None):
        W = self.w_est
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        for iter in range(1, max_iter + 1):
            ## Compute the (sub)gradient of the objective
            M = torch.inverse(s * torch.eye(self.d) - W * W) + 1e-16
            while torch.any(M < 0):  # sI - W o W is not an M-matrix
                if iter == 1 or s <= 0.9:
                    return W, False
                else:
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, True
                    W -= lr * grad
                    M = torch.inverse(s * torch.eye(self.d) - W * W) + 1e-16

            G_score = -mu * self.cov @ (torch.eye(self.d) - W)

            Gobj = G_score + mu * self.lambda_1 * torch.sign(W) + 2 * W * M.T

            ## adam step
            grad = self._adam_update(Gobj, iter)
            W -= lr * grad

            ## check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, score, h = self._func(cov, mu, s)
                if torch.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    if pbar is not None:
                        pbar.update(max_iter - iter + 1)
                    break
                obj_prev = obj_new
            if pbar is not None:
                pbar.update(1)
        return W, True


class DagmaDiscovery(BaseDiscovery):
    def __init__(self, lambdaa: float = 0.02, device=None, lr: float = 0.0003,
                 mu_init: float = 1.0, mu_factor: float = 0.1, warm_iter=3e4,
                 max_iter=6e4, T: int = 5, dtype=None, random_state=None):
        super(DagmaDiscovery, self).__init__()
        self.T = T
        self.lr = lr
        self.mu_init = mu_init
        self.dtype = dtype
        self.mu_factor = mu_factor
        self.device = device
        self.lambdaa = lambdaa
        self.random_state = random_state
        self.s = [1.0, .9, .8, .7, .6]
        self.warm_iter, self.max_iter = warm_iter, max_iter

    def train(self, model, X, pbar):
        self.cov = model(X)
        mu = self.mu_init
        if type(self.s) == list:
            if len(self.s) < self.T:
                self.s = self.s + (self.T - len(self.s)) * [self.s[-1]]
        elif type(self.s) in [int, float]:
            self.s = self.T * [self.s]
        else:
            ValueError("s should be a list, int, or float.")
        for i in range(int(self.T)):
            lr_adam, success = self.lr, False
            inner_iters = int(self.max_iter) if i == self.T - 1 else int(self.warm_iter)

            while success is False:
                W_temp, success = model.minimize(self.cov, mu, inner_iters,
                                                 lr=lr_adam, s=self.s[i], pbar=pbar)
                if success is False:
                    lr_adam *= 0.5
                    self.s[i] += 0.1
            self.w_est = W_temp
            mu *= self.mu_factor
        return self.w_est

    def __call__(self, data, *, return_dict=False, threshold=0.01,
                 lr=0.0003, verbose=False, T=5, warm_iter=3e4,
                 s=[1.0, .9, .8, .7, .6], max_iter=6e4, **kwargs):
        assert isinstance(data, (np.ndarray, pd.DataFrame))

        set_random_state(self.random_state)

        if isinstance(data, pd.DataFrame):
            columns = data.columns.tolist()
            data = data.values
        else:
            columns = None

        device = self.device
        if device is None or device == 'auto':
            device = 'cuda' if _is_cuda_available else 'cpu'

        dtype = self.dtype
        if dtype is None:
            dtype = torch.float64

        dims = [data.shape[1]] + [data.shape[0]]

        logger.info(f'learning data{data.shape} with device={device}, dtype={dtype}, dims={dims}')

        model = Dagma(dims, dtype=dtype, device=device)
        data_t = torch.tensor(data, dtype=dtype, device=device)

        epoch = (T - 1) * warm_iter + max_iter
        pbar = tqdm(total=epoch, leave=False, desc='learning..') if verbose else None
        w_est = self.train(model, data_t, pbar)
        matrix = w_est.detach().numpy()

        logger.info(f'trim causal matrix to DAG, threshold={threshold}.')
        if threshold is not None:
            matrix[np.abs(matrix) < threshold] = 0
            matrix[np.abs(matrix) > threshold] = 1
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