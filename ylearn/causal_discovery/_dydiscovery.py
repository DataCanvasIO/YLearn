import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ylearn.utils import drop_none, set_random_state, logging
from ._base import BaseDiscovery

_is_cuda_available = torch.cuda.is_available()

logger = logging.get_logger(__name__)


class Dynotears(nn.Module):
    def __init__(self, dims, order=1, dtype=None, device=None):
        super(Dynotears, self).__init__()
        self.d = dims[0]
        self.n = dims[1]
        self.k = order
        self.w_est = torch.ones(self.d, self.d, dtype=dtype, device=device)
        self.w_est = nn.Parameter(self.w_est, requires_grad=True)
        self.p_est = torch.ones(self.d, self.d, self.k, dtype=dtype, device=device)
        self.p_est = nn.Parameter(self.p_est, requires_grad=True)

    def forward(self, Xlags):
        M = torch.matmul(Xlags[self.k:], self.w_est)
        for k in range(self.k):
            M += torch.matmul(Xlags[:-self.k], self.p_est[:, :, k])
        return M

    def h_func(self):
        h = torch.trace(torch.matrix_exp(self.w_est * self.w_est)) - self.d
        return h

    def diag_zero(self):
        diag_loss = torch.trace(self.w_est * self.w_est)
        return diag_loss

    def l1_reg(self):
        return torch.abs(self.w_est).sum()

    def L1norm(self):
        loss = 0.0
        for order in range(self.k):
            loss += torch.abs(self.p_est[:, :, order]).sum()
        return loss

    @torch.no_grad()
    def is_weight_nan(self):
        return torch.isnan(self.w_est).any().cpu().detach().numpy()


class DyCausalDiscovery(BaseDiscovery):
    def __init__(self, lambdaa: float = 0.01, lambdab: float = 0.01, h_tol: float = 1e-6,
                 rho_max: float = 1e6, scale='auto', device=None, dtype=None,
                 random_state=None):
        self.lambdaa = lambdaa
        self.lambdab = lambdab
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.device = device
        self.dtype = dtype
        self.random_state = random_state

    @staticmethod
    def _squared_loss(output, target):
        n = target.shape[0]
        loss = 0.5 / n * torch.sum((output - target) ** 2)
        return loss

    def _rho_h_update(self, model, optimizer, X, order, rho, alpha, h):
        h_new = None
        while rho < self.rho_max:
            def closure():
                optimizer.zero_grad()
                X_hat = model(X)
                loss = self._squared_loss(X_hat, X[order:])
                h_val = model.h_func()
                diag_loss = model.diag_zero()
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                l1_reg_intra = 0.5 * self.lambdaa * model.l1_reg()
                l1_reg_inter = 0.5 * self.lambdab * model.L1norm()
                primal_obj = loss + 100 * penalty + 1000 * diag_loss + l1_reg_inter + l1_reg_intra
                primal_obj.backward()
                return primal_obj

            optimizer.step(closure)
            with torch.no_grad():
                h_new = model.h_func()
            if h_new.item() > 0.25 * h:
                rho *= 10
            else:
                break
        alpha += rho * h_new
        return rho, alpha, h_new

    def __call__(self, data, *, return_dict=False, threshold=0.3,
                 epoch=100, lr=0.01, max_iter=1500, order=1, step=5,
                 verbose=False, **kwargs):
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
            # dtype = torch.float32 if str(device) == 'cuda' else torch.float64
            dtype = torch.float64
        data = data.reshape(int(step + order), int(data.shape[0] / (step + order)), data.shape[1])

        dims = [data.shape[2]] + [data.shape[1]] + [data.shape[0]]

        logger.info(f'learning data{data.shape} with device={device}, dtype={dtype}, dims={dims}')

        model = Dynotears(dims=dims, order=order, dtype=dtype, device=device)
        data_t = torch.tensor(data, dtype=dtype, device=device)

        pbar = tqdm(total=epoch, leave=False, desc='learning..') if verbose else None
        rho, alpha, h = 1.0, 0.0, np.inf
        for i in range(epoch):
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                **drop_none(lr=lr, max_iter=max_iter)
            )
            rho, alpha, h = self._rho_h_update(model, optimizer, data_t, order, rho, alpha, h)
            if pbar is not None:
                pbar.update(i)
                pbar.desc = f'rho={rho}, alpha={alpha}, h={h}'
            if h <= self.h_tol or rho >= self.rho_max or model.is_weight_nan():
                break
        w_matrix = model.w_est.detach().numpy()
        p_matrix = model.p_est.detach().numpy()

        logger.info(f'trim causal matrix to DAG, threshold={threshold}.')
        if threshold is not None:
            w_matrix[np.abs(w_matrix) < threshold] = 0
            p_matrix[np.abs(p_matrix) < 0.1] = 0

        w_matrix = self._to_dag(w_matrix)

        if columns is not None:
            matrix_ = pd.DataFrame(w_matrix, columns=columns, index=columns)

        if return_dict:
            matrix_ = self.matrix2dict(w_matrix)

        return w_matrix, p_matrix

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


class DyGolem(nn.Module):
    def __init__(self, dims, order, dtype=None, device=None):
        super(DyGolem, self).__init__()
        self.d = dims[0]
        self.n = dims[1]
        self.k = order
        self.w_est = torch.ones(self.d, self.d, dtype=dtype, device=device)
        self.w_est = nn.Parameter(self.w_est, requires_grad=True)
        self.p_est = torch.ones(self.d, self.d, self.k, dtype=dtype, device=device)
        self.p_est = nn.Parameter(self.p_est, requires_grad=True)

    def h_func(self):
        h = torch.trace(torch.matrix_exp(self.w_est * self.w_est)) - self.d
        return h

    def l1_reg(self):
        return torch.abs(self.w_est).sum()

    def L1norm(self):
        loss = 0.0
        for order in range(self.k):
            loss += torch.abs(self.p_est[:, :, order]).sum()
        return loss

    def forward(self, Xlags):
        M = torch.matmul(Xlags[self.k:], self.w_est)
        for k in range(self.k):
            M += torch.matmul(Xlags[:-self.k], self.p_est[:, :, k])
        return M

class DygolemCausalDiscovery(BaseDiscovery):
    def __init__(self, equal_variances=True, lambdaa=0.001, lambdab=0.001,
                 lambdac=0.5, scale='auto', device=None, dtype=None,
                 random_state=None):
        self.equal_variances = equal_variances
        self.lambdaa = lambdaa
        self.lambdab = lambdab
        self.lambdac = lambdac
        self.scale = scale
        self.device = device
        self.dtype = dtype
        self.random_state = random_state

    def _compute_likelihood(self, output, target):
        d = target.shape[2]
        if self.equal_variances:
            return 0.5 * d * torch.log(
                torch.sum(torch.square(output - target))
            )
        else:
            return 0.5 * torch.sum(torch.log(
                torch.sum((output - target) ** 2)
            ))

    def train(self, model, optimizer, X, order):
        d = X.shape[2]
        def closure():
            optimizer.zero_grad()
            X_hat = model(X)
            loss = self._compute_likelihood(X_hat, X[order:])
            logdet = torch.slogdet(torch.eye(d) - model.w_est)[1]
            likelihood = loss - logdet
            h_val = model.h_func()
            l1_reg_intra = self.lambdaa * model.l1_reg()
            l1_reg_inter = self.lambdab * model.L1norm()
            primal_obj = likelihood + l1_reg_intra + l1_reg_inter + self.lambdac * h_val
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)
        with torch.no_grad():
            h_new = model.h_func()

        return h_new

    def __call__(self, data, *, return_dict=False, threshold=0.3,
                 epoch=6000, lr=0.001, order=1, step=5, verbose=True,
                 **kwargs):
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
            # dtype = torch.float32 if str(device) == 'cuda' else torch.float64
            dtype = torch.float64
        data = data.reshape(int(step + order), int(data.shape[0] / (step + order)), data.shape[1])

        dims = [data.shape[2]] + [data.shape[1]] + [data.shape[0]]

        logger.info(f'learning data{data.shape} with device={device}, dtype={dtype}, dims={dims}')

        model = Dynotears(dims=dims, order=order, dtype=dtype, device=device)
        data_t = torch.tensor(data, dtype=dtype, device=device)

        pbar = tqdm(total=epoch, leave=False, desc='learning..') if verbose else None
        for i in range(epoch):
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr)
            h = self.train(model, optimizer, data_t, order)
            if pbar is not None:
                pbar.update(i)
                pbar.desc = f'h={h}'

        w_matrix = model.w_est.detach().numpy()
        p_matrix = model.p_est.detach().numpy()

        logger.info(f'trim causal matrix to DAG, threshold={threshold}.')
        if threshold is not None:
            w_matrix[np.abs(w_matrix) < threshold] = 0
            p_matrix[np.abs(p_matrix) < 0.3] = 0

        w_matrix = self._to_dag(w_matrix)

        if columns is not None:
            matrix_ = pd.DataFrame(w_matrix, columns=columns, index=columns)

        if return_dict:
            matrix_ = self.matrix2dict(w_matrix)

        return w_matrix, p_matrix

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