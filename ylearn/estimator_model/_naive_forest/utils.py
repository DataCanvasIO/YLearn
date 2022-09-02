import numpy as np

from numpy.linalg import inv


def grad(x_dif, n):
    return np.einsum("ni,nj->nij", x_dif, x_dif).sum(axis=0) / n


def grad_coef(x_dif, y_dif, ls_coef):
    return x_dif * (y_dif - np.einsum("nj,j->n", x_dif, ls_coef)).reshape(-1, 1)


def inverse_grad(grad, eps=1e-5):
    grad += np.eye(grad.shape[0]) * eps
    return inv(grad)
