import threading
import numpy as np

from ._grf_tree import _GrfTree
from .._generalized_forest import BaseCausalForest
from .utils import inverse_grad


class NaiveGrf(BaseCausalForest):
    """Avoid using this class to estimate causal effect when assuming discrete treatment."""

    def __init__(
        self,
        n_estimators=100,
        *,
        sub_sample_num=None,
        max_depth=None,
        min_split_tolerance=1e-5,
        n_jobs=None,
        random_state=None,
        max_samples=None,
        categories="auto",
    ):
        base_estimator = _GrfTree()
        estimator_params = ("max_depth", "min_split_tolerance")
        self.min_split_tolerance = min_split_tolerance

        super().__init__(
            base_estimator=base_estimator,
            estimator_params=estimator_params,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            max_samples=max_samples,
            categories=categories,
            random_state=random_state,
            sub_sample_num=sub_sample_num,
        )

    def fit(
        self,
        data,
        outcome,
        treatment,
        adjustment=None,
        covariate=None,
    ):
        super().fit(
            data, outcome, treatment, adjustment=adjustment, covariate=covariate
        )

    def _compute_aug(self, y, x, alpha):
        r"""Formula:
        We first need to repeat vectors in training set the number of the #test set times to enable
        tensor calculation.

        The first half is
            g_n^j_k = \sum_i \alpha_n^i (x_n^i_j - \sum_i \alpha_n^i x_n^i_j)(x_n^i_j - \sum_i \alpha_n^i x_n^i_j)^T
        while the second half is
            \theta_n^j = \sum_i \alpha_n^i (x_n^i_j - \sum_i \alpha_n^i x_n^i_j)(y_n^i - \sum_i \alpha_n^i y_n^i)
        which are then combined to give
            g_n^j_k \theta_{nj}

        Parameters
        ----------
        y : ndarray
            outcome vector of the training set, shape (i,)
        x : ndarray
            treatment vector of the training set, shape(i, j)
        alpha : ndarray
            The computed alpha, shape (n, i) where i is the number of training set

        Returns
        -------
        ndarray, ndarray
            the first is inv_grad while the second is theta_
        """
        n_test, n_train = alpha.shape
        x = np.tile(x.reshape((1,) + x.shape), (n_test, 1, 1))
        x_dif = x - (alpha.reshape(n_test, -1, 1) * x).sum(axis=1).reshape(
            n_test, 1, -1
        )
        grad_ = alpha.reshape(n_test, n_train, 1, 1) * np.einsum(
            "nij,nik->nijk", x_dif, x_dif
        )
        grad_ = grad_.sum(1)
        inv_grad = inverse_grad(grad_)

        y = np.tile(y.reshape(1, -1), (n_test, 1))
        y_dif = y - alpha * y
        theta_ = ((alpha * y_dif).reshape(n_test, n_train, 1) * x_dif).sum(1)

        return inv_grad, theta_
