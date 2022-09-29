# Some snippets of code are from scikit-learn
import numbers
import numpy as np
import threading

from abc import abstractmethod
from copy import deepcopy
from joblib import Parallel, delayed

from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight


from ..utils import convert2array, inverse_grad, count_leaf_num

from ..base_models import BaseEstModel
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from ylearn.sklearn_ex.cloned.tree import _tree

DOUBLE = _tree.DOUBLE
MAX_INT = np.iinfo(np.int32).max
# we ignore the warm start and inference parts in the current version
class BaseForest:
    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        n_jobs=None,
        random_state=None,
        warm_start=None,
        sub_sample_num=None,
        class_weight=None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.sub_sample_num = sub_sample_num
        self.class_weight = class_weight

    def _validate_estimator(self, default=None):
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError(
                f"n_estimators must be an integer, got {type(self.n_estimators)}."
            )

        if self.n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be greater than zero, got {self.n_estimators}."
            )

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self, append=True, random_state=None):
        estimator = deepcopy(self.base_estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})

        if random_state is not None:
            random_state = check_random_state(random_state)

            to_set = {}
            for key in sorted(estimator.get_params(deep=True)):
                if key == "random_state" or key.endswith("__random_state"):
                    to_set[key] = random_state.randint(np.iinfo(np.int32).max)

            if to_set:
                estimator.set_params(**to_set)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def __len__(self):
        return len(self.estimators_)

    def __getitem__(self, index):
        return self.estimators_[index]

    def __iter__(self):
        return iter(self.estimators_)


def _generate_sub_samples(random_state, all_idx, sub_samples):
    rd = check_random_state(random_state)
    z = rd.choice(all_idx, size=sub_samples, replace=False)
    # print(z)
    return z


def _prediction(e, w, v, v_train, lock, i):
    pred = e._predict_with_array(w, v).reshape(-1, 1)
    y_pred = e.leaf_record.reshape(1, -1)
    with lock:
        temp = y_pred == pred
        # TODO: note that this line actually calculates counts multiple times so it may be improved
        num = np.count_nonzero(temp, axis=1).reshape(-1, 1)
        return temp / num


class BaseCausalForest(BaseEstModel, BaseForest):
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        *,
        sub_sample_num=None,
        estimator_params=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        # bootstrap=True,
        # oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        categories="auto",
        is_discrete_treatment=True,
        is_discrete_outcome=False,
        ccp_alpha=0.0,
    ):
        if estimator_params is None:
            estimator_params = (
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
            )
        # TODO: modify the multiple inheritance
        BaseForest.__init__(
            self,
            base_estimator=base_estimator,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            random_state=random_state,
            warm_start=warm_start,
            sub_sample_num=sub_sample_num,
        )
        BaseEstModel.__init__(
            self,
            is_discrete_outcome=is_discrete_outcome,
            is_discrete_treatment=is_discrete_treatment,
            categories=categories,
        )

        self.verbose = verbose
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.sub_sample_num = sub_sample_num
        self.ccp_alpha = ccp_alpha

    # TODO: the current implementation is a simple version
    # TODO: add shuffle sample
    def fit(
        self,
        data,
        outcome,
        treatment,
        sample_weight=None,
        adjustment=None,
        covariate=None,
    ):
        super().fit(
            data, outcome, treatment, adjustment=adjustment, covariate=covariate
        )

        y, x, w, v = convert2array(data, outcome, treatment, adjustment, covariate)

        # TODO: add check data

        for k, value in {"y": y, "x": x, "v": v}.items():
            setattr(self, "_" + k, value)

        n_train = y.shape[0]
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Determin treatment settings
        if self.categories == "auto" or self.categories is None:
            categories = "auto"
        else:
            categories = list(self.categories)

        if self.is_discrete_treatment:
            # self.transformer = OrdinalEncoder(categories=categories)
            # # self.transformer = OneHotEncoder(categories=categories)
            # self.transformer.fit(x)
            # x = self.transformer.transform(x)
            # # x += np.ones_like(x)
            pass

        self.n_outputs_ = y.shape[1]

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, v)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
        if getattr(x, "dtype", None) != DOUBLE or not x.flags.contiguous:
            x = np.ascontiguousarray(x, dtype=DOUBLE)

        sub_sample_num_ = self._get_sub_samples_num(
            n_samples=x.shape[0], sub_samples=self.sub_sample_num
        )

        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        # y_ = y.squeeze()
        if n_more_estimators < 0:
            raise ValueError(
                f"n_estimators={self.n_estimators} must be larger or equal to "
                f"len(estimators_)={len(self.estimators_)} when warm_start==True"
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # get sub samples idices
            all_idx = np.arange(start=0, stop=self._y.shape[0])
            self.sub_sample_idx = [
                _generate_sub_samples(t.random_state, all_idx, sub_sample_num_)
                for t in trees
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(t._fit_with_array)(x[s], y[s], w[s], v[s], i)
                for i, (t, s) in enumerate(zip(trees, self.sub_sample_idx))
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        self._is_fitted = True

        return self

    def estimate(self, data=None, **kwargs):
        effect_ = self._prepare4est(data=data)
        return effect_

    def effect_nji(self, *args, **kwargs):
        return super().effect_nji(*args, **kwargs)

    def apply(self):
        pass

    def decision_path(
        self,
    ):
        pass

    def feature_importances_(self):
        pass

    @property
    def n_features_(self):
        pass

    # TODO: support oob related methods

    def _get_sub_samples_num(self, n_samples, sub_samples):
        if sub_samples is None:
            return round(n_samples * 0.85)

        if isinstance(sub_samples, numbers.Integral):
            if sub_samples > n_samples:
                raise ValueError(
                    f"sub_samples_num must be <= n_samples={n_samples} but was given {sub_samples}"
                )
            return sub_samples

        if isinstance(sub_samples, numbers.Real):
            return round(n_samples * sub_samples)

    def _check_features(self, data):
        v = self._v if data is None else convert2array(data, self.covariate)[0]
        return v

    def _prepare4est(self, data=None):
        assert self._is_fitted, "The model is not fitted yet."
        v = self._check_features(data=data)
        alpha = self._compute_alpha(v)
        inv_grad_, theta_ = self._compute_aug(self._y, self._x, alpha)
        theta = np.einsum("njk,nk->nj", inv_grad_, theta_)
        return theta

    def _compute_alpha(self, v):
        # first implement a version which only take one example as its input
        lock = threading.Lock()
        w = v.copy()
        if self.n_outputs_ > 1:
            raise ValueError(
                "Currently do not support the number of output which is larger than 1"
            )
        else:
            alpha = np.zeros((v.shape[0], self._v.shape[0]))

        alpha_collection = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,)(
            delayed(_prediction)(e, w, v, self._v, lock, i)
            for i, (e, s) in enumerate(zip(self.estimators_, self.sub_sample_idx))
        )
        for alpha_, s in zip(alpha_collection, self.sub_sample_idx):
            alpha[:, s] += alpha_
        return alpha / self.n_estimators

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
