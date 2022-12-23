# Some snippets of theses codes are from scikit-learn
import numbers
import numpy as np
import threading

from abc import abstractmethod
from copy import deepcopy
from joblib import Parallel, delayed
from collections import defaultdict

from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight


from ..utils import convert2array, inverse_grad

from ..base_models import BaseEstModel
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
        honest=None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        self.honest = honest
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
    return z


# #####: I modify this line

# def _prediction(e, v):
#     pred = e._predict_with_array(None, v).reshape(-1, 1)
#     y_pred = e.leaf_record.reshape(1, -1)

#     # compute the leaf value to which every test sample belongs
#     temp = y_pred == pred

#     # compute the number of samples in that leaf
#     num = np.count_nonzero(temp, axis=1).reshape(-1, 1)

#     with np.errstate(divide="ignore", invalid="ignore"):
#         alpha = np.where(num > 0, temp / num, 0)
#         return alpha


def _prediction(e, v, n_training_samples):
    pred = e.apply(None, v)
    # leaf_record is a dict where its key is the leaf id while its value include the indices of training samples falling into that leaf
    leaf_record = e.leaf_record
    n, i = v.shape[0], n_training_samples

    # create alpha
    alpha = np.zeros(shape=(n, i))
    for leaf, row in zip(pred, alpha):
        index = leaf_record[leaf]
        if len(index) != 0:
            row[np.array(index)] = 1 / len(index)

    return alpha


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
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        categories="auto",
        is_discrete_treatment=True,
        is_discrete_outcome=False,
        ccp_alpha=0.0,
        honest_subsample_num=None,
    ):
        """
        A base class of forest based estimator models for estimating the causal effect.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees for growing the GRF.

        sub_sample_num: int or float, default=None
            The number of samples to train each individual tree.
            - If a float is given, then the number of ``sub_sample_num*n_samples`` samples will be sampled to train a single tree
            - If an int is given, then the number of ``sub_sample_num`` samples will be sampled to train a single tree

        max_depth: int, default=None
            The max depth that a single tree can reach. If ``None`` is given, then there is no limit of
            the depth of a single tree.

        min_samples_split: int, default=2
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
            `ceil(min_samples_split * n_samples)` are the minimum
            number of samples for each split.

        min_samples_leaf: int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.

                - If int, then consider `min_samples_leaf` as the minimum number.
                - If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.

        min_weight_fraction_leaf: float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.

        max_features: int, float or {"sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:

                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and `int(max_features * n_features)` features are considered at each split.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.

        random_state: int
            Controls the randomness of the estimator.

        max_leaf_nodes: int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease: float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.

        n_jobs: int, default=None
            The number of jobs to run in parallel. :meth:`fit`, :meth:`estimate`, and :meth:`apply` are all parallelized over the
            trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors. See :term:`Glossary
            <n_jobs>` for more details.

        verbose: int, default=0
            Controls the verbosity when fitting and predicting

        honest_subsample_num: int or float, default=None
            The number of samples to train each individual tree in an honest manner. Typically set this value will have better performance. Use all ``sub_sample_num`` if ``None`` is given.
            - If a float is given, then the number of ``honest_subsample_num*sub_sample_num`` samples will be used to train a single tree while the rest ``(1 - honest_subsample_num)*sub_sample_num`` samples will be used to label the trained tree.
            - If an int is given, then the number of ``honest_subsample_num`` samples will be sampled to train a single tree while the rest ``sub_sample_num - honest_subsample_num`` samples will be used to label the trained tree.
        """
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
                "honest",
            )

        BaseForest.__init__(
            self,
            base_estimator=base_estimator,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            random_state=random_state,
            warm_start=warm_start,
            sub_sample_num=sub_sample_num,
            honest=honest_subsample_num,
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
        self.honest_subsample_num = honest_subsample_num

        self.honest_sample = None

    def fit(
        self,
        data,
        outcome,
        treatment,
        sample_weight=None,
        adjustment=None,
        covariate=None,
        **kwargs,
    ):
        """Fit the model.

        Parameters
        ----------
        data: pandas.DataFrame
            The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.
        outcome: list of str, optional
            Names of the outcomes.
        treatment: list of str, optional
            Names of the treatments.
        covariate: list of str, optional, default=None
            Names of the covariate vectors.
        adjustment: list of str, optional, default=None
            This will be the same as the covariate.
        sample_weight:  ndarray, optional, default=None
            Weight of each sample of the training set.

        Returns
        ----------
            An instance of GRForest.
        """
        super().fit(
            data, outcome, treatment, adjustment=adjustment, covariate=covariate
        )

        y, x, w, v = convert2array(data, outcome, treatment, adjustment, covariate)

        self._fit_with_array(y, x, w, v, sample_weight=sample_weight, **kwargs)

    def estimate(self, data=None, **kwargs):
        """Estimate the causal effect of the treatment on the outcome in data.


        Parameters
        ----------
        data : pandas.DataFrame, optional, default=None
            If None, data will be set as the training data.

        Returns
        -------
        ndarray or float, optional
            The estimated causal effect.
        """
        effect_ = self._prepare4est(data=data)
        return effect_

    def effect_nji(self, *args, **kwargs):
        return super().effect_nji(*args, **kwargs)

    def apply(self, v):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        v : array-like of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        v_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint v_i in v and for each tree in the forest,
            return the index of the leaf v ends up in.
        """
        results = Parallel(**self._job_options())(
            delayed(tree.apply)(v, v, check_input=False) for tree in self.estimators_
        )

        return np.array(results).T

    @property
    def feature_importance(self):
        return self.feature_importances_()

    def feature_importances_(self):
        all_importances = Parallel(**self._job_options())(
            delayed(getattr)(tree, "feature_importances_")
            for tree in self.estimators_
            if tree.tree_.node_count > 1
        )

        if not all_importances:
            return np.zeros(self.n_features_in_, dtype=np.float64)

        all_importances = np.mean(all_importances, axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    # TODO: add check data
    def _fit_with_array(self, y, x, w, v, sample_weight=None, **kwargs):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        for k, value in {"y": y, "x": x, "v": v}.items():
            setattr(self, "_" + k, value)

        # Determin treatment settings
        # if self.categories == "auto" or self.categories is None:
        #     categories = "auto"
        # else:
        #     categories = list(self.categories)

        # if self.is_discrete_treatment:
        #     # self.transformer = OrdinalEncoder(categories=categories)
        #     # # self.transformer = OneHotEncoder(categories=categories)
        #     # self.transformer.fit(x)
        #     # x = self.transformer.transform(x)
        #     # # x += np.ones_like(x)
        #     pass

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

        self.honest_sample = self._get_honest_samples_num(sub_samples=sub_sample_num_)

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
            trees = Parallel(**self._job_options())(
                # delayed(t._fit_with_array)(x[s], y[s], w[s], v[s], i)
                # for i, (t, s) in enumerate(zip(trees, self.sub_sample_idx))
                delayed(self._fit)(
                    t,
                    x,
                    y,
                    w,
                    v,
                    s,
                    i,
                    self.honest_sample,
                    sample_weight,
                    self.verbose,
                    **kwargs,
                )
                for i, (t, s) in enumerate(zip(trees, self.sub_sample_idx))
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        self._is_fitted = True

        return self

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

    def _get_honest_samples_num(self, sub_samples):
        if self.honest_subsample_num is None:
            return None

        if isinstance(self.honest_subsample_num, numbers.Integral):
            if self.honest_subsample > sub_samples:
                raise ValueError(
                    f"honest_subsample must be <= sub_samples={sub_samples} but was given {self.honest_subsample}"
                )
            return self.honest_subsample_num

        if isinstance(self.honest_subsample_num, numbers.Real):
            return round(self.honest_subsample_num * sub_samples)

    def _check_features(self, data):
        v = self._v if data is None else convert2array(data, self.covariate)[0]
        return v

    def _prepare4est(self, data=None, batch_size=100):
        assert self._is_fitted, "The model is not fitted yet."
        v = self._check_features(data=data)
        a = [
            self.__do_prepare4est(v[i : i + batch_size])
            for i in range(0, len(v), batch_size)
        ]
        return np.concatenate(a, axis=0)

    def __do_prepare4est(self, data):
        alpha = self._compute_alpha(data)
        inv_grad_, theta_ = self._compute_aug(self._y, self._x, alpha)
        theta = np.einsum("njk,nk->nj", inv_grad_, theta_)
        return theta

    def _fit(
        self,
        t,
        x,
        y,
        w,
        v,
        s,
        i,
        honest_sample_num,
        sample_weight=None,
        verbose=0,
        **kwargs,
    ):
        if verbose != 0:
            print(f"Fit the {i + 1} tree")

        sample_weight_honest = None

        if sample_weight is not None:
            sample_weight = sample_weight[s]
            sample_weight_honest = sample_weight[honest_sample_num:]

        if honest_sample_num is None:
            t._fit_with_array(
                x[s], y[s], w[s] if w is not None else None, v[s], sample_weight
            )
            # v_pred = t._predict_with_array(None, v[s])
            v_pred = t.apply(None, v[s])
            # TODO: idx may not be necessary
            idx = s
        else:
            t._fit_with_array(
                x[s][honest_sample_num:],
                y[s][honest_sample_num:],
                w[s][honest_sample_num:] if w is not None else None,
                v[s][honest_sample_num:],
                sample_weight_honest,
            )
            # v_pred = t._predict_with_array(None, v[s][:honest_sample_num])
            v_pred = t.apply(None, v[s][:honest_sample_num])
            idx = s[:honest_sample_num]

        # TODO: note that if a predict value is 0 this line may cause some unexpected result, so it may be more approriate to create an nan array instead
        leaf_record = np.zeros(v.shape[0])
        leaf_record[idx] = v_pred

        # #####: I modify this line
        collect_all = defaultdict(list)
        for i, node_id in enumerate(leaf_record):
            collect_all[node_id].append(i)

        # t.leaf_record = leaf_record.reshape(1, -1)
        t.leaf_record = collect_all

        return t

    def _compute_alpha(self, v):
        if self.n_outputs_ > 1:
            raise ValueError(
                "Currently do not support the number of output which is larger than 1"
            )
        n_training_samples = self._y.shape[0]

        alpha_collection = Parallel(**self._job_options())(
            # delayed(_prediction)(e, w, v, self._v, lock, s)
            # for i, (e, s) in enumerate(zip(self.estimators_, sub_sample_idx))
            delayed(_prediction)(
                e,
                v,
                n_training_samples,
            )
            for i, e in enumerate(self.estimators_)
        )
        alpha = np.mean(alpha_collection, axis=0)
        return alpha

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

    def _job_options(self):
        return dict(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads")
