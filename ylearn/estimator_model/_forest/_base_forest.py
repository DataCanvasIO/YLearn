# Some snippets of code are from scikit-learn

import numbers
import numpy as np

from abc import abstractmethod
from copy import deepcopy
from joblib import Parallel, delayed

from sklearn.utils import check_random_state

from ..utils import convert2array

from ..base_models import BaseEstModel
from ..causal_tree import CausalTree


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
        max_samples=None,
        class_weight=None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.max_samples = max_samples
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

    def __len__(self):
        return len(self.estimators_)

    def __getitem__(self, index):
        return self.estimators_[index]

    def __iter__(self):
        return iter(self.estimators_)


class BaseCausalForest(BaseEstModel, BaseForest):
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        *,
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
        # verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        categories="auto",
    ):
        estimator_params = (
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "random_state",
            "ccp_alpha",
        )
        super().__init__(
            # random_state,
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            is_discrete_treatment=True,
            is_discrete_outcome=False,
            _is_fitted=False,
            categories=categories,
            random_state=random_state,
            warm_start=warm_start,  # currently we do not implement this one
            max_samples=max_samples,
            n_jobs=n_jobs,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

    def fit(
        self,
        data,
        outcome,
        treatment,
        adjustment=None,
        covariate=None,
        treat=None,
        control=None,
    ):
        y = convert2array(data, outcome)[0]
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_outputs_ = y.shape[1]

        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                f"n_estimators={self.n_estimators} must be larger or equal to "
                f"len(estimators_)={len(self.estimators_)} when warm_start==True"
            )
        else:
            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
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
                delayed(t.fit)(
                    data,
                    outcome,
                    treatment,
                    adjustment=None,
                    covariate=None,
                    treat=None,
                    control=None,
                )
                for t in trees
            )  # need to be imporved (the current implementation will call the same transformers many times)

            # Collect newly grown trees
            self.estimators_.extend(trees)

        return self

    def estimate(self, data=None, **kwargs):
        return super().estimate(data, **kwargs)

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

    def _check_features(
        self,
    ):
        pass

    def _prepare4est(self, data=None):
        pass
