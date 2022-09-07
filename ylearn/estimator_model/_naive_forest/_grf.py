import numpy as np

from ._grf_tree import _GrfTree
from .._forest import BaseCausalForest


class NaiveGrf(BaseCausalForest):
    def __init__(
        self,
        n_estimators=100,
        *,
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
