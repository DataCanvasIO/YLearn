from sklearn.utils import check_random_state

from .tree import GrfTree
from ._base_forest import BaseCausalForest

# causal forest without local centering technique
class GRForest(BaseCausalForest):
    def __init__(
        self,
        n_estimators=100,
        *,
        sub_sample_num=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        n_jobs=None,
        random_state=None,
        # categories="auto",
        ccp_alpha=0.0,
        is_discrete_treatment=True,
        is_discrete_outcome=False,
        verbose=0,
        warm_start=False,
    ):
        base_estimator = GrfTree()

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
            ),
            n_jobs=n_jobs,
            # categories=categories,
            random_state=random_state,
            sub_sample_num=sub_sample_num,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            is_discrete_treatment=is_discrete_treatment,
            is_discrete_outcome=is_discrete_outcome,
            verbose=verbose,
            warm_start=warm_start,
        )

    def fit(
        self,
        data,
        outcome,
        treatment,
        adjustment=None,
        covariate=None,
        sample_weight=None,
    ):
        super().fit(
            data,
            outcome,
            treatment,
            adjustment=adjustment,
            covariate=covariate,
            sample_weight=sample_weight,
        )
