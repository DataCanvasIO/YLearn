from ._grf import NaiveGrf
from ..double_ml import DoubleML


class NaiveCausalForest(DoubleML):
    """A version of causal forest with pure python implementation, i.e., it is super slow. Thus
    this one is only for experimental use.
    """

    print("Do not use currently.")

    def __init__(
        self,
        x_model,
        y_model,
        cf_fold=1,
        random_state=2022,
        is_discrete_treatment=False,
        categories="auto",
        *,
        n_estimators=100,
        sub_sample_num=None,
        max_depth=None,
        min_split_tolerance=1e-5,
        n_jobs=None,
    ):
        yx_model = NaiveGrf(
            n_estimators=n_estimators,
            sub_sample_num=sub_sample_num,
            max_depth=max_depth,
            min_split_tolerance=min_split_tolerance,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        super().__init__(
            x_model,
            y_model,
            yx_model,
            cf_fold,
            random_state,
            is_discrete_treatment,
            categories,
        )
