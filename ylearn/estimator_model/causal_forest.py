import numbers
from typing_extensions import Self
import numpy as np

from abc import abstractmethod
from copy import deepcopy

from sklearn.utils import check_random_state

from . import CausalTree, DoubleML, BaseEstModel
from .utils import get_tr_ctrl
from ._generalized_forest._grf import GRForest


"""
Two different kinds of causal forest will be implemented, including
 1. A causal forest directly serving as an average of a bunch of causal trees (honest or not)
 3. A causal forest by growing generalized random forest tree (these trees may grow in a dfferent
    way when compared to the causal tree) combined with the local centering technique.
"""


class CausalForest(DoubleML):
    def __init__(
        self,
        x_model,
        y_model,
        n_estimators=100,
        *,
        cf_fold=1,
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
        adjustment_transformer=None,
        covariate_transformer=None,
        categories="auto",
        proba_output=False,
        honest_subsample_num=None,
    ):
        yx_model = GRForest(
            n_estimators=n_estimators,
            sub_sample_num=sub_sample_num,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            n_jobs=n_jobs,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
            is_discrete_treatment=is_discrete_treatment,
            is_discrete_outcome=is_discrete_outcome,
            verbose=verbose,
            warm_start=warm_start,
            honest_subsample_num=honest_subsample_num,
        )

        super().__init__(
            x_model=x_model,
            y_model=y_model,
            yx_model=yx_model,
            cf_fold=cf_fold,
            adjustment_transformer=adjustment_transformer,
            covariate_transformer=covariate_transformer,
            random_state=random_state,
            is_discrete_outcome=is_discrete_outcome,
            is_discrete_treatment=is_discrete_treatment,
            categories=categories,
            proba_output=proba_output,
        )

    def _fit_2nd_stage(self, yx_model, x_prime, y_prime, v, **kwargs):
        yx_model.covariate = self.covariate
        sample_weight = kwargs.get("sample_weight", None)
        yx_model._fit_with_array(y_prime, x_prime, v, v, sample_weight=sample_weight)

    def _cal_x_prime(self, x, x_hat, v):
        x_prime = x - x_hat
        if self.is_discrete_treatment:
            if hasattr(self, "control") and self.control is not None:
                ctrl = self.control
                if not isinstance(ctrl, np.ndarray):
                    self._is_fitted = True  # TODO: this line is for temp useage
                    ctrl = get_tr_ctrl(
                        ctrl,
                        self.comp_transormer,
                        treat=False,
                        one_hot=False,
                        discrete_treat=True,
                    )
                    self._is_fitted = False
            else:
                ctrl = 0

            x_prime = np.delete(x_prime, ctrl, axis=1)

        return x_prime

    def estimate(
        self, data=None, treat=None, control=None, quantity=None, target_outcome=None
    ):
        return self.yx_model.estimate(data=data)


# for iv
class IVCausalForest:
    def __init__(self) -> None:
        pass
