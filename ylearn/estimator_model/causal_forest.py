import numpy as np

from . import DoubleML
from .utils import get_tr_ctrl
from ._generalized_forest._grf import GRForest


"""
Two different kinds of causal forest will be implemented, including
 1. A causal forest directly serving as an average of a bunch of causal trees (honest or not, not implemented yet)
 2. A causal forest by growing generalized random forest tree (these trees may grow in a dfferent
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
        """
        A GRForest utilizing the local centering technique aka double machine learning framework.

        Parameters
        ----------
        x_model : estimator
            Machine learning models for fitting x. Any such models should implement
            the :py:func:`fit` and :py:func:`predict`` (also :py:func:`predict_proba` if x is discrete) methods.

        y_model : estimator
            The machine learning model which is trained to modeling the outcome. Any valid y_model should implement
            the :py:func:`fit()` and :py:func:`predict()` methods.

        cf_fold : int, default=1
            The number of folds for performing cross fit in the first stage.

        n_estimators : int, default=100
            The number of trees for growing the GRF.

        sub_sample_num: int or float, default=None
            The number of samples to train each individual tree.
            - If a float is given, then the number of ``sub_sample_num*n_samples`` samples will be
                sampled to train a single tree
            - If an int is given, then the number of ``sub_sample_num`` samples will be sampled to
                train a single tree

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
                - If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)`
                    are the minimum number of samples for each node.

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
            The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
            :meth:`decision_path` and :meth:`apply` are all parallelized over the
            trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors. See :term:`Glossary
            <n_jobs>` for more details.

        verbose: int, default=0
            Controls the verbosity when fitting and predicting

        honest_subsample_num: int or float, default=None
            The number of samples to train each individual tree in an honest manner. Typically
            setting this value will have better performance. Use all ``sub_sample_num`` if ``None``
            is given.

            - If a float is given, then the number of ``honest_subsample_num*sub_sample_num`` samples
                will be used to train a single tree while the rest ``(1 - honest_subsample_num)*sub_sample_num``
                samples will be used to label the trained tree.
            - If an int is given, then the number of ``honest_subsample_num`` samples will be sampled
                to train a single tree while the rest ``sub_sample_num - honest_subsample_num`` samples will
                be used to label the trained tree.

        proba_output : bool, default=False
            Whether to estimate probability of the outcome if it is a discrete one. If True, then the given
            y_model must have the method ``predict_proba``.
        """
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

    def fit(
        self, data, outcome, treatment, adjustment=None, covariate=None, control=None
    ):
        return super().fit(
            data=data,
            outcome=outcome,
            treatment=treatment,
            adjustment=adjustment,
            covariate=covariate,
            control=control,
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

    def estimate(self, data=None, **kwargs):
        return self.yx_model.estimate(data=data)

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
        return self.yx_model.apply(v)

    @property
    def feature_importance(self):
        return self.yx_model.feature_importances_()


# for iv
class IVCausalForest:
    def __init__(self) -> None:
        pass
