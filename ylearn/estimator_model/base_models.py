import numpy as np
import pandas as pd

from ylearn.utils import to_repr

# from ..utils._common import check_cols

#TODO: add support for assigning different treatment values for different examples for all models.


class BaseEstModel:
    """
    Base class for various estimation learner.

    Attributes
    ----------
    ml_model_dic : dict
        A dictionary of default machine learning sklearn models currently
        including
            'LR': LinearRegression
            'LogistR': LogisticRegression.

    Methods
    ----------
    _prepare(data, outcome, treatment, adjustment, individual=None)
        Prepare (fit the model) for estimating various quantities including
        ATE, CATE, ITE, and CITE.
    estimate(data, outcome, treatment, adjustment, quantity='ATE',
                 condition_set=None, condition=None, individual=None)
        Integrate estimations for various quantities into a single method.
    estimate_ate(self, data, outcome, treatment, adjustment)
    estimate_cate(self, data, outcome, treatment, adjustment,
                      condition_set, condition)
    estimate_ite(self, data, outcome, treatment, adjustment, individual)
    estimate_cite(self, data, outcome, treatment, adjustment,
                      condition_set, condition, individual)
    """

    def __init__(
        self,
        random_state=2022,
        is_discrete_treatment=False,
        is_discrete_outcome=False,
        _is_fitted=False,
        # is_discrete_instrument=False,
        categories='auto'
    ):
        self.random_state = random_state if random_state is not None else 2022
        self.is_discrete_treatment = is_discrete_treatment
        self.is_discrete_outcome = is_discrete_outcome
        self.categories = categories

        self._is_fitted = _is_fitted

        # fitted
        self.treatment = None
        self.outcome = None
        self.treats_ = None

    def fit(
        self,
        data,
        outcome,
        treatment,
        # adjustment=None,
        # covariate=None,
        **kwargs,
    ):
        assert data is not None and isinstance(data, pd.DataFrame)

        # check_cols(data, treatment, outcome)
        if isinstance(treatment, str):
            treatment = [treatment]

        self.treatment = treatment
        self.outcome = outcome

        for k, v in kwargs.items():
            setattr(self, k, v)
            # check_cols(data, v)

        if data is not None and self.is_discrete_treatment:
            treats = {}
            for t in treatment:
                treats[t] = tuple(np.sort(data[t].unique()).tolist())
        else:
            treats = None
        self.treats_ = treats

        return self

    def effect_nji(self, *args, **kwargs):
        raise NotImplementedError()

    #
    # def _prepare_(
    #     self,
    #     data,
    # ):
    #     pass
    #
    # def _prepare(
    #     self,
    #     data,
    #     outcome,
    #     treatment,
    #     adjustment,
    #     individual=None,
    #     **kwargs
    # ):
    #     # TODO: We should add a method like check_is_fitted.
    #     # This does not need the parameters like treatment. By calling this
    #     # function first, we then estimate quantites like ATE easily.
    #     r"""Prepare (fit the model) for estimating the quantities
    #         ATE: E[y|do(x_1)] - E[y|do(x_0)] = E_w[E[y|x=x_1,w] - E[y|x=x_0, w]
    #                                        := E_{adjustment}[
    #                                            Delta E[outcome|treatment,
    #                                                             adjustment]]
    #         CATE: E[y|do(x_1), z] - E[y|do(x_0), z] = E_w[E[y|x=x_1, w, z] -
    #                                                     E[y|x=x_0, w, z]]
    #         ITE: y_i(do(x_1)) - y_i(do(x_0))
    #         CITE: y_i(do(x_1))|z_i - y_i(do(x_0))|z_i
    #
    #     Parameters
    #     ----------
    #     data : DataFrame
    #     outcome : str
    #         Name of the outcome.
    #     treatment : str
    #         Name of the treatment.
    #     adjustment : set or list
    #         The adjutment set for the causal effect,
    #         i.e., P(outcome|do(treatment)) =
    #             \sum_{adjustment} P(outcome|treatment, adjustment)P(adjustment)
    #     individual : DataFrame, default to None
    #         The individual data for computing its causal effect.
    #
    #     Returns
    #     ----------
    #     np.array
    #     """
    #     pass

    def estimate(self, data=None, **kwargs):
        if not hasattr(self, '_is_fitted'):
            raise Exception('The estimator has not been fitted yet.')
    #
    # # def estimate(
    # #     self,
    # #     data,
    # #     outcome,
    # #     treatment,
    # #     adjustment,
    # #     quantity='ATE',
    # #     condition_set=None,
    # #     condition=None,
    # #     individual=None,
    # #     **kwargs
    # # ):
    # #     # This is the basic API for estimating a causal effect.
    #
    # #     # TODO: Note that we require that if parameters like treatment change
    # #     # then we should recall self.fit()
    # #     # method.
    # #     """General estimation method for quantities like ATE of
    # #     outcome|do(treatment).
    #
    # #     Parameter
    # #     ----------
    # #     data : DataFrame
    # #     outcome : str
    # #         Name of the outcome.
    # #     treatment : str
    # #         Name of the treatment.
    # #     adjustment : set or list
    # #         The valid adjustment set.
    # #     quantity: str
    # #         The type of desired quantity, including ATE, CATE, ITE and
    # #         CITE. Defaults to 'ATE'.
    # #     condition_set : set or list. Defaults to None
    # #     condition : list
    # #         A list whose length is the size of dataset and elements are
    # #         boolean such that we only perform the computation of
    # #         quantities if the corresponding element is True]. Defaults
    # #         to None.
    # #     individual : DataFrame. Defaults to None
    # #     kwargs : dict
    #
    # #     Raises
    # #     ----------
    # #     Exception
    # #         Raise exception if the quantity is not in ATE, CATE, ITE or CITE.
    #
    # #     Returns
    # #     ----------
    # #     float
    # #         The desired causal effect.
    # #     """
    # #     if quantity == 'ATE':
    # #         return self.estimate_ate(
    # #             data, outcome, treatment, adjustment, **kwargs
    # #         )
    # #     elif quantity == 'CATE':
    # #         return self.estimate_cate(
    # #             data, outcome, treatment, adjustment, condition_set,
    # #             condition, **kwargs
    # #         )
    # #     elif quantity == 'ITE':
    # #         return self.estimate_ite(
    # #             data, outcome, treatment, adjustment, individual, **kwargs
    # #         )
    # #     elif quantity == 'CITE':
    # #         return self.estimate_cite(
    # #             data, outcome, treatment, adjustment, condition_set,
    # #             condition, **kwargs
    # #         )
    # #     else:
    # #         raise Exception(
    # #             'Do not support estimation of quantities other'
    # #             'than ATE, CATE, ITE, or CITE'
    # #         )
    #
    # def estimate_ate(self, data, outcome, treatment, adjustment, **kwargs):
    #     """Estimate E[outcome|do(treatment=x1) - outcome|do(treatment=x0)]
    #
    #     Parameters
    #     ----------
    #     data : DataFrame
    #     outcome : str
    #         Name of the outcome.
    #     treatment : str
    #         Name of the treatment.
    #     adjustment : set or list of str
    #         The valid adjustment set.
    #
    #     Returns
    #     ----------
    #     float
    #     """
    #     return self._prepare(
    #         data, outcome, treatment, adjustment=adjustment, **kwargs
    #     ).mean()
    #
    # def estimate_cate(self, data, outcome, treatment, adjustment,
    #                   condition_set, condition, **kwargs):
    #     """Estimate E[outcome|do(treatment=x1), condition_set=condition
    #                 - outcome|do(treatment=x0), condition_set=condition]
    #
    #     Parameters
    #     ----------
    #     data : DataFrame
    #     outcome : str
    #         Name of the outcome.
    #     treatment : str
    #         Name of the treatment.
    #     adjustment : set or list of str
    #         The valid adjustment set.
    #     condition_set : set of str
    #     condition : boolean
    #         The computation will be performed only using data where conition is
    #         True.
    #
    #     Returns
    #     ----------
    #     float
    #     """
    #     # TODO: considering let the final model also be the function of
    #     # conditional variables.
    #     # modify the data for estmating cate.
    #     assert condition_set is not None, \
    #         'Need an explicit condition set to perform the analysis.'
    #
    #     new_data = data.loc[condition].drop(list(condition_set), axis=1)
    #     cate = self.estimate_ate(
    #         new_data, outcome, treatment, adjustment, **kwargs
    #     )
    #     return cate
    #
    # def estimate_ite(self, data, outcome, treatment, adjustment,
    #                  individual, **kwargs):
    #     assert individual is not None, \
    #         'Need an explicit individual to perform computation of individual'
    #     'causal effect.'
    #
    #     return self._prepare(
    #         data, outcome, treatment, adjustment,
    #         individual=individual, **kwargs
    #     )
    #
    # def estimate_cite(self, data, outcome, treatment, adjustment,
    #                   condition_set, condition, individual, **kwargs):
    #     assert individual is not None, \
    #         'Need an explicit individual to perform computation of individual'
    #     'causal effect.'
    #     assert condition_set is not None, \
    #         'Need an explicit condition set to perform the analysis.'
    #
    #     new_data = data.loc[condition].drop(list(condition_set), axis=1)
    #     cite = self._prepare(
    #         new_data, outcome, treatment, adjustment, individual, **kwargs
    #     )
    #     return cite
    #
    # def counterfactual_prediction(self):
    #     pass

    def __repr__(self):
        return to_repr(self)

