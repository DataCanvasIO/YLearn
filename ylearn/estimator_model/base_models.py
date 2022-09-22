import numpy as np
import pandas as pd
import inspect

from ylearn.utils import to_repr
from collections import defaultdict

# from ..utils._common import check_cols

# TODO: add support for assigning different treatment values for different examples for all models.


class BaseEstModel:
    """
    Base class for various estimation learner.

    Attributes
    ----------

    Methods
    ----------
    """

    def __init__(
        self,
        random_state=2022,
        is_discrete_treatment=False,
        is_discrete_outcome=False,
        _is_fitted=False,
        # is_discrete_instrument=False,
        categories="auto",
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

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values() if p.name != "self"]
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):

        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set parameters.

        Returns
        -------
        params : dict
            parameters

        Raises
        ------
        self
            An instance of estimator

        ValueError
            raise error if wrong parameters are given
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def _validate_params(self):
        """Validate types and values of constructor parameters
        The expected type and values must be defined in the `_parameter_constraints`
        class attribute, which is a dictionary `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for a description of the
        accepted constraints.
        """
        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(deep=False),
            caller_name=self.__class__.__name__,
        )

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
        if not hasattr(self, "_is_fitted"):
            raise Exception("The estimator has not been fitted yet.")

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
