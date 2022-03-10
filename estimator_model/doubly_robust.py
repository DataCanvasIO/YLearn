import numpy as np

from copy import deepcopy
from sklearn import linear_model

from . import propensity_score, base_models
from .base_models import BaseEstLearner


class DoublyRobust(BaseEstLearner):
    r"""
    The doubly robust estimator has 3 steps
    (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3070495/pdf/kwq439.pdf
    for reference, see also the slide
    https://www4.stat.ncsu.edu/~davidian/double.pdf):
    1. Split the data into the treatment group (xt_data) and control group
        (x0_data),then fit two models in these groups to predict the outcome y:
       yt_i = xt_model.predict(w_i), y0_i = x0_model.predict(w_i)
    2. Estimate the propensity score
       ps(w_i) = ps_model.predict(w_i)
    3. Calculate the final result (expected effect, this is the case for binary
        treatment)
        1/n \sum_i^n [
            (
                \frac{y_i x_i}{ps(w_i)}
                -\frac{x_i - ps(w_i)}{ps(w_i)}yt_i
            )
            -(
                \frac{y_i(1-x_i)}{1-ps(w_i)}
                +\frac{y0_i(x_i - ps(w_i))}{1-ps(w_i)}
            )
            ]
        If the treatment is not binary, we calculate (note that ps_model should
        be a multi-classification here):
        1/n \sum_i^n [
            (\frac{I(x_i=xt)y_i}{ps_{x_i=xt}(w_i)}
            - yt_i\frac{I(x_i=xt)-ps_{x_i=xt}(w_i)}{ps_{x_i=xt}(w_i)})
            (\frac{I(x_i=x0)y_i}{ps_{x_i=x0}(w_i)}
            - y0_i\frac{I(x_i=x0)-ps_{x_i=x0}(w_i)}{ps_{x_i=x0}(w_i)})
        ]

    Attributes
    ----------
    ml_model_dic : dict
        A dictionary of default machine learning sklearn models currently
        including
            'LR': LinearRegression
            'LogistR': LogisticRegression.
    ps_model : PropensityScore
    xt_model : MLModel, optional
        The machine learning model trained in the treated group.
    x0_model : MLModel, optional
        The machine learning model trained in the control group.

    Methods
    ----------
    """

    def __init__(self, ps_model, est_model):
        """
        Parameters
        ----------
        ps_model : PropensityScore or str
        est_model : str, optional
            All valid est_model should implement the fit and predict methods
            which can be done by wrapping the corresponding machine learning
            models with the class MLModel.
        """
        super().__init__()

        if type(ps_model) is str:
            ps_model = self.ml_model_dic[ps_model]
        if type(est_model) is str:
            est_model = self.ml_model_dic[est_model]

        # TODO: should we train different regression models for different
        # treatment groups or simply train one model once to model the
        # relation between treatment and adjustment?
        self.ps_model = propensity_score.PropensityScore(ml_model=ps_model)
        self.xt_model = est_model
        self.x0_model = deepcopy(est_model)

    def prepare(self, data, outcome, treatment, adjustment,
                individual=None, treatment_value=None):
        # TODO: categorical treatment. Currently I hope to convert categorical
        # treatments to integers such that treatment={1, 2,...,n} where n is
        # the number of different treatments. Is there any better idea?

        # The default treatment group data are those whose data[treatment] == 1
        num_treatment = data[treatment].value_counts().shape[0]
        if treatment_value is None or num_treatment == 2:
            treatment_value = 1

        # step 1, fit the treatment group model and the control group model
        xt_data = data.loc[data[treatment] == treatment_value]
        x0_data = data.loc[data[treatment] == 0]
        self.xt_model.fit(xt_data[adjustment], xt_data[outcome])
        self.x0_model.fit(x0_data[adjustment], x0_data[outcome])

        # step 2, fit the propensity score model
        self.ps_model.fit(data, treatment, adjustment)

        # step 3, calculate the final result
        if individual:
            data_ = individual
        else:
            data_ = data
        yt = self.xt_model.predict(data_[adjustment])
        y0 = self.x0_model.predict(data_[adjustment])
        x, y = data_[treatment], data_[outcome]

        if num_treatment == 2:
            # binary case
            o, ps = np.ones(len(data_)), self.ps_model.predict(
                data_, adjustment)
            one_x, one_ps = o - x, o - ps
            x_ps = x - ps

            result = (
                (y * x / ps - yt * x_ps / ps)
                - (y * one_x / one_ps + y0 * x_ps / one_ps)
            )
        else:
            # discrete case
            x0_index = (x == 0).astype(int)
            x0_prob = self.ps_model.predict_prob(data_, adjustment, 0)
            xt_index = (x == treatment_value).astype(int)
            xt_prob = self.ps_model.predict_prob(
                data_, adjustment, treatment_value
            )
            result = (
                (xt_index * y / xt_prob - yt * (xt_index - xt_prob) / xt_prob)
                - (x0_index * y / x0_prob
                   - y0 * (x0_index - x0_prob) / x0_prob)
            )
        return result
