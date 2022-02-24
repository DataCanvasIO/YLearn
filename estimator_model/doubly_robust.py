import numpy as np

from copy import deepcopy
from sklearn import linear_model

from . import propensity_score, base_models
from .base_models import BaseEstLearner


class DoublyRobust(BaseEstLearner):
    """The doubly robust estimator has 3 steps
    (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3070495/pdf/kwq439.pdf
    for reference):
    1. Split the data into the treatment group (x1_data) and control group
        (x0_data),then fit two models in these groups to predict the outcome y:
       y1_i = x1_model(w_i), y0_i = x0_model(w_i)
    2. Estimate the propensity score
       ps(w_i) = ps_model.predict(w_i)
    3. Calculate the final result (expected effect)
       1/n \\sum_i^n [(\frac{y_i x_i}{ps(w_i)}
       - \\frac{x_i - ps(w_i)}{ps(w_i)}y1_i)
       - (\\frac{y_i(1-x_i)}{1-ps(w_i)}
       + \\frac{y0_i(x_i - ps(w_i))}{1-ps(w_i)})]
    """

    def __init__(self, ps_model, est_model):
        super().__init__()

        if type(ps_model) is str:
            ps_model = self.ml_model_dic[ps_model]
        if type(est_model) is str:
            est_model = self.ml_model_dic[est_model]

        self.ps_model = propensity_score.PropensityScore(ml_model=ps_model)
        self.x1_model = est_model
        self.x0_model = deepcopy(est_model)

    def prepare(self, data, outcome, treatment, adjustment, individual=None):
        # step 1
        x1_data = data.loc[data[treatment] > 0]
        x0_data = data.loc[data[treatment] <= 0]
        self.x1_model.fit(x1_data[adjustment], x1_data[outcome])
        self.x0_model.fit(x0_data[adjustment], x0_data[outcome])

        # step 2
        self.ps_model.fit(data, treatment, adjustment)

        # step 3
        if individual:
            data_ = individual[adjustment]
        else:
            data_ = data[adjustment]

        y1 = self.x1_model.predict(data_)
        y0 = self.x0_model.predict(data_)
        x, y = data_[treatment], data_[outcome]
        o, ps = np.ones(len(data_)), self.ps_model.predict(data_, adjustment)
        one_x, one_ps = o - x, o - ps
        x_ps = x - ps
        result = (
            (y * x / ps - y1 * x_ps / ps)
            - (y * one_x / one_ps + y0 * x_ps / one_ps)
        )
        return result
