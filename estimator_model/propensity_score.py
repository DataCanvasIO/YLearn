import numpy as np

from sklearn import linear_model

from .base_models import BaseEstLearner
# TODO: consider treatments other than binary treatment.


class PropensityScore:
    def __init__(self, ml_model='LogisticR'):
        self.ml_model_dic = {
            'LogisticR': linear_model.LogisticRegression()
        }

        if type(ml_model) is str:
            model = self.ml_model_dic[ml_model]

        self.ml_model = model

    def fit(self, data, treatment, adjustment):
        self.ml_model.fit(data[adjustment], data[treatment])

    def predict(self, data, adjustment):
        return self.ml_model.predict(data[adjustment])

    def fit_predict(self, train_data, treatment, adjustment, pre_data):
        self.ml_model.fit(train_data, treatment, adjustment)
        return self.predict(pre_data, adjustment)


class InversePorbWeighting(BaseEstLearner):
    """
    Inverse Probability Weighting. The identification equation is defined as
        E[y|do(x)] = E[I(X=x)y / P(x|W)],
    where I is the indicator function and W is the adjustment set.
    For binary treatment, we have
        ATE = E[y|do(x=1) - y|do(x=0)] = 
            E[I(X=1)y / e(W)] - E[E[I(X=0)y / (1 - e(w))]
    where e(w) is the propensity score 
        e(w) = P(x|W).
    Therefore, the final estimated ATE should be
        1 / n_1 \sum_{i| x_i = 1} y_i / e(w_i)
            - 1 / n_2 \sum_{j| x_j = 0} y_j / (1 - e(w_i)).
    """
    # TODO: support more methods.

    def __init__(self, ew_model) -> None:
        super.__init__()
        if type(ew_model) is str:
            ew_model = self.ml_model_dic[ew_model]

        self.ew_model = PropensityScore(ml_model=ew_model)

    def prepare(self, data, outcome, treatment, adjustment, individual=None):
        self.ew_model.fit(data, treatment, adjustment)
        ew = self.ew_model.predict(data, adjustment)
        o = np.ones(len(ew))
        # The following computation is valid only for binary treatment.
        # TODO: consider continuous treatment.
        result = data[treatment] * data[outcome] / ew \
            - (o - data[treatment]) * data[outcome] / (o - ew)
        return result

    # The following method is the old version.
    # def estimate_ate(self, data, outcome, treatment, adjustment):
    #     self.ew_model.fit(data, treatment, adjustment)
    #     t1_data = data.loc[data[treatment] > 0]
    #     t0_data = data.loc[data[treatment] <= 0]
    #     t1_ew = self.ew_model.predict(t1_data, adjustment)
    #     t0_ew = np.ones(len(t0_data)) \
    #         - self.ew_model.predict(t0_data, adjustment)
    #     result = (t1_data[outcome] / t1_ew).mean() \
    #         - (t0_data[outcome] / t0_ew).mean()
    #     return result
