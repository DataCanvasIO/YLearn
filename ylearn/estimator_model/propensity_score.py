import numpy as np

from sklearn import linear_model

from .base_models import BaseEstModel

# TODO: consider treatments other than binary treatment.
"""Contents in this file are not complete. Avoid using them
"""


class PropensityScore:
    """
    The class wrapping a machine learning model to be able to predict the
    propensity score
        ps_{x=xt}(w) = P(x=xt|w)

    Attributes
    ----------
    ml_model_dic : dict
        Default model dictionary where keys are name of machine learning
        models while values are corresponding models.
    ml_model : MLModel, optional

    Methods
    ----------
    fit(data, treatment, adjustment)
    predict(data, adjustment)
    predict_proba(data, adjustment)
        Use this method when the treatment is discrete rather than binary.
    fit_predict(train_data, treatment, adjustment, pre_data)
        Combination of fit() and predict(), where the model is trained in the
        train_data and predict the pre_data if it's not None, otherwise the
        model will predict labels of the train_data.
    """

    def __init__(self, ml_model=None):
        """
        Parameters
        ----------
        ml_model : str, optional. Defaults to None.
        """
        self.ml_model_dic = {"LogisticR": linear_model.LogisticRegression()}

        if ml_model is None:
            ml_model = self.ml_model_dic["LogisticR"]
        if type(ml_model) is str:
            ml_model = self.ml_model_dic[ml_model]

        self.ml_model = ml_model

    def fit(self, data, treatment, adjustment):
        self.ml_model.fit(data[adjustment], data[treatment])

    def predict(self, data, adjustment):
        return self.ml_model.predict(data[adjustment])

    def predict_proba(self, data, adjustment, target=None):
        p = self.ml_model.predict_proba(data[adjustment])
        try:
            p_ = p.transpose()[target]
        except TypeError:
            p_ = p.transpose(0, 1)[target]
        return p_

    def fit_predict(self, train_data, treatment, adjustment, pre_data=None):
        """Combination of fit() and predict(), where the model is trained in the
        train_data and predict the pre_data if it's not None, otherwise the
        model will predict labels of the train_data.

        Parameters
        ----------
        train_data : pd.DataFrame
        treatment : str
        adjustment : list of str
        pre_data : pd.DataFrame. Defaults to None

        Returns
        ----------
        np.array
        """
        self.ml_model.fit(train_data, treatment, adjustment)
        if pre_data is None:
            pre_data = train_data
        return self.predict(pre_data, adjustment)


class InversePbWeighting(BaseEstModel):
    r"""
    Inverse Probability Weighting. The identification equation is defined as
        E[y|do(x)] = E[I(X=x)y / P(x|W)],
    where I is the indicator function and W is the adjustment set.
    For binary treatment, we have
        ATE = E[y|do(x=1) - y|do(x=0)] =
            E[I(X=1)y / e(W)] - E[E[I(X=0)y / (1 - ps(w))]
    where ps(w) is the propensity score
        ps(w) = P(x|W).
    Therefore, the final estimated ATE should be
        1 / n_1 \sum_{i| x_i = 1} y_i / e(w_i)
            - 1 / n_2 \sum_{j| x_j = 0} y_j / (1 - e(w_i)).

    Attributes
    ----------
    ml_model_dic : dict
        A dictionary of default machine learning sklearn models currently
        including
            'LR': LinearRegression
            'LogisticR': LogisticRegression.

    ps_model : PropensityScore
        The wrapped machine learning model for the propensity score.

    Methods
    ----------
    _prepare4est(data, outcome, treatment, adjustment, individual=None)
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
    # TODO: support more methods, treatments other than binary treatment.

    def __init__(self, ps_model):
        super().__init__()
        if type(ps_model) is str:
            ps_model = self.ml_model_dic[ps_model]

        self.ps_model = PropensityScore(ml_model=ps_model)

    def _prepare4est(self, data, outcome, treatment, adjustment, individual=None):
        self.ps_model.fit(data, treatment, adjustment)
        ps = self.ps_model.predict_proba(data, adjustment, target=1)
        o = np.ones(len(ps))
        eps = 1e-7  # numerical stability
        # The following computation is valid only for binary treatment.
        # TODO: consider continuous treatment.
        result = (data[treatment] * data[outcome] + eps) / (ps + eps) - (
            (o - data[treatment]) * data[outcome] + eps
        ) / (o - ps + eps)
        return result

    # The following method is the old version.
    # def estimate_ate(self, data, outcome, treatment, adjustment):
    #     self.ps_model.fit(data, treatment, adjustment)
    #     t1_data = data.loc[data[treatment] > 0]
    #     t0_data = data.loc[data[treatment] <= 0]
    #     t1_ew = self.ps_model.predict(t1_data, adjustment)
    #     t0_ew = np.ones(len(t0_data)) \
    #         - self.ps_model.predict(t0_data, adjustment)
    #     result = (t1_data[outcome] / t1_ew).mean() \
    #         - (t0_data[outcome] / t0_ew).mean()
    #     return result
