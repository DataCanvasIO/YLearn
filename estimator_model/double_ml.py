from copy import deepcopy
from .base_models import BaseEstLearner

class DoubleML(BaseEstLearner):
    """
    Double machine learning has two stages:
    In stage I, we
        1. fit a model (y_model) to predict outcome (y) from confounders (w) to
            get the predicted (py);
        2. fit a model (x_model) to predict treatment (x) from confounders (w)
            to get the predicted treatement (px).
    In stage II, we
        fit a final model (yx_model) to predict y - py from x - px.

    See https://arxiv.org/pdf/1608.00060.pdf for reference.
    """
    # TODO: support more final models, e.g., non-parametric models.

    def __init__(self, y_model, x_model, yx_model):
        super().__init__()
        if type(y_model) is str:
            y_model = self.ml_model_dic[y_model]
        if type(x_model) is str:
            x_model = deepcopy(self.ml_model_dic[x_model])
        if type(yx_model) is str:
            yx_model = deepcopy(self.ml_model_dic[yx_model])

        self.y_model = y_model
        self.x_model = x_model
        self.yx_model = yx_model

    def prepare(self, data, outcome, treatment, adjustment, individual=None):
        self.y_model.fit(data[adjustment], data[outcome])
        self.x_model.fit(data[adjustment], data[treatment])

        py = self.y_model.predict(data[adjustment])
        px = self.x_model.predict(data[adjustment])

        self.yx_model.fit(data[treatment] - px, data[outcome] - py)
        # TODO: support cate, now only support ate
        result = self.yx_model.coef_
        return result

    def estimate_cate(self, data, outcome, treatment, adjustment, condition_set, condition):
        raise NotImplementedError
