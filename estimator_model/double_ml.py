
from pandas.core.indexes.datetimes import date_range
from sklearn import clone
from copy import deepcopy

from .base_models import BaseEstLearner
from .utils import convert2array


class DoubleML(BaseEstLearner):
    r"""
    Double machine learning has two stages:
    In stage I, we
        1. fit a model (y_model) to predict outcome (y) from confounders (w) to
            get the predicted outcome (py);
        2. fit a model (x_model) to predict treatment (x) from confounders (w)
            to get the predicted treatement (px).
    In stage II, we
        fit a final model (yx_model) to predict y - py from x - px.

    See https://arxiv.org/pdf/1608.00060.pdf for reference.

    Attributes
    ----------

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

    def _prepare(self, data, outcome, treatment, adjustment, individual=None):
        self.y_model.fit(data[adjustment], data[outcome])
        self.x_model.fit(data[adjustment], data[treatment])

        py = self.y_model.predict(data[adjustment])
        px = self.x_model.predict(data[adjustment])

        self.yx_model.fit(data[treatment] - px, data[outcome] - py)
        # TODO: support cate, now only support ate
        result = self.yx_model.coef_
        return result

    def estimate_cate(self, data, outcome, treatment, adjustment,
                      condition_set, condition):
        raise NotImplementedError


class DML4CATE(BaseEstLearner):
    r"""Double machine learning for estimating CATE.
    # TODO: convert the einstein notations in this section to the usual ones.

    (- Skip this if you are only interested in the implementation.)
    A typical double machine learning for CATE solves the following treatment
    effect estimation (note that we use the einstein notation here):
        y^i = f_j^i(v^k) x^j + g^i(v^k, w^l) + \epsilon
        x^j = h^j(v^k, w^l) + \eta
    where f_j^i(v^k) is the CATE conditional on C=c and takes the form
        f_j^i(v^k) = F_j^i_k \rho^k
    with \rho^k: c \to R being v^k in the simplest case. Thus we have
        y^i = F_j^i_k \rho^k x^j + g^i(v^k, w^l) + \epsilon.
    The coefficients F_j^i_k can be estimated from the newly-formed data
    (\rho^k x^j, y^i) with linear regression where F_j^i_k are just
    coefficients of every feature in {1, 2, ..., k*j}. For a simple example, if
    both y and x only have one dimention, then the CATE for an input with
    covariate (c^1, c^2, c^3) will be F_1c^1, F_2c^2, and F_3c^3.

    (- Start of the implementation.)
    We implement a complicated version of the double machine learning same as
    the algorithm described in the [1]:
        1. Let k (cf_folds in our class) be an int. Form a k-fold random
        partition {..., (train_data_i, test_data_i), ...,
        (train_data_k, test_data_k)}.
        2. For each i, train y_model and x_model on train_data_i, then evaluate
        their performances in test_data_i whoes results will be saved as
        (y_hat_k, x_hat_k). All (y_hat_k, x_hat_k) will be the form
        (y_hat, x_hat).
        3. Define the differences
            y_prime = y - y_hat,
            x_prime = (x - x_hat) \otimes v.
        Then form the new dataset (y_prime, x_prime).
        4. Perform linear regression on the dataset (y_prime, x_prime) whoes
        coefficients will be saved in a vector F. The estimated CATE given V=v
        will just be
            F \dot v.
        On the other hand, the ATE can be simply estimated by taking average
        of F \dot v over the original data.

    Attributes
    ----------

    Methods
    ----------

    Reference
    ----------
    [1] V. Chernozhukov, et al. Double Machine Learning for Treatment and Causal
        Parameters. arXiv:1608.00060.
    """

    def __init__(
        self,
        x_model,
        y_model,
        yx_model=None,
        cf_fold=1,
        random_state=2022,
    ):
        self.x_model = clone(x_model)
        self.y_model = clone(y_model)
        self.yx_model = clone(yx_model)
        self.cf_fold = cf_fold
        super().__init__(random_state=random_state)

    def fit(
        self,
        data,
        outcome,
        treatment,
        adjustment=None,
        covariate=None,
    ):
        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )

    def _prepare():
        pass

    def estimate():
        pass

    def _cross_fit():
        pass

    def _fit_1st_stage(self):
        model_list = []
        para_hat = []
        new_index = []
        return model_list, para_hat, new_index

    def _fit_2nd_stage(self):
        pass
