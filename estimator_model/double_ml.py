from copy import deepcopy
from collections import defaultdict

from sklearn.model_selection import KFold
from sklearn import clone

from .base_models import BaseEstLearner
from .utils import convert2array, nd_kron


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
    covariate (c^1, c^2, c^3) will be F_1c^1, F_2c^2, and F_3c^3. #TODO:
    However, note that letting \rho^k simply be v^k actually implicitly assume
    that the value of v^k is small thus is a good approximation of \rho^k. 


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
    [1] V. Chernozhukov, et al. Double Machine Learning for Treatment and
        Causal Parameters. arXiv:1608.00060.
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
        """Note that we will use the following notation in this method:
                y: out

        Args:
            data (_type_): _description_
            outcome (_type_): _description_
            treatment (_type_): _description_
            adjustment (_type_, optional): _description_. Defaults to None.
            covariate (_type_, optional): _description_. Defaults to None.
        """
        self.outcome = outcome
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate
        
        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )
        cv = self.cf_fold
        random_state = self.random_state

        # step 1: split the data
        folds = KFold(n_splits=cv, random_state=random_state).split(x, y)

        # step 2: cross fit to give the estimated y and x
        x_hat_dict = self._cross_fit(self.x_model, w, v, folds=folds)
        y_hat_dict = self._cross_fit(self.y_model, w, v, folds=folds)
        x_hat, y_hat = x_hat_dict['paras'], y_hat_dict['paras']
        
        # step 3: calculate the differences
        x_diff = x - x_hat
        y_diff = y - y_hat
        x_prime = nd_kron(x_diff.reshape(-1, 1), v.reshape(-1, 1))

        # step 4: fit the regression problem
        self.yx_model.fit(x_prime, y_diff)
        
        return self
    def _prepare(self, data=None):
        pass

    def estimate():
        pass

    def _cross_fit(self, model, *args, **kwargs):
        try:
            folds = kwargs.pop('folds')
        except:
            folds = KFold(
                n_splits=self.cf_fold, random_state=self.random_state
            ).split(args[0])
        else:
            folds = None
            
        fitted_result = defaultdict(list)
        if folds is None:
            model.fit(*args, **kwargs)
            p_hat = model.predict(*args, **kwargs)
            fitted_result['models'].append(clone(model))
            fitted_result['paras'].append(p_hat)
            return fitted_result
        
        for i, (train_id, test_id) in enumerate(folds):
            model_ = clone(model)
            new_args_ = tuple(arg[train_id] for arg in args)
            fitted_result['models'].append(model_)
            model_.fit(*args, **kwargs)
            

    def _fit_1st_stage(self):
        model_list = []
        para_hat = []
        new_index = []
        return model_list, para_hat, new_index

    def _fit_2nd_stage(self):
        pass
