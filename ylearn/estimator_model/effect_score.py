"""
Beside effect score which measures the ability of estimating the causal
effect, we should also implement training_score which can measure
performances of machine learning models.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

from .double_ml import DML4CATE
from .utils import convert2array, get_wv


class RLoss(DML4CATE):
    def __init__(
        self,
        x_model,
        y_model,
        cf_fold=1,
        adjustment_transformer=None,
        covariate_transformer=None,
        random_state=2022,
        is_discrete_treatment=False,
        categories='auto',
    ):
        self.cf_fold = cf_fold
        self.x_model = clone(x_model)
        self.y_model = clone(y_model)

        self.adjustment_transformer = adjustment_transformer
        self.covariate_transformer = covariate_transformer

        self.x_hat_dict = defaultdict(list)
        self.y_hat_dict = defaultdict(list)
        self.x_hat_dict['is_fitted'].append(False)
        self.y_hat_dict['is_fitted'].append(False)

        super().__init__(
            random_state=random_state,
            is_discrete_treatment=is_discrete_treatment,
            categories=categories,
        )

    def fit(
        self,
        data,
        outcome,
        treatment,
        adjustment=None,
        covariate=None,
    ):
        assert covariate is not None, \
            'Need covariates to use RLoss.'

        super().fit(
            data, outcome, treatment,
            adjustment=adjustment,
            covariate=covariate,
        )

        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )
        self._v = v
        self._y_d = y.shape[1]
        cfold = self.cf_fold

        if self.adjustment_transformer is not None and w is not None:
            w = self.adjustment_transformer.fit_transform(w)

        if self.covariate_transformer is not None and v is not None:
            v = self.covariate_transformer.fit_transform(v)

        if self.is_discrete_treatment:
            if self.categories == 'auto' or self.categories is None:
                categories = 'auto'
            else:
                categories = list(self.categories)

            # convert discrete treatment features to onehot vectors
            self.transformer = OneHotEncoder(categories=categories)
            self.transformer.fit(x)
            x = self.transformer.transform(x).toarray()
        else:
            self.transformer = None

        self._x_d = x.shape[1]
        wv = get_wv(w, v)

        # step 1: split the data
        if cfold > 1:
            cfold = int(cfold)
            folds = [
                KFold(n_splits=cfold).split(x), KFold(n_splits=cfold).split(y)
            ]
        else:
            folds = None

        # step 2: cross fit to give the estimated y and x
        self.x_hat_dict, self.y_hat_dict = super()._fit_1st_stage(
            self.x_model, self.y_model, y, x, wv, folds=folds
        )
        x_hat = self.x_hat_dict['paras'][0].reshape((x.shape))
        y_hat = self.y_hat_dict['paras'][0].reshape((y.shape))

        x_prime = x - x_hat
        y_prime = y - y_hat

        self.x_hat_dict['res'].append(x_prime)
        self.y_hat_dict['res'].append(y_prime)

        self._is_fitted = True

        return self

    def score(self, test_estimator):
        x_prime, y_prime = self.x_hat_dict['res'], self.y_hat_dict['res']
        v = self._v

        if test_estimator.covariate is None:
            assert test_estimator.adjustment is not None
            names = test_estimator.adjustment
        else:
            names = test_estimator.covariate

        data_dict = {}
        for i, name in enumerate(names):
            data_dict[name] = v[:, i]
        test_data = pd.DataFrame(data_dict)

        # shape (n, y_d, x_d)
        # TODO: may need more modifications
        test_effect = test_estimator.estimate(data=test_data).reshape(
            v.shape[0], self._y_d, self._x_d
        )
        y_pred = np.einsum('nij, nj->ni', test_effect, x_prime)
        rloss = np.mean((y_prime - y_pred)**2, axis=0)

        return rloss