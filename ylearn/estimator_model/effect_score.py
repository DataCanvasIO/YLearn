"""
Beside effect score which measures the ability of estimating the causal
effect, we should also implement training_score measuring
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
        self.yx_model = None

        super().__init__(
            x_model=x_model,
            y_model=y_model,
            cf_fold=cf_fold,
            adjustment_transformer=adjustment_transformer,
            covariate_transformer=covariate_transformer,
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
        combined_treatment=True,
    ):
        assert covariate is not None, \
            'Need covariates to use RLoss.'

        self.combined_treatment = combined_treatment

        super().fit(
            data, outcome, treatment,
            adjustment=adjustment,
            covariate=covariate,
        )

        y, x, w, v = convert2array(
            data, outcome, treatment, adjustment, covariate
        )
        self._w = w
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
        x_prime, y_prime = self.x_hat_dict['res'][0], self.y_hat_dict['res'][0]
        v = self._v
        w = self._w

        data_dict = {}
        if self.covariate is not None:
            for i, name in enumerate(test_estimator.covariate):
                data_dict[name] = v[:, i]
        if self.adjustment is not None:
            for i, name in enumerate(test_estimator.adjustment):
                data_dict[name] = w[:, i]

        test_data = pd.DataFrame(data_dict)

        # shape (n, y_d, x_d)
        # TODO: may need more modifications

        test_effect = test_estimator.estimate(data=test_data,)

        if self.is_discrete_treatment:
            if self.combined_treatment:
                x_d = 1
                x_prime = x_prime[:, test_estimator.treat].reshape(-1, 1)
        else:
            pass

        test_effect = test_effect.reshape(v.shape[0], self._y_d, x_d)
        y_pred = np.einsum('nij, nj->ni', test_effect, x_prime)
        rloss = np.mean((y_prime - y_pred)**2, axis=0)

        return rloss


class PredLoss:
    def __init__(self) -> None:
        pass

    def fit(self,):
        pass

    def score(self):
        pass


# class Score:
#     def __init__(self):
#         pass

#     def fit(self, data):
#         pass

#     def score(self,):
#         pass
