import sys
from itertools import product

import pandas as pd
import numpy as np
np.random.seed(0)

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

from ylearn.exp_dataset.exp_data import single_continuous_treatment, single_binary_treatment
from ylearn.estimator_model.double_ml import DML4CATE
from ylearn.effect_interpreter.ce_interpreter import CEInterpreter

import pytest

need_at_least_py37 = pytest.mark.skipif(sys.version_info[1] <= 6, reason="skip if <=python3.6")


@need_at_least_py37
class TestPolicyModel:

    def setup_class(cls):
        pass

    def run_interprete(self, data_func, dml):
        train, val, treatment_effect = data_func()
        adjustment = train.columns[:-4]
        covariate = 'c_0'
        outcome = 'outcome'
        treatment = 'treatment'

        def exp_te(x): return np.exp(2 * x)

        dat = np.array(list(product(np.arange(0, 1, 0.01), repeat=1))).ravel()

        data_test = pd.DataFrame({'c_0': dat})
        true_te = np.array([exp_te(xi) for xi in data_test[covariate]])

        dml.fit(
            train,
            outcome,
            treatment,
            adjustment,
            covariate,
        )
        cei = CEInterpreter(max_depth=3)
        cei.fit(data=data_test, est_model=dml)

        ret_list = cei.decide(data_test)

        assert ret_list is not None
        assert ret_list.shape[0] == data_test.shape[0]
        return ret_list

    @pytest.mark.parametrize('covariate_transformer', [None, PolynomialFeatures(degree=3, include_bias=False)])
    def test_binary(self, covariate_transformer):
        dml = DML4CATE(
            x_model=RandomForestClassifier(),
            y_model=RandomForestRegressor(),
            cf_fold=2,
            covariate_transformer=covariate_transformer
        )
        self.run_interprete(single_binary_treatment, dml)

    @pytest.mark.parametrize('covariate_transformer', [None, PolynomialFeatures(degree=3, include_bias=False)])
    def test_continuous(self, covariate_transformer):
        dml = DML4CATE(
            x_model=RandomForestRegressor(),
            y_model=RandomForestRegressor(),
            cf_fold=3,
            covariate_transformer=covariate_transformer
        )
        self.run_interprete(single_continuous_treatment, dml)
