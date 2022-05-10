from itertools import product

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from ylearn.estimator_model.double_ml import DML4CATE
from ylearn.exp_dataset.exp_data import single_continuous_treatment, single_binary_treatment
from . import _dgp
from ._common import validate_leaner

_test_settings = {
    # data_generator: (x_model,y_model )
    _dgp.generate_data_x1b_y1: (RandomForestClassifier(),
                                RandomForestRegressor(),
                                ),
    _dgp.generate_data_x1b_y2: (RandomForestClassifier(),
                                RandomForestRegressor(),
                                ),
    _dgp.generate_data_x1m_y1: (RandomForestClassifier(),
                                RandomForestRegressor(),
                                ),
    # _dgp.generate_data_x2b_y1: (RandomForestClassifier(),
    #                             RandomForestRegressor(),
    #                             ),
    # _dgp.generate_data_x2b_y2: (RandomForestClassifier(),
    #                             RandomForestRegressor(),
    #                             ),
}

_test_settings_to_be_fixed = {
    # data_generator: (x_model,y_model )
    _dgp.generate_data_x2b_y1: (RandomForestClassifier(),
                                RandomForestRegressor(),
                                ),
    _dgp.generate_data_x2b_y2: (RandomForestClassifier(),
                                RandomForestRegressor(),
                                ),
}


@pytest.mark.parametrize('dg', _test_settings.keys())
def test_double_ml(dg):
    x_model, y_model = _test_settings[dg]
    estimator = DML4CATE(x_model=x_model, y_model=y_model, cf_fold=1, random_state=2022, is_discrete_treatment=True)
    validate_leaner(dg, estimator, check_fitted=False)


@pytest.mark.parametrize('dg', _test_settings_to_be_fixed.keys())
# @pytest.mark.xfail(reason='to be fixed')
def test_double_ml_to_be_fixed(dg):
    x_model, y_model = _test_settings_to_be_fixed[dg]
    estimator = DML4CATE(x_model=x_model, y_model=y_model, cf_fold=1, random_state=2022, is_discrete_treatment=True)
    validate_leaner(dg, estimator, check_fitted=False)


########################################################################################

def exp_te(x):
    return np.exp(2 * x)


def test_dml_single_continuous_treatment():
    train, val, treatment_effect = single_continuous_treatment()
    adjustment = train.columns[:-4]
    covariate = 'c_0'
    outcome = 'outcome'
    treatment = 'treatment'

    dml = DML4CATE(
        x_model=RandomForestRegressor(),
        y_model=RandomForestRegressor(),
        cf_fold=3,
    )
    dml.fit(
        train,
        outcome,
        treatment,
        adjustment,
        covariate,
    )
    dat = np.array(list(product(np.arange(0, 1, 0.01), repeat=1))).ravel()
    data_test = pd.DataFrame({'c_0': dat})
    true_te = np.array([exp_te(xi) for xi in data_test[covariate]])
    ested_te = dml.estimate(data_test).ravel()
    s = r2_score(true_te, ested_te)
    print(s)


def test_dml_with_single_continuous_treatment_with_covariate_transformer():
    train, val, treatment_effect = single_continuous_treatment()
    adjustment = train.columns[:-4]
    covariate = 'c_0'
    outcome = 'outcome'
    treatment = 'treatment'

    dml = DML4CATE(
        x_model=RandomForestRegressor(),
        y_model=RandomForestRegressor(),
        cf_fold=3,
        covariate_transformer=PolynomialFeatures(degree=3, include_bias=False)
    )
    dml.fit(
        train,
        outcome,
        treatment,
        adjustment,
        covariate,
    )
    dat = np.array(list(product(np.arange(0, 1, 0.01), repeat=1))).ravel()
    data_test = pd.DataFrame({'c_0': dat})
    true_te = np.array([exp_te(xi) for xi in data_test[covariate]])
    ested_te = dml.estimate(data_test).ravel()
    s = r2_score(true_te, ested_te)
    print(s)


def test_dml_single_binary_treatment():
    train1, val1, treatment_effect1 = single_binary_treatment()

    def exp_te(x): return np.exp(2 * x[0])

    n = 1000
    n_x = 4
    X_test1 = np.random.uniform(0, 1, size=(n, n_x))
    X_test1[:, 0] = np.linspace(0, 1, n)
    data_test_dict = {
        'c_0': X_test1[:, 0],
        'c_1': X_test1[:, 1],
        'c_2': X_test1[:, 2],
        'c_3': X_test1[:, 3],
    }
    data_test1 = pd.DataFrame(data_test_dict)
    true_te = np.array([exp_te(x_i) for x_i in X_test1])

    adjustment1 = train1.columns[:-7]
    covariate1 = train1.columns[-7:-3]
    # t_effect1 = train1['t_effect']
    treatment = 'treatment'
    outcome = 'outcome'
    dml1 = DML4CATE(
        x_model=RandomForestClassifier(),
        y_model=RandomForestRegressor(),
        cf_fold=1,
        is_discrete_treatment=True,
    )
    dml1.fit(
        data=train1,
        outcome=outcome,
        treatment=treatment,
        adjustment=adjustment1,
        covariate=covariate1,
    )
    predicted = dml1.estimate(data_test1)
