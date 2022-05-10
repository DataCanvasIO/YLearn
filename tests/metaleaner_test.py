from itertools import product

import pytest
from sklearn import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, MultiTaskLasso

from ylearn.estimator_model.meta_learner import SLearner, TLearner, XLearner
from . import _dgp
from ._common import validate_leaner

_test_settings = {
    # data_generator: model
    _dgp.generate_data_x1b_y1: GradientBoostingRegressor(),
    _dgp.generate_data_x2b_y1: GradientBoostingRegressor(),
    _dgp.generate_data_x1b_y2: LinearRegression(),
    _dgp.generate_data_x2b_y2: MultiTaskLasso(),
    _dgp.generate_data_x1m_y1: GradientBoostingRegressor(),
    _dgp.generate_data_x1b_y1_w5v0: GradientBoostingRegressor(),
    _dgp.generate_data_x2b_y1_w5v0: GradientBoostingRegressor(),
    _dgp.generate_data_x1b_y2_w5v0: LinearRegression(),
    # _dgp.generate_data_x2b_y2_w5v0: MultiTaskLasso(),
    _dgp.generate_data_x2b_y2_w5v0: LinearRegression(),
    _dgp.generate_data_x1m_y1_w5v0: GradientBoostingRegressor(),
}


@pytest.mark.parametrize('dg,combined',
                         product(_test_settings.keys() - [_dgp.generate_data_x2b_y2,
                                                          _dgp.generate_data_x2b_y2_w5v0],
                                 [True, False]))
def test_sleaner(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, SLearner(model=clone(model)), fit_kwargs=dict(combined_treatment=combined))


@pytest.mark.parametrize('dg,combined', product([_dgp.generate_data_x2b_y2_w5v0, ], [True, False]))
# @pytest.mark.xfail(reason='to be fixed')
def test_sleaner_to_be_fixed(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, SLearner(model=clone(model)), fit_kwargs=dict(combined_treatment=combined))


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_tleaner(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, TLearner(model=clone(model)), fit_kwargs=dict(combined_treatment=combined))


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_xleaner(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, XLearner(model=clone(model)), fit_kwargs=dict(combined_treatment=combined))
