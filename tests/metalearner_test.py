from itertools import product

import pytest
from sklearn import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from ylearn.estimator_model.meta_learner import SLearner, TLearner, XLearner
from . import _dgp
from ._common import validate_leaner

_test_settings = {
    # data_generator: model
    _dgp.generate_data_x1b_y1: GradientBoostingRegressor(),
    _dgp.generate_data_x1b_y2: LinearRegression(),
    _dgp.generate_data_x1m_y1: GradientBoostingRegressor(),
    _dgp.generate_data_x1b_y1_w5v0: GradientBoostingRegressor(),
    _dgp.generate_data_x1b_y2_w5v0: LinearRegression(),
    _dgp.generate_data_x1m_y1_w5v0: GradientBoostingRegressor(),
}

_test_settings_x2b = {
    # data_generator: model
    _dgp.generate_data_x2b_y1: GradientBoostingRegressor(),
    _dgp.generate_data_x2b_y2: LinearRegression(),
    _dgp.generate_data_x2b_y1_w5v0: GradientBoostingRegressor(),
    _dgp.generate_data_x2b_y2_w5v0: LinearRegression(),
    # _dgp.generate_data_x2b_y2_w5v0: MultiTaskLasso(),
    _dgp.generate_data_x2mb_y1: GradientBoostingRegressor(),
}


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_sleaner(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, SLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined),
                    check_effect=dg.__name__.find('y2') < 0,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_sleaner_with_treat(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, SLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined, treat=1, control=0),
                    check_effect=dg.__name__.find('y2') < 0,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings_x2b.keys(), [True, False]))
def test_sleaner_x2b(dg, combined):
    model = _test_settings_x2b[dg]
    validate_leaner(dg, SLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined, treat=[1, 1], control=[0, 0]),
                    check_effect=False,  # dg.__name__.find('y2') < 0,
                    )


def test_sleaner_with_treat_control():
    dg = _dgp.generate_data_x2b_y1
    model = GradientBoostingRegressor()
    validate_leaner(dg,
                    TLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=True, treat=[1, 1], control=[0, 0]),
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_tlearner(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, TLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined),
                    check_effect=dg.__name__.find('y2') < 0,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_tlearner_with_treat(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, TLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined, treat=1, control=0),
                    check_effect=dg.__name__.find('y2') < 0,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings_x2b.keys(), [True, False]))
def test_tlearner(dg, combined):
    model = _test_settings_x2b[dg]
    validate_leaner(dg, TLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined, treat=[1, 1], control=[0, 0]),
                    check_effect=dg.__name__.find('y2') < 0,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_xleaner(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, XLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined),
                    check_effect=dg.__name__.find('y2') < 0,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_xleaner_with_treat(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, XLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined, treat=1, control=0),
                    check_effect=dg.__name__.find('y2') < 0,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings_x2b.keys(), [True, False]))
def test_xleaner_x2b(dg, combined):
    model = _test_settings_x2b[dg]
    validate_leaner(dg, XLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined, treat=[1, 1], control=[0, 0]),
                    check_effect=dg.__name__.find('y2') < 0,
                    )
