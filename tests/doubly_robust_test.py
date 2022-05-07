import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

from ylearn.estimator_model.doubly_robust import DoublyRobust
from . import _dgp
from ._common import validate_leaner

_test_settings = {
    # data_generator: (x_model,y_model,yx_model)
    _dgp.generate_data_x1b_y1: (RandomForestClassifier(n_estimators=100, max_depth=100),
                                GradientBoostingRegressor(n_estimators=100, max_depth=100),
                                GradientBoostingRegressor(n_estimators=100, max_depth=100),
                                ),
    # _dgp.generate_data_x2b_y1: (RandomForestClassifier(n_estimators=100, max_depth=100),
    #                            GradientBoostingRegressor(n_estimators=100, max_depth=100),
    #                            GradientBoostingRegressor(n_estimators=100, max_depth=100),
    #                            ),
    _dgp.generate_data_x1b_y2: (RandomForestClassifier(n_estimators=100, max_depth=100),
                                LinearRegression(),
                                LinearRegression(),
                                ),
    # _dgp.generate_data_x2b_y2: (RandomForestClassifier(n_estimators=100, max_depth=100),
    #                            LinearRegression(),
    #                            LinearRegression(),
    #                            ),
    _dgp.generate_data_x1m_y1: (RandomForestClassifier(n_estimators=100, max_depth=100),
                                GradientBoostingRegressor(n_estimators=100, max_depth=100),
                                GradientBoostingRegressor(n_estimators=100, max_depth=100),
                                ),
}

_test_settings_to_be_fix = {
    # data_generator: (x_model,y_model,yx_model)
    _dgp.generate_data_x2b_y1: (RandomForestClassifier(n_estimators=100, max_depth=100),
                                GradientBoostingRegressor(n_estimators=100, max_depth=100),
                                GradientBoostingRegressor(n_estimators=100, max_depth=100),
                                ),
    _dgp.generate_data_x2b_y2: (RandomForestClassifier(n_estimators=100, max_depth=100),
                                LinearRegression(),
                                LinearRegression(),
                                ),
}


@pytest.mark.parametrize('dg', _test_settings.keys())
def test_doubly_robust(dg):
    x_model, y_model, yx_model = _test_settings[dg]
    dr = DoublyRobust(x_model=x_model, y_model=y_model, yx_model=yx_model, cf_fold=1, random_state=2022, )
    validate_leaner(dg, dr, check_fitted=False, check_effect=False)


@pytest.mark.parametrize('dg', _test_settings_to_be_fix.keys())
@pytest.mark.xfail(reason='to be fixed')
def test_doubly_robust_to_be_fix(dg):
    x_model, y_model, yx_model = _test_settings_to_be_fix[dg]
    dr = DoublyRobust(x_model=x_model, y_model=y_model, yx_model=yx_model, cf_fold=1, random_state=2022, )
    validate_leaner(dg, dr, check_fitted=False, check_effect=False)
