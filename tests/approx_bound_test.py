import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from ylearn.estimator_model.approximation_bound import ApproxBound
from . import _dgp
from ._common import validate_leaner

_test_settings = {
    # data_generator: (y_model,x_model,x_prob)
    _dgp.generate_data_x1b_y1_w0v5: (LinearRegression(), RandomForestClassifier(), None),
    # _dgp.generate_data_x2b_y1_w0v5: (LinearRegression(), RandomForestClassifier(), None),
    _dgp.generate_data_x1m_y1_w0v5: (LinearRegression(), RandomForestClassifier(), None),
}


@pytest.mark.parametrize('dg', _test_settings.keys())
def test_approx_bound(dg):
    y_model, x_model, x_proba = _test_settings[dg]
    dr = ApproxBound(x_model=x_model, y_model=y_model, x_prob=x_proba, random_state=2022)
    est, effect = validate_leaner(dg, dr, check_effect=False)

    # ApproxBound estimate effect: tuple(lower_bound, upper_bound)
    assert isinstance(effect, tuple) and len(effect) == 2
    assert isinstance(effect[0], (pd.Series, np.ndarray)) and effect[0].min() < effect[0].max()
    assert isinstance(effect[1], (pd.Series, np.ndarray)) and effect[1].min() < effect[1].max()
