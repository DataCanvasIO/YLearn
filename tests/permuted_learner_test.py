from itertools import product

import pytest
from sklearn import clone

from ylearn.estimator_model import PermutedSLearner, PermutedTLearner, PermutedXLearner, PermutedDoublyRobust
from ._common import validate_leaner
from .doubly_robust_test import _test_settings as dr_test_settings, _test_settings_x2b as dr_test_settings_x2b
from .metalearner_test import _test_settings, _test_settings_x2b


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_sleaner(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, PermutedSLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined),
                    check_effect=dg.__name__.find('y2') < 0,
                    check_effect_nji=combined,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_sleaner_with_treat(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, PermutedSLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined, ),
                    estimate_kwargs=dict(treat=1, control=0),
                    # check_effect=dg.__name__.find('y2') < 0,
                    check_effect=False,
                    check_effect_nji=combined,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings_x2b.keys(), [True, False]))
def test_sleaner_x2b(dg, combined):
    model = _test_settings_x2b[dg]
    validate_leaner(dg, PermutedSLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined, n_jobs=1),
                    estimate_kwargs=dict(treat=[1, 1], control=[0, 0]),
                    # check_effect=dg.__name__.find('y2') < 0,
                    check_effect=False,
                    check_effect_nji=combined,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_tlearner(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, PermutedTLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined),
                    check_effect=dg.__name__.find('y2') < 0,
                    check_effect_nji=combined,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_tlearner_with_treat(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, PermutedTLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined),
                    estimate_kwargs=dict(treat=1, control=0),
                    check_effect=dg.__name__.find('y2') < 0,
                    check_effect_nji=combined,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings_x2b.keys(), [True, False]))
def test_tlearner(dg, combined):
    model = _test_settings_x2b[dg]
    validate_leaner(dg, PermutedTLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined, ),
                    estimate_kwargs=dict(treat=[1, 1], control=[0, 0]),
                    check_effect=dg.__name__.find('y2') < 0,
                    check_effect_nji=combined,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_xleaner(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, PermutedXLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined),
                    check_effect=dg.__name__.find('y2') < 0,
                    check_effect_nji=combined,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings.keys(), [True, False]))
def test_xleaner_with_treat(dg, combined):
    model = _test_settings[dg]
    validate_leaner(dg, PermutedXLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined, ),
                    estimate_kwargs=dict(treat=1, control=0),
                    check_effect=dg.__name__.find('y2') < 0,
                    check_effect_nji=combined,
                    )


@pytest.mark.parametrize('dg,combined', product(_test_settings_x2b.keys(), [True, False]))
def test_xleaner_x2b(dg, combined):
    model = _test_settings_x2b[dg]
    validate_leaner(dg, PermutedXLearner(model=clone(model)),
                    fit_kwargs=dict(combined_treatment=combined, ),
                    estimate_kwargs=dict(treat=[1, 1], control=[0, 0]),
                    check_effect=dg.__name__.find('y2') < 0,
                    check_effect_nji=combined,
                    )


@pytest.mark.parametrize('dg', dr_test_settings.keys())
def test_doubly_robust(dg):
    x_model, y_model, yx_model = dr_test_settings[dg]
    dr = PermutedDoublyRobust(x_model=x_model, y_model=y_model, yx_model=yx_model, cf_fold=1, random_state=2022, )
    validate_leaner(dg, dr,
                    check_effect_nji=True,
                    )


@pytest.mark.parametrize('dg', dr_test_settings.keys())
def test_doubly_robust_with_treat(dg):
    x_model, y_model, yx_model = dr_test_settings[dg]
    dr = PermutedDoublyRobust(x_model=x_model, y_model=y_model, yx_model=yx_model, cf_fold=1, random_state=2022, )
    validate_leaner(dg, dr,
                    estimate_kwargs=dict(treat=1, control=0),
                    check_effect_nji=True,
                    )


@pytest.mark.parametrize('dg', dr_test_settings_x2b.keys())
def test_doubly_robust_x2b(dg):
    x_model, y_model, yx_model = dr_test_settings_x2b[dg]
    dr = PermutedDoublyRobust(x_model=x_model, y_model=y_model, yx_model=yx_model, cf_fold=1, random_state=2022, )
    validate_leaner(dg, dr,
                    estimate_kwargs=dict(treat=[1, 1], control=[0, 0]),
                    check_effect_nji=True,
                    )
