import numpy as np
import pytest

from ylearn import Why
from . import _dgp


def _validate_it(why, test_data, check_score=True):
    print('-' * 30)
    e = why.causal_effect()
    print('causal effect:', e, sep='\n')

    print('-' * 30)
    e = why.causal_effect(test_data)
    print('cohort causal effect:', e, sep='\n')

    print('-' * 30)
    e = why.individual_causal_effect(test_data)
    print('local causal effect:', e, sep='\n')

    if check_score:
        score = why.score(test_data)
        print("score:", score)


def test_basis():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    _validate_it(why, test_data)


@pytest.mark.xfail(reason='to be fixed')
def test_iv():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, instrument=adjustment, covariate=covariate)

    _validate_it(why, test_data, check_score=False)


@pytest.mark.xfail(reason='to be fixed')
def test_iv_w():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, instrument=adjustment[:2],
            adjustment=adjustment[2:], covariate=covariate)

    _validate_it(why, test_data, check_score=False)


def test_identify_treatment():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why()
    # why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)
    why.fit(data, outcome[0], treatment=None, adjustment=adjustment, covariate=covariate)

    _validate_it(why, test_data)


def test_whatif_discrete():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    new_value = np.ones_like(test_data[treatment[0]])
    new_y = why.whatif(test_data, new_value, treatment[0])
    assert new_y is not None
    print(new_y.shape, new_y)


def test_whatif_continuous():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1m_y1()
    data[treatment] = data[treatment].astype('float32')
    test_data[treatment] = test_data[treatment].astype('float32')
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    new_value = np.ones_like(test_data[treatment[0]])
    new_y = why.whatif(test_data, new_value, treatment[0])
    assert new_y is not None
    print(new_y.shape, new_y)


def test_policy_tree():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1m_y1()
    # data[treatment] = data[treatment].astype('float32')
    # test_data[treatment] = test_data[treatment].astype('float32')
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    ptree = why.policy_tree(test_data)
    assert ptree is not None


def test_policy_tree_dml():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1m_y1()
    # data[treatment] = data[treatment].astype('float32')
    # test_data[treatment] = test_data[treatment].astype('float32')
    why = Why(estimator='dml')
    # why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)
    why.fit(data, treatment[0], treatment=outcome, adjustment=adjustment, covariate=covariate)

    ptree = why.policy_tree(test_data)
    assert ptree is not None


def test_policy_interpreter():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1m_y1()
    # data[treatment] = data[treatment].astype('float32')
    # test_data[treatment] = test_data[treatment].astype('float32')
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    pi = why.policy_interpreter(test_data)
    assert pi is not None


def test_discovery_treatment():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why(identifier='discovery')
    # w.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)
    why.fit(data, outcome[0], treatment=None, adjustment=adjustment, covariate=covariate)

    _validate_it(why, test_data)


# @pytest.mark.xfail(reason='to be fixed')
def test_discovery_taci():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why(identifier='discovery')
    # w.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)
    why.fit(data, outcome[0])

    _validate_it(why, test_data, check_score=False)
