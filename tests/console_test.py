import numpy as np
import pytest

from ylearn.causal_console import CausalConsole
from . import _dgp


def _validate_it(cc, test_data):
    print('-' * 30)
    e = cc.causal_effect()
    print('causal effect:', e, sep='\n')

    print('-' * 30)
    e = cc.cohort_causal_effect(test_data)
    print('cohort causal effect:', e, sep='\n')

    print('-' * 30)
    e = cc.local_causal_effect(test_data)
    print('local causal effect:', e, sep='\n')

    if cc.scorers_ is not None:
        score = cc.score()
        print("score:", score)


def test_basis():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    cc = CausalConsole()
    cc.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    _validate_it(cc, test_data)


def test_identify_treatment():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    cc = CausalConsole()
    # cc.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)
    cc.fit(data, outcome[0], treatment=None, adjustment=adjustment, covariate=covariate)

    _validate_it(cc, test_data)


def test_whatif_discrete():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    cc = CausalConsole()
    cc.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    new_value = np.ones_like(test_data[treatment[0]])
    new_y = cc.whatif(test_data, new_value, treatment[0])
    assert new_y is not None
    print(new_y.shape, new_y)


def test_whatif_continuous():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1m_y1()
    data[treatment] = data[treatment].astype('float32')
    test_data[treatment] = test_data[treatment].astype('float32')
    cc = CausalConsole()
    cc.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    new_value = np.ones_like(test_data[treatment[0]])
    new_y = cc.whatif(test_data, new_value, treatment[0])
    assert new_y is not None
    print(new_y.shape, new_y)


def test_policy_tree():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1m_y1()
    # data[treatment] = data[treatment].astype('float32')
    # test_data[treatment] = test_data[treatment].astype('float32')
    cc = CausalConsole()
    cc.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    ptree = cc.policy_tree(test_data)
    assert ptree is not None


def test_policy_tree_dml():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1m_y1()
    # data[treatment] = data[treatment].astype('float32')
    # test_data[treatment] = test_data[treatment].astype('float32')
    cc = CausalConsole(estimator='dml')
    # cc.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)
    cc.fit(data, treatment[0], treatment=outcome, adjustment=adjustment, covariate=covariate)

    ptree = cc.policy_tree(test_data)
    assert ptree is not None


def test_policy_interpreter():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1m_y1()
    # data[treatment] = data[treatment].astype('float32')
    # test_data[treatment] = test_data[treatment].astype('float32')
    cc = CausalConsole()
    cc.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    pi = cc.policy_interpreter(test_data)
    assert pi is not None


@pytest.mark.xfail(reason='to be fixed')
def test_discovery_treatment():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    cc = CausalConsole(identify='discovery')
    # cc.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)
    cc.fit(data, outcome[0], treatment=None, adjustment=adjustment, covariate=covariate)

    _validate_it(cc, test_data)


@pytest.mark.xfail(reason='to be fixed')
def test_discovery_taci():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    cc = CausalConsole(identify='discovery')
    # cc.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)
    cc.fit(data, outcome[0])

    _validate_it(cc, test_data)


@pytest.mark.xfail(reason='to be fixed')
def test_score():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    cc = CausalConsole(scorer='auto')
    cc.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    _validate_it(cc, test_data)
