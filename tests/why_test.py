import numpy as np
import pandas as pd
import pytest

from ylearn import Why
from . import _dgp
from ._common import if_torch_ready, if_policy_tree_ready, is_policy_tree_ready

try:
    import castle
    from ylearn.causal_discovery import GCastleProxy

    _g = GCastleProxy()
    is_gcastle_ready = True
except ImportError:
    is_gcastle_ready = False


def _validate_it(why, test_data, check_score=True):
    print('-' * 30)
    e = why.causal_effect()
    print('causal effect:', e, sep='\n')

    print('-' * 30)
    e = why.causal_effect(test_data)
    print('causal effect:', e, sep='\n')

    print('-' * 30)
    e = why.individual_causal_effect(test_data)
    print('individual causal effect:', e, sep='\n')

    if check_score:
        score = why.score(test_data, scorer='rloss')
        print("rloss:", score)


def test_basis():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1b_y1()
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    _validate_it(why, test_data)
    effect = why.causal_effect(combine_treatment=True)
    assert effect is not None


def test_x2b():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    _validate_it(why, test_data)


def test_x2mb():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2mb_y1()
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    _validate_it(why, test_data)

    effect = why.causal_effect(combine_treatment=True)
    assert effect is not None


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


@if_policy_tree_ready
def test_policy_interpreter():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1m_y1()
    # data[treatment] = data[treatment].astype('float32')
    # test_data[treatment] = test_data[treatment].astype('float32')
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    pi = why.policy_interpreter(test_data)
    assert pi is not None

    r = pi.decide(test_data)
    assert isinstance(r, np.ndarray) and len(r) == len(test_data)


@if_policy_tree_ready
def test_policy_interpreter_dml():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1m_y1()
    # data[treatment] = data[treatment].astype('float32')
    # test_data[treatment] = test_data[treatment].astype('float32')
    why = Why(estimator='dml')
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    pi = why.policy_interpreter(test_data)
    assert pi is not None

    r = pi.decide(test_data)
    assert isinstance(r, np.ndarray) and len(r) == len(test_data)


@if_policy_tree_ready
def test_policy_interpreter_discrete_x2():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2mb_y1()
    # data[treatment] = data[treatment].astype('float32')
    # test_data[treatment] = test_data[treatment].astype('float32')
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    pi = why.policy_interpreter(test_data)
    assert pi is not None


@if_policy_tree_ready
def test_policy_interpreter_discrete_x2_yb_tlearner():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    m = data[outcome].values.mean()
    data[outcome] = (data[outcome] > m).astype('int')
    test_data[outcome] = (test_data[outcome] > m).astype('int')
    # why = Why()
    why = Why(estimator='ml', estimator_options=dict(learner='t', model='lr'))
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    pi = why.policy_interpreter(test_data)
    assert pi is not None

    pi = why.policy_interpreter(test_data, target_outcome=0)
    assert pi is not None


@if_policy_tree_ready
def test_policy_interpreter_discrete_x2_yb_dml():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    m = data[outcome].values.mean()
    data[outcome] = (data[outcome] > m).astype('int')
    test_data[outcome] = (test_data[outcome] > m).astype('int')
    why = Why(estimator='dml')
    # why = Why(estimator='ml', estimator_options=dict(learner='t', model='lr'))
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    pi = why.policy_interpreter(test_data)
    assert pi is not None

    pi = why.policy_interpreter(test_data, target_outcome=0)
    assert pi is not None


@if_torch_ready
def test_discovery_treatment():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why(identifier='discovery')
    # w.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)
    why.fit(data, outcome[0], treatment=None, adjustment=adjustment, covariate=covariate)

    _validate_it(why, test_data)


# @pytest.mark.xfail(reason='to be fixed')
@if_torch_ready
def test_discovery_taci():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why(identifier='discovery')
    # w.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)
    why.fit(data, outcome[0])

    _validate_it(why, test_data, check_score=False)


@if_torch_ready
def test_discovery_taci_dfs():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why(identifier='discovery', discovery_options=dict(method='dfs'))
    why.fit(data, outcome[0])

    _validate_it(why, test_data, check_score=False)


@pytest.mark.skipif(not is_gcastle_ready, reason='gcastle is not ready')
def test_discovery_taci_with_gcastle():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why(identifier='gcastle', discovery_options=dict(method='dfs'))
    why.fit(data, outcome[0])

    _validate_it(why, test_data, check_score=False)


def test_default_identifier():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why()
    # w.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)
    why.fit(data, outcome[0])

    _validate_it(why, test_data)


def test_uplift():
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    m = data[outcome].values.mean()
    data[outcome] = (data[outcome] > m).astype('int')
    test_data[outcome] = (test_data[outcome] > m).astype('int')
    why = Why()
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    _validate_it(why, test_data)

    s = why.score(test_data, scorer='qini')
    print('qini:', s)

    s = why.score(test_data, scorer='auuc')
    print('auuc:', s)

    um = why.uplift_model(test_data)
    assert um is not None

    r = um.get_gain()
    assert isinstance(r, pd.DataFrame)

    r = um.get_qini()
    assert isinstance(r, pd.DataFrame)

    s = um.auuc_score()
    print('auuc:', s)

    s = um.gain_top_point()
    print('gain_top_point:', s)


def test_customized_estimator():
    from sklearn.ensemble import RandomForestRegressor
    from ylearn.estimator_model import TLearner

    my_estimator = TLearner(model=RandomForestRegressor())
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why(estimator=my_estimator)
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    _validate_it(why, test_data)


def test_fn_cost():
    def _cost(row):
        eff = row['effect'] * 1.1
        return eff

    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x2b_y1()
    why = Why(fn_cost=_cost)
    why.fit(data, outcome[0], treatment=treatment, adjustment=adjustment, covariate=covariate)

    _validate_it(why, test_data)

    if is_policy_tree_ready:
        pi = why.policy_interpreter(test_data)
        assert pi is not None
