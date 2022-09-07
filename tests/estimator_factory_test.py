import pytest

from ylearn.estimator_model import ESTIMATOR_FACTORIES
from . import _dgp
from ._common import if_policy_tree_ready, is_policy_tree_ready, is_torch_installed


@pytest.mark.parametrize('key', ESTIMATOR_FACTORIES.keys())
def test_list_factory(key):
    if key.lower().find('tree') >= 0 and not is_policy_tree_ready:
        return

    if (key.lower().find('div') >= 0 or key.lower().find('deep_iv') >= 0) and not is_torch_installed:
        return

    factory_cls = ESTIMATOR_FACTORIES[key]
    factory = factory_cls()
    assert factory is not None


@pytest.mark.parametrize('key', ['slearner', 'tlearner', 'xlearner', 'dml', 'dr'])
def test_Xb_Yc(key):
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1b_y1()

    factory = ESTIMATOR_FACTORIES[key]()
    est = factory(data, outcome[0], treatment, 'regression', 'binary',
                  adjustment=adjustment, covariate=covariate, random_state=123)
    assert est is not None

    est.fit(data, outcome, treatment, adjustment=adjustment, covariate=covariate, n_jobs=1)
    effect = est.estimate(test_data)
    assert effect.shape[0] == len(test_data)


@if_policy_tree_ready
@pytest.mark.parametrize('key', ['tree', ])
def test_Xb_Yc_tree(key):
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1b_y1()

    factory = ESTIMATOR_FACTORIES[key]()
    est = factory(data, outcome[0], treatment, 'regression', 'binary',
                  adjustment=adjustment, covariate=covariate, random_state=123)
    assert est is not None

    est.fit(data, outcome, treatment, adjustment=adjustment, covariate=covariate, n_jobs=1)
    effect = est.estimate(test_data)
    assert effect.shape[0] == len(test_data)


@pytest.mark.parametrize('key', ['slearner', 'tlearner', 'xlearner'])
def test_Xb_Yb(key):
    data, test_data, outcome, treatment, adjustment, covariate = _dgp.generate_data_x1b_y1()
    m = data[outcome].values.mean()
    data[outcome] = (data[outcome] > m).astype('int')
    test_data[outcome] = (test_data[outcome] > m).astype('int')

    factory = ESTIMATOR_FACTORIES[key]()
    est = factory(data, outcome[0], treatment, 'binary', 'binary',
                  adjustment=adjustment, covariate=covariate, random_state=123)
    assert est is not None

    est.fit(data, outcome, treatment, adjustment=adjustment, covariate=covariate, n_jobs=1)
    effect = est.estimate(test_data)
    assert effect.shape[0] == len(test_data)
