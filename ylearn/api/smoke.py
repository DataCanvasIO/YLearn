from ylearn.api import Why
from ylearn.exp_dataset.exp_data import single_binary_treatment


def smoke(estimator='auto'):
    print('-' * 20, 'smoke with estimator', estimator, '-' * 20)

    train, val, _ = single_binary_treatment()
    te = train.pop('TE')
    te = val.pop('TE')
    adjustment = [c for c in train.columns.tolist() if c.startswith('w')]
    covariate = [c for c in train.columns.tolist() if c.startswith('c')]

    if estimator == 'grf':
        covariate.extend(adjustment)
        adjustment = None

    why = Why(estimator=estimator)
    why.fit(train, outcome='outcome', treatment='treatment', adjustment=adjustment, covariate=covariate)

    cate = why.causal_effect(val)
    print('CATE:\n', cate)

    auuc = why.score(val, scorer='auuc')
    print('AUUC', auuc)


if __name__ == '__main__':
    from ylearn.utils import logging

    logging.set_level('info')
    for est in ['slearner', 'tlearner', 'xlearner', 'dr', 'dml', 'tree', 'grf']:
        smoke(est)

    print('\n<done>')
