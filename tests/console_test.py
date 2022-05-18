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
