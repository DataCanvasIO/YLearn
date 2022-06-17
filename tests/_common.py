import numpy as np
import pandas as pd


def validate_leaner(data_generator, leaner,
                    fit_kwargs=None, estimate_kwargs=None,
                    check_fitted=True, check_effect=True,
                    check_effect_nji=False,
                    ):
    # generate data
    data, test_data, outcome, treatment, adjustment, covariate = data_generator()

    # fit
    kwargs = {}
    if adjustment:
        kwargs['adjustment'] = adjustment
    if covariate:
        kwargs['covariate'] = covariate

    if fit_kwargs:
        kwargs.update(fit_kwargs)
    leaner.fit(data, outcome, treatment, **kwargs)
    if check_fitted:
        assert hasattr(leaner, '_is_fitted') and getattr(leaner, '_is_fitted')
    #
    # # estimate
    # kwargs = dict(quantity='ATE')
    # if estimate_kwargs:
    #     kwargs.update(estimate_kwargs)
    # ate = leaner.estimate(**kwargs)
    # assert ate is not None
    # if check_effect:
    #     assert isinstance(ate, (float, np.ndarray))
    #     if isinstance(ate, np.ndarray):
    #         assert ate.dtype.kind == 'f'
    #         assert len(ate.ravel()) == len(outcome)

    # estimate
    kwargs = dict(data=test_data, quantity=None)
    if estimate_kwargs:
        kwargs.update(estimate_kwargs)
    effect = leaner.estimate(**kwargs)
    assert effect is not None
    if check_effect:
        assert isinstance(effect, (np.ndarray, pd.Series))
        assert effect.min() != effect.max()

    if check_effect_nji:
        effect_nji = leaner.effect_nji(test_data)
        assert isinstance(effect_nji, np.ndarray)
        assert effect_nji.shape[0] == len(test_data)
        assert effect_nji.shape[1] == len(outcome)

    return leaner, effect
