import numpy as np
import pandas as pd


def validate_leaner(data_generator, leaner,
                    fit_kwargs=None, estimate_kwargs=None,
                    check_fitted=True, check_effect=True):
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
    pred = leaner.estimate(**kwargs)
    assert pred is not None
    if check_effect:
        assert isinstance(pred, (np.ndarray, pd.Series))
        assert pred.min() != pred.max()

    return leaner, pred
