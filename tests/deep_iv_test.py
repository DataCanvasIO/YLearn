import numpy as np
import pandas as pd
import pytest

from . import _dgp
from ._common import if_torch_ready
from .iv_test import validate_it

try:
    import torch
    import torch.nn.functional as F
    from ylearn.estimator_model.deepiv import DeepIV
except ImportError:
    pass

_test_settings = {
    # data_generator: any
    _dgp.generate_data_x1b_y1_w5v0: 'tbd',
    _dgp.generate_data_x2b_y1_w5v0: 'tbd',
    _dgp.generate_data_x1m_y1_w5v0: 'tbd',
}


@if_torch_ready
@pytest.mark.parametrize('dg', _test_settings.keys())
# @pytest.mark.xfail(reason='to be fixed: expected scalar type Double but found Float')
def test_iv_with_params(dg):
    # y_model, x_model = _test_settings[dg]
    dr = DeepIV(num_gaussian=10)
    validate_it(dg, dr,
                float64=True,
                fit_kwargs=dict(
                    sample_n=2,
                    lr=0.5,
                    epoch=1,
                    device='cpu',
                    batch_size=1000
                ), )


@if_torch_ready
# @pytest.mark.xfail(reason='to be fixed')
def test_deep_iv_basis():
    n = 5000
    dtype = torch.float64
    itype = torch.int64

    # Initialize exogenous variables; normal errors, uniformly distributed covariates and instruments
    e = np.random.normal(size=(n, 1))
    w = np.random.uniform(low=0.0, high=10.0, size=(n, 1))
    z = np.random.uniform(low=0.0, high=10.0, size=(n, 1))

    e, w, z = torch.tensor(e, dtype=dtype), torch.tensor(w, dtype=dtype), torch.tensor(z, dtype=dtype)
    weight_w = torch.randn(1, dtype=dtype)
    weight_z = torch.randn(1, dtype=dtype)

    def to_treatment(w, z, e):
        x = torch.sqrt(w) * weight_w + torch.sqrt(z) * weight_z + e
        x = (torch.sign(x) + 1) / 2
        return F.one_hot(x.reshape(-1).to(int))

    # Outcome equation
    weight_x = torch.randn(2, 1, dtype=dtype)
    weight_wx = torch.randn(2, 1, dtype=dtype)

    def to_outcome(w, e, treatment_):
        wx = torch.mm(treatment_.to(dtype), weight_x)
        wx1 = (w * treatment_.to(dtype)).matmul(weight_wx)
        # wx1 = w
        return (wx ** 2) * 10 - wx1 + e / 2

    treatment = to_treatment(w, z, e)
    y = to_outcome(w, e, treatment)

    data_dict = {
        'z': z.squeeze().to(dtype),
        'w': w.squeeze().to(dtype),
        'x': torch.argmax(treatment, dim=1).to(itype),
        'y': y.squeeze().to(dtype)
    }
    data = pd.DataFrame(data_dict)

    # iv = DeepIV(is_discrete_treatment=True)
    # iv.fit(
    #     data=data,
    #     outcome='y',
    #     treatment='x',
    #     instrument='z',
    #     adjustment='w',
    #     device='cpu',
    #     batch_size=2500,
    #     lr=0.5,
    #     epoch=1,
    # )
    iv = DeepIV(num_gaussian=10)
    iv.fit(
        data=data,
        outcome='y',
        treatment='x',
        instrument='z',
        adjustment='w',
        sample_n=2,
        lr=0.5,
        epoch=1,
        device='cpu',
        batch_size=5000
    )

    p = iv.estimate()
    print(p)
