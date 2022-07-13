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
# @pytest.mark.xfail(reason='to be fixed: effect is tuple')
def test_iv_with_params(dg):
    # y_model, x_model = _test_settings[dg]
    dr = DeepIV(num_gaussian=10)
    validate_it(dg, dr,
                float32=True,
                fit_kwargs=dict(
                    sample_n=2,
                    lr=0.5,
                    epoch=1,
                    device='cpu',
                    batch_size=1000
                ), )


@if_torch_ready
@pytest.mark.xfail(reason='to be fixed')
def test_deep_iv_basis():
    n = 5000

    # Initialize exogenous variables; normal errors, uniformly distributed covariates and instruments
    e = np.random.normal(size=(n, 1))
    w = np.random.uniform(low=0.0, high=10.0, size=(n, 1))
    z = np.random.uniform(low=0.0, high=10.0, size=(n, 1))

    e, w, z = torch.tensor(e), torch.tensor(w), torch.tensor(z)
    weight_w = torch.randn(1)
    weight_z = torch.randn(1)

    def treatment(w, z, e):
        x = torch.sqrt(w) * weight_w + torch.sqrt(z) * weight_z + e
        x = (torch.sign(x) + 1) / 2
        return F.one_hot(x.reshape(-1).to(int))

    # Outcome equation
    weight_x = torch.randn(2, 1)
    weight_wx = torch.randn(2, 1)

    def outcome(w, e, treatment):
        wx = torch.mm(treatment.to(torch.float32), weight_x)
        wx1 = (w * treatment.to(torch.float32)).to(torch.float32).matmul(weight_wx.to(torch.float32))
        # wx1 = w
        return (wx ** 2) * 10 - wx1 + e / 2

    treatment = treatment(w, z, e)
    y = outcome(w, e, treatment)

    data_dict = {
        'z': z.squeeze().to(torch.float32),
        'w': w.squeeze().to(torch.float32),
        'x': torch.argmax(treatment, dim=1),
        'y': y.squeeze().to(torch.float32)
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
