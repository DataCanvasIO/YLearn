import numpy as np
import pandas as pd

from ylearn.causal_discovery import CausalDiscovery
from ylearn.exp_dataset.gen import gen
from ._common import if_torch_ready


@if_torch_ready
def test_ndarray():
    X1 = gen()
    # X1 = pd.DataFrame(X1, columns=[f'x{i}' for i in range(X1.shape[1])])
    cd = CausalDiscovery(hidden_layer_dim=[3], device='cpu')
    est = cd(X1, threshold=0.01)
    print(est)
    assert isinstance(est, np.ndarray)
    assert est.shape[0] == est.shape[1]

    m = cd.matrix2dict(est)
    assert isinstance(m, dict)

    m = cd.matrix2df(est)
    assert isinstance(m, pd.DataFrame)


@if_torch_ready
def test_dataframe():
    X1 = gen()
    X1 = pd.DataFrame(X1, columns=[f'x{i}' for i in range(X1.shape[1])])
    cd = CausalDiscovery(hidden_layer_dim=[3], device='cpu')
    est = cd(X1, threshold=0.01)
    print(est)
    assert isinstance(est, pd.DataFrame)
    assert est.columns.to_list() == X1.columns.to_list()
    assert est.shape[0] == est.shape[1]


@if_torch_ready
def test_dataframe_disable_scale():
    X1 = gen()
    X1 = pd.DataFrame(X1, columns=[f'x{i}' for i in range(X1.shape[1])])
    cd = CausalDiscovery(hidden_layer_dim=[3], scale=False, device='cpu')
    est = cd(X1, threshold=0.01)
    print(est)
    assert isinstance(est, pd.DataFrame)
    assert est.columns.to_list() == X1.columns.to_list()
    assert est.shape[0] == est.shape[1]


@if_torch_ready
def test_dataframe_with_maxabs_sacler():
    from sklearn.preprocessing import MaxAbsScaler
    X1 = gen()
    X1 = pd.DataFrame(X1, columns=[f'x{i}' for i in range(X1.shape[1])])
    cd = CausalDiscovery(hidden_layer_dim=[3], scale=MaxAbsScaler(), device='cpu')
    est = cd(X1, threshold=0.01)
    print(est)
    assert isinstance(est, pd.DataFrame)
    assert est.columns.to_list() == X1.columns.to_list()
    assert est.shape[0] == est.shape[1]


@if_torch_ready
def test_return_dict():
    X1 = gen()
    cd = CausalDiscovery(hidden_layer_dim=[3], device='cpu')
    est = cd(X1, threshold=0.01, return_dict=True)
    print(est)
    assert isinstance(est, dict)
