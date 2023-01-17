import numpy as np
import pandas as pd

from ylearn.estimator_model.utils import nd_kron
from ylearn.utils import tit, tit_report, tic_toc


def _nd_kron_original(x, y):
    dim = x.shape[0]
    assert dim == y.shape[0]
    kron_prod = np.kron(x[0], y[0]).reshape(1, -1)

    if dim > 1:
        for i, vec in enumerate(x[1:], 1):
            kron_prod = np.concatenate(
                (kron_prod, np.kron(vec, y[i]).reshape(1, -1)), axis=0
            )

    return kron_prod


def test_kron():
    n = 10

    # test (n,1)x(n,3)
    x = np.random.random((n, 1))
    y = np.random.random((n, 3))
    k1 = nd_kron(x, y)
    k2 = _nd_kron_original(x, y)
    assert k1.shape == k2.shape
    assert (k1 == k2).all()

    # test (n,2)x(n,3)
    x = np.random.random((n, 2))
    y = np.random.random((n, 3))
    k1 = nd_kron(x, y)
    k2 = _nd_kron_original(x, y)
    assert k1.shape == k2.shape
    assert (k1 == k2).all()


def foo(*args, **kwargs):
    for i, a in enumerate(args):
        print('arg', i, ':', a)
    for k, v in kwargs.items():
        print('kwarg', k, ':', v)


@tic_toc()
def bar(*args, **kwargs):
    foo(*args, **kwargs)


def test_tit():
    fn = tit(foo)
    fn('a', 1, x='xxx')
    fn('b', 2, x='xxx')

    bar('b', 2, x='xxx')

    report = tit_report()
    assert isinstance(report, pd.DataFrame)

    fn_name = f'{foo.__module__}.{foo.__qualname__}'
    assert fn_name in report.index.tolist()
    assert report.loc[fn_name]['count'] == 2

    assert hasattr(bar, 'tic_toc_')
    bar_fn = bar.tic_toc_
    fn_name = f'{bar_fn.__module__}.{bar_fn.__qualname__}'
    assert fn_name in report.index.tolist()
    assert report.loc[fn_name]['count'] == 1
