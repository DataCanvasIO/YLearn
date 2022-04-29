import numpy as np

from ylearn.estimator_model.utils import nd_kron


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
