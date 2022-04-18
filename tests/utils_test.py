import numpy as np
from estimator_model.utils import nd_kron, nd_kron_original


def test_kron():
    n = 10

    # test (n,1)x(n,3)
    x = np.random.random((n, 1))
    y = np.random.random((n, 3))
    k1 = nd_kron(x, y)
    k2 = nd_kron_original(x, y)
    assert k1.shape == k2.shape
    assert (k1 == k2).all()

    # test (n,2)x(n,3)
    x = np.random.random((n, 2))
    y = np.random.random((n, 3))
    k1 = nd_kron(x, y)
    k2 = nd_kron_original(x, y)
    assert k1.shape == k2.shape
    assert (k1 == k2).all()
