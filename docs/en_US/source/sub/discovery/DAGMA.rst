********
DAGMA
********

A novel acyclicity characterization based on the log-determinant (log-det) function (DAGMA), 
which leverages the nilpotency property of directed acyclic graphs (DAGs) [Kevin2022]_. 
Then the DAGMA constrain and first-order optimizer is exploited to reveal the structures of 
directed acyclic graphs (DAGs). Specifically, for a given vector :math:`x \in \mathbb{R}^d` such 
that there exists a matrix :math:`V` which satisifies :math:`x = Vx + \eta` for some noise vector
:math:`\eta \in \mathbb{R}^d`, the optimization problem can be summarized as follows:

.. math::

    \min_{W \in \mathbb{R}^{d\times d}} F(W)+\lambda_1\|W\|_1 \\
    s.t. \quad & h^s_{ldet}(W) = 0,

where :math:`F(W)` is a continuous function measuring :math:`\|x - Wx\|`, :math:`\lambda_1 ||W||_1` is a penalty term encouraging sparsity, i.e., having fewer edges,and 

.. math::

    h^s_{ldet}(W) = -log det(sI- W \circ W) + d log(s)

where :math:`\circ` is the Hadamard product. This optimization can then be solved with some optimization technique, such as gradient desscent.

The YLearn class for the DAGMA algorithm is :class:`DagmaDiscovery`.


.. topic:: Example

    .. code-block:: python

        import pandas as pd

        from ylearn.exp_dataset.gen import gen
        from ylearn.causal_discovery import DagmaDiscovery

        X = gen()
        X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
        Dagmacd = DagmaDiscovery(hidden_layer_dim=[3])
        est = Dagmacd(X, threshold=0.01, return_dict=True)

        print(est)
    
    >>> OrderedDict([('x0', []), ('x1', ['x0', 'x2', 'x3', 'x4']), ('x2', ['x0']), ('x3', ['x0', 'x2', 'x4']), ('x4', ['x0', 'x2'])])
