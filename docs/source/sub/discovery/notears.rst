********
No-Tears
********

The problem of revealing the structures of directed acyclic graphs (DAGs) can be solved by formulating
a continuous optimization problem over real matrices with the constraint enforcing the acyclicity condition [Zheng2018]_.
Specifically, for a given vector :math:`x \in \mathbb{R}^d` such that there exists a matrix :math:`V` which satisifies :math:`x = Vx + \eta` for some noise vector :math:`\eta \in \mathbb{R}^d`, the optimization problem can be summarized as follows:

.. math::

    \min_{W \in \mathbb{R}^{d\times d}} & F(W) \\
    s.t. \quad & h(W) = 0,

where :math:`F(W)` is a continuous function measuring :math:`\|x - Wx\|` and 

.. math::

    h(W) = tr\left( e^{W \circ W} \right)

where :math:`\circ` is the Hadamard product. This optimization can then be solved with some optimization technique, such as gradient desscent.

The YLearn class for the NO-TEARS algorithm is :class:`CausalDiscovery`.

.. topic:: Example

    .. code-block:: python

        import pandas as pd

        from ylearn.exp_dataset.gen import gen
        from ylearn.causal_discovery import CausalDiscovery

        X = gen()
        X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
        cd = CausalDiscovery(hidden_layer_dim=[3])
        est = cd(X, threshold=0.01, return_dict=True)

        print(est)
    
    >>> OrderedDict([('x0', []), ('x1', ['x0', 'x2', 'x3', 'x4']), ('x2', ['x0']), ('x3', ['x0', 'x2', 'x4']), ('x4', ['x0', 'x2'])])