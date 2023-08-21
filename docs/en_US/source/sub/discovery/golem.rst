********
Golem
********

The problem of revealing the structures of directed acyclic graphs (DAGs) can be solved 
by formulating a continuous optimization problem over real matrices with soft and sparse 
constraints enforcing the acyclicity condition [Ignavier2020]_. Specifically, for a given 
vector :math:`x \in \mathbb{R}^d` such that there exists a matrix :math:`V` which satisifies
:math:`x = Vx + \eta` for some noise vector :math:`\eta \in \mathbb{R}^d`, the optimization problem can be summarized as follows:

.. math::

    \min_{W \in \mathbb{R}^{d\times d}} \mathcal{S} (W;x) = \mathcal{L}(W;x) + \lambda_1 ||W||_1 + \lambda_2 h(W)

where :math:`\mathcal{L}(W;x)` is the Maximum Likelihood Estimator (MLE):

.. math::

    \mathcal{L}(W;x) = 1/2 \sum_{i=1}^d log\left( \sum_{k=1}^n ||X-WX||^2 \right) - log|det(I-W)|

If one further assumes that the noise variances are equal, it becomes

.. math::

    \mathcal{L}(W;x) = d/2 log\left(\sum_{i=1}^d \sum_{k=1}^n ||X-WX||^2 \right) - log|det(I-W)|

:math:`\lambda_1 ||W||_1` is a penalty term encouraging sparsity, i.e., having fewer edges, 

and :math:`\lambda_2 h(W)` is a penalty term encouraging DAGness on W.

.. math::

    h(W) = tr\left( e^{W \circ W} \right)

where :math:`\circ` is the Hadamard product. This optimization can then be solved with some optimization technique, such as gradient desscent.

The YLearn class for the Golem algorithm is :class:`GolemDiscovery`.

.. topic:: Example

    .. code-block:: python

        import pandas as pd

        from ylearn.exp_dataset.gen import gen
        from ylearn.causal_discovery import GolemDiscovery

        X = gen()
        X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
        Gocd = GolemDiscovery(hidden_layer_dim=[3])
        est = Gocd(X, threshold=0.01, return_dict=True)
        print(est)
    
    >>> OrderedDict([('x0', []), ('x1', ['x0', 'x2', 'x3', 'x4']), ('x2', ['x0']), ('x3', ['x0', 'x2', 'x4']), ('x4', ['x0', 'x2'])])
