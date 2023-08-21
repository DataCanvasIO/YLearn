********
DYNOTEARS: Structure Learning from Time-Series Data
********

The problem of estimating contemporaneous (intra-slice) and time-lagged (inter-slice) relationships between variables in a time-series simultaneously can also be solved by formulating
a continuous optimization problem over real matrices with the constraint enforcing the acyclicity condition [Zheng2018]_.
Specifically, for a given vector :math:`x \in \mathbb{R}^d` and its time-lagged versions :math:`y \in \mathbb{R}^{p \times d}` such that there exists a matrix :math:`V` and a matrix :math:`Q` which satisifies :math:`x = Vx + Qy + \eta` for some noise vector :math:`\eta \in \mathbb{R}^d`, the optimization problem can be summarized as follows:

.. math::

    \min_{W, A \in \mathbb{R}^{d\times d}} & F(W, A) \\
    s.t. \quad & h(W) = 0,

where :math:`F(W, A)` is a continuous function measuring :math:`\|x - Wx - Ay\| + \lambda_W\|W\|_1 + \lambda_A\|A\|_1` and 

.. math::

    h(W) = tr\left( e^{W \circ W} \right)

where :math:`\circ` is the Hadamard product. This optimization can then be solved with some optimization technique, such as gradient desscent.

The YLearn class for the DYNOTEARS algorithm is :class:`DyCausalDiscovery`.

.. topic:: Example

    .. code-block:: python

        import pandas as pd

        from ylearn.exp_dataset.gen import dygen
        from ylearn.causal_discovery import DyCausalDiscovery

        X, W_true, P_true = dygen(n=500, d=10, step=5, order=2)
        X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
        dycd = DyCausalDiscovery()
        w_est_dy, p_est_dy  = dycd(X, threshold=0.3, order=2, step=5, return_dict=True)
        acc_dy = count_accuracy(W_true, w_est_dy != 0)
        print(acc_dy)
        for k in range(P_true.shape[2]):
            acc_dy = count_accuracy(P_true[:, :, k], p_est_dy[:, :, k] != 0)
            print(acc_dy)
    
    >>> {'fdr': 0.0, 'tpr': 1.0, 'fpr': 0.0, 'shd': 0, 'nnz': 10}
        {'fdr': 0.0, 'tpr': 1.6, 'fpr': 0.0, 'shd': 0, 'nnz': 10}
        {'fdr': 0.8, 'tpr': 0.6, 'fpr': 0.22857142857142856, 'shd': 11, 'nnz': 10}