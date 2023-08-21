********
DYGolem
********

The problem of estimating contemporaneous (intra-slice) and time-lagged (inter-slice) relationships between variables in a time-series simultaneously can also be solved by formulating
a continuous optimization problem over real matrices with soft and sparse constraints enforcing the acyclicity condition [Ignavier2020]_.
Specifically, for a given vector :math:`x \in \mathbb{R}^d` and its time-lagged versions :math:`y \in \mathbb{R}^{p \times d}` such that there exists a matrix :math:`V` and a matrix :math:`Q` which satisifies :math:`x = Vx + Qy + \eta` for some noise vector :math:`\eta \in \mathbb{R}^d`, the optimization problem can be summarized as follows:

.. math::

    \min_{W, A} \mathcal{S} (W,A;x) = \mathcal{L}(W,A;x) + \lambda_1 ||W||_1 + \lambda_2 h(W) + \lambda_3 ||A||_1

where :math:`\mathcal{L}(W,A;x)` is the Maximum Likelihood Estimator (MLE):

.. math::

    \mathcal{L}(W,A;x) = 1/2 \sum_{i=1}^d log\left( \sum_{k=1}^n \|X-WX-YA\|^2 \right) - log|det(I-W)|

If one further assumes that the noise variances are equal, it becomes

.. math::

    \mathcal{L}(W,A;x) = d/2 log\left(\sum_{i=1}^d \sum_{k=1}^n \|X-WX-YA\|^2 \right) - log|det(I-W)|

where :math:`\lambda_1 \|W\|_1` and :math:`\lambda_3 \|A\|_1` is penalty term encouraging sparsity, i.e., having fewer edges, 

and :math:`\lambda_2 h(W)` is a penalty term encouraging DAGness on W.

.. math::
    h(W) = tr\left( e^{W \circ W} \right)

where :math:`\circ` is the Hadamard product. This optimization can then be solved with some optimization technique, such as gradient desscent.


The YLearn class for the DYGolem algorithm is :class:`DygolemCausalDiscovery`.

.. topic:: Example

    .. code-block:: python

        import pandas as pd

        from ylearn.exp_dataset.gen import dygen
        from ylearn.causal_discovery import DygolemCausalDiscovery

        X, W_true, P_true = dygen(n=500, d=10, step=5, order=2)
        X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])

        dygo = DygolemCausalDiscovery()
        w_est_dy, p_est_dy = dygo(X, threshold=0.3, order=2, step=5, return_dict=True)
        acc_dy = count_accuracy(W_true, w_est_dy != 0)
        print(acc_dy)
        for k in range(P_true.shape[2]):
            acc_dy = count_accuracy(P_true[:, :, k], p_est_dy[:, :, k] != 0)
            print(acc_dy)
    
    >>> {'fdr': 0.55, 'tpr': 0.9, 'fpr': 0.3142857142857143, 'shd': 12, 'nnz': 20}
        {'fdr': 0.0, 'tpr': 0.6, 'fpr': 0.0, 'shd': 7, 'nnz': 0}
        {'fdr': 0.0, 'tpr': 0.4, 'fpr': 0.0, 'shd': 8, 'nnz': 0}