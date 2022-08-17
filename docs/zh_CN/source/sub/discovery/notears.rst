********
No-Tears 算法
********

No-Tears 揭示了有向无环图 (DAG) 结构的问题可以通过在具有强制执行无环条件的约束的实矩阵上制定连续优化问题来解决 [Zheng2018]_.
具体地说，对于给定的向量 :math:`x \in \mathbb{R}^d` ，存在一个矩阵 :math:`V` ，满足某个噪声向量 :math:`x = Vx + \eta` ，使 :math:`\eta \in \mathbb{R}^d`的优化问题可以描述如下:

.. math::

    \min_{W \in \mathbb{R}^{d\times d}} & F(W) \\
    s.t. \quad & h(W) = 0,

其中， :math:`F(W)` 是一个衡量 :math:`\|x - Wx\|`  的连续方程；

此外，

.. math::

    h(W) = tr\left( e^{W \circ W} \right)

其中 :math:`\circ` 是阿达玛乘积（Hadamard product）。整个公式可以通过一些优化技术来解决，比如梯度下降。

NO-TEARS 算法的类是 :class:`CausalDiscovery`.

.. topic:: 举例

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
