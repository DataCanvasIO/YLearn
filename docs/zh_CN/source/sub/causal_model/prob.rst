.. _prob:

*****************************
概率表达式
*****************************

YLearn能够输出和修改类似如下的概率表达式：

.. math::

    P(x, y|z),

用户能够定义一个 :class:`Prob` 的实例，以及改变它的属性.

.. topic:: 举例
    
    如一个概率表达式：

    .. math::

        \sum_{w}P(v|y)[P(w|z)P(x|y)P(u)]

    它由两个部分组成：一个条件概率:math:`P(v|y)` 和一个多概率的乘积 :math:`P(w|z)P(x|y)P(u)` ，然后这两个部分的积在给定的:math:`w` 下求和。

    首先定义第一个条件概率:math:`P(v|y)` ：

    .. code-block:: python
        
        from ylearn.causal_model.prob import Prob
        
        var = {'v'}
        conditional = {'y'} # the conditional set
        marginal = {'w'} # sum on w

    然后， 继续定义第二个多条件概率的乘积 :math:`P(w|z)P(x|y)P(u)`

    .. code-block:: python

        p1 = Prob(variables={'w'}, conditional={'z'})
        p2 = Prob(variables={'x'}, conditional={'y'})
        p3 = Prob(variables={'u'})
        product = {p1, p2, p3}

    最终的结果可以表示为：

    .. code-block:: python

        P = Prob(variables=var, conditional=conditional, marginal=marginal, product=product)
        P.show_latex_expression()
    
    >>> :math:`\sum_w P(v|y)[P(u)][P(w|z)][P(x|y)]`

    :class:`Prob` 的实例还可以输出LaTex代码：
    
    .. code-block:: python

        P.parse()
    
    >>> '\\sum_{w}P(v|y)\\left[P(u)\\right]\\left[P(w|z)\\right]\\left[P(x|y)\\right]'

.. py:class:: ylearn.causal_model.prob.Prob(variables=set(), conditional=set(), divisor=set(), marginal=set(), product=set())

    一个概率分布表达式如下：
    
    .. math::
        
        \sum_{w}P(v|y)[P(w|z)P(x|y)P(u)]. 
    
    用上述例子来阐明参数的含义：

    :param set, default=set() variables: The variables (:math:`v` in the above example) of the probability.
    :param set, default=set() conditional: The conditional set (:math:`y` in the above example) of the probability.
    :param set, default=set() marginal: The sum set (:math:`w` in the above example) for marginalizing the probability.
    :param set, default=set() product: If not set(), then the probability is composed of the first probability
            object :math:`(P(v|y))` and several other probability objects that are all saved
            in the set product, e.g., product = {P1, P2, P3} where P1 for :math:`P(w|z)`,
            P2 for :math:`P(x|y)`, and P3 for :math:`P(u)` in the above example.

    .. py:method:: parse

        返回概率分布的表达式

        :returns: Expression of the encoded probability
        :rtype: str

    .. py:method:: show_latex_expression
        
         显示latex表达式
