.. _prob:

*****************************
Representation of Probability
*****************************

To represent and modifies probabilities such as 

.. math::

    P(x, y|z),

one can define an instance of :class:`Prob` and change its attributes.

.. topic:: Examples
    
    The probabiity

    .. math::

        \sum_{w}P(v|y)[P(w|z)P(x|y)P(u)]

    is composed of two parts: :math:`P(v|y)` and a product :math:`P(w|z)P(x|y)P(u)` and then they are sumed on :math:`w`.
    We first define the first part:

    .. code-block:: python
        
        from ylearn.causal_model.prob import Prob
        
        var = {'v'}
        conditional = {'y'} # the conditional set
        marginal = {'w'} # sum on w

    Now we continue to define the second part, the product,

    .. code-block:: python

        p1 = Prob(variables={'w'}, conditional={'z'})
        p2 = Prob(variables={'x'}, conditional={'y'})
        p3 = Prob(variables={'u'})
        product = {p1, p2, p3}

    The final result is

    .. code-block:: python

        P = Prob(variables=var, conditional=conditional, marginal=marginal, product=product)
        P.show_latex_expression()
    
    >>> :math:`\sum_w P(v|y)[P(u)][P(w|z)][P(x|y)]`

    An instance of :class:`Prob` can also output the latex code
    
    .. code-block:: python

        P.parse()
    
    >>> '\\sum_{w}P(v|y)\\left[P(u)\\right]\\left[P(w|z)\\right]\\left[P(x|y)\\right]'

.. py:class:: ylearn.causal_model.prob.Prob(variables=set(), conditional=set(), divisor=set(), marginal=set(), product=set())

    Probability distribution, e.g., the probability expression
    
    .. math::
        
        \sum_{w}P(v|y)[P(w|z)P(x|y)P(u)]. 
    
    We will clarify below the meanings of our variables with this example.

    :param set, default=set() variables: The variables (:math:`v` in the above example) of the probability.
    :param set, default=set() conditional: The conditional set (:math:`y` in the above example) of the probability.
    :param set, default=set() marginal: The sum set (:math:`w` in the above example) for marginalizing the probability.
    :param set, default=set() product: If not set(), then the probability is composed of the first probability
            object :math:`(P(v|y))` and several other probabiity objects that are all saved
            in the set product, e.g., product = {P1, P2, P3} where P1 for :math:`P(w|z)`,
            P2 for :math:`P(x|y)`, and P3 for :math:`P(u)` in the above example.

    .. py:method:: parse

        Return the expression of the probability distribution.

        :returns: Expression of the encoded probabiity
        :rtype: str

    .. py:method:: show_latex_expression
        
        Show the latex expression.
