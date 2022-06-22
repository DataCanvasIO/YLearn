*********************************
Policy: Selecting the Best Option
*********************************

In tasks such as policy evaluation, besides the causal effects, we may also have interets in other questions such as whether an example should be assigned to a treament and if the answer is yes, which option is
the best to assign among all possible treament values. YLearn implement :class:`PolicyTree` for such purpose. Given a trained estimator model or estimated causal effects, it finds the optimal polices for each
example by building a decision tree model which aims to maximize the causal effect of each example.

The criterion for training the tree is 

.. math::

    S = \sum_i\sum_k g_{ik}e_{ki}

where :math:`g_{ik} = \phi(v_i)_k` with :math:`\phi: \mathbb{R}^D \to \mathbb{R}^K` being a map from :math:`v_i\in \mathbb{R}^D` to a basis vector with only one nonzero element in :math:`\mathbb{R}^K`.