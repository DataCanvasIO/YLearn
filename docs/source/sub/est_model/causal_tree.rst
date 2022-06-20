***********
Causal Tree
***********

Causal Tree is a data-driven approach to partition the data into subpopulations which differ in the magnitude
of their causal effects [Athey2015]_. This method is applicable when the unconfoundness is satisified given the adjustment
set (covariate) :math:`V`. The interested causal effects is the CATE:

.. math::

    \tau(v) := \mathbb{}[Y_i(do(X=x_t)) - Y_i(do(X=x_0)) | V_i = v]

Due to the fact that the counterfactuals can never be observed, [Athey2015]_ developed an honest approach where the loss
function (criterion for building a tree) is designed as

.. math::

    e (S_{tr}, \Pi) := \frac{1}{N_{tr}} \sum_{i \in S_{tr}} \hat{\tau}^2 (V_i; S_{tr}, \Pi) - \frac{2}{N_{tr}} \cdot \sum_{\ell \in \Pi} \left( \frac{\Sigma^2_{S_{tr}^{treat}}(\ell)}{p} + \frac{\Sigma^2_{S_{tr}^{control}}(\ell)}{1 - p}\right)

where :math:`N_{tr}` is the nubmer of samples in the training set :math:`S_{tr}`, :math:`p` is the ratio of the nubmer of samples in the treat group to that of the control group in the trainig set, and

.. math::

    \hat{\tau}(v) = \frac{1}{\#(\{i\in S_{treat}: V_i \in \ell(v; \Pi)\})} \sum_{ \{i\in S_{treat}: V_i \in \ell(v; \Pi)\}} Y_i \\
    - \frac{1}{\#(\{i\in S_{control}: V_i \in \ell(v; \Pi)\})} \sum_{ \{i\in S_{control}: V_i \in \ell(v; \Pi)\}} Y_i.


Class Structures
================

.. autoclass:: ylearn.estimator_model.causal_tree.CausalTree


.. topic:: Example

    pass