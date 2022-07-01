***********
Quick Start
***********

In this part, we first show several simple example usages of YLearn. These examples cover the most common functionalities. Then we present a case stuy with :class:`Why` to unveil the hidden
causal relations in data.

Example usages
==============

We present several necessary example usages of YLearn in this section. Please see their specifical documentations to for more details.

1. Representation of causal graph
   
   For a given causal graph :math:`X \leftarrow W \rightarrow Y`, the causal graph is represented by :class:`CausalGraph`

    .. code-block:: python

        causation = {'X': ['W'], 'W':[], 'Y':['W']}
        cg = CausalGraph(causation=causation)

   :py:attr:`cg` will be the causal graph represented in YLearn.

2. Identification of causal effect

   Suppose that we are interested in identifying the causal estimand :math:`P(Y|do(X=x))` in the causal graph `cg`, then we should
   first define an instance of :class:`CausalModel` and call the :py:func:`identify()` method:

    .. code-block:: python

        cm = CausalModel(causal_graph=cg)
        cm.identify(treatment={'X'}, outcome={'Y'}, identify_method=('backdoor', 'simple'))

3. Estimation of causal effect

   The estimation of causal effect with an :class:`EstimatorModel` is composed of 4 steps:
   
    * Given data in the form of :class:`pandas.DataFrame`, find the names of `treatment, outcome, adjustment, covariate`.
    * Call :py:func:`fit()` method of :class:`EstimatorModel` to train the model.
    * Call :py:func:`estimate()` method of :class:`EstimatorModel` to estimate causal effects in test data.


Case Study
==========