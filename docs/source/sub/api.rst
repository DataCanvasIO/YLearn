.. _api:

****************************
API: Interacting with YLearn 
****************************

.. list-table:: All-in-one API

    * - Class Name
      - Description
    * - :py:class:`Why`
      - An API which encapsulates almost everything in YLearn, such as *identifying causal effects* and *scoring a trained estimator model*. It provides to users a simple and efficient way to use YLearn.

.. list-table:: Causal Structures Discovery

    * - Class Name
      - Description
    * - :py:class:`CausalDiscovery`
      - Find causal structures in observational data.

.. list-table:: Causal Model

    * - Class Name
      - Description
    * - :py:class:`CausalGraph`
      - Express the causal structures and support other operations related to causal graph, e.g., add and delete edges to the graph.
    * - :py:class:`CausalModel`
      - Encode causations represented by the :py:class:`CausalGraph`. Mainly support causal effect identification, e.g., backdoor adjustment.
    * - :py:class:`Prob`
      - Represent the probability distribution.

.. list-table:: Estimator Models

    * - Class Name
      - Description
    * - :py:class:`ApproxBound`
      - A model used for estimating the upper and lower bounds of the causal effects. This model does not need the unconfoundedness condition.
    * - :py:class:`CausalTree`
      - A class for estimating causal effect with decision tree. The unconfoundedness condition is required.
    * - :py:class:`DeepIV`
      - Instrumental variables with deep neural networks. Must provide the names of instrumental variables.
    * - :py:class:`NP2SLS`
      - Nonparametric instrumental variables. Must provide the names of instrumental variables.
    * - :py:class:`DML4CATE`
      - Double machine learning model for the estimation of CATE. The unconfoundedness condition is required.
    * - :py:class:`DoublyRobust` and :py:class:`PermutedDoublyRobust`
      - Doubly robust method for the estimation of CATE. The permuted version considers all possible treatment-control pairs. The unconfoundedness condition is required and the treatment must be discrete.
    * - :py:class:`SLearner` and :py:class:`PermutedSLearner`
      - SLearner. The permuted version considers all possible treatment-control pairs. The unconfoundedness condition is required and the treatment must be discrete.
    * - :py:class:`TLearner` and :py:class:`PermutedTLearner`
      - TLearner with multiple machine learning models. The permuted version considers all possible treatment-control pairs. The unconfoundedness condition is required and the treatment must be discrete.
    * - :py:class:`XLearner` and :py:class:`PermutedXLearner`
      - XLearner with multiple machine learning models. The permuted version considers all possible treatment-control pairs. The unconfoundedness condition is required and the treatment must be discrete.
    * - :py:class:`RLoss`
      - Effect score for measuring the performances of estimator models. The unconfoundedness condition is required.

.. list-table:: Policy

    * - Class Name
      - Description
    * - :py:class:`PolicyTree`
      - A class for finding the optimal policy for maximizing the causal effect with the tree model.

.. list-table:: Interpreter

    * - Class Name
      - Description
    * - :py:class:`CEInterpreter`
      - An object used to interpret the estimated CATE using the decision tree model.
    * - :py:class:`PolicyInterpreter`
      - An object used to interpret the policy given by some :py:class:`PolicyModel`.

