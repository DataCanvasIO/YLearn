***********
Quick Start
***********

In this part, we first show several simple example usages of YLearn. These examples cover the most common functionalities. Then we present a case study with :class:`Why` to unveil the hidden
causal relations in data.

Example usages
==============

We present several necessary example usages of YLearn in this section, which covers defining a causal graph, identifying the causal effect, and training an estimator model, etc.  Please see their specific documentations to for more details.

1. Representation of causal graph
   
   Given a set of variables, the representation of its causal graph in YLearn requires a python :py:class:`dict` to denote the causal relations of variables, in which the *keys* of the :py:class:`dict` are children of all elements in the
   corresponding values which usually should be a list of names of variables. For an instance, in the simplest case, for a given causal graph :math:`X \leftarrow W \rightarrow Y`, we first define a python :py:class:`dict` for the causal relations,
   which will then be passed to :py:class:`CausalGraph` as a parameter:

    .. code-block:: python

        causation = {'X': ['W'], 'W':[], 'Y':['W']}
        cg = CausalGraph(causation=causation)

   :py:attr:`cg` will be the causal graph encoding the causal relation :math:`X \leftarrow W \rightarrow Y` in YLearn. If there exist unobserved confounders in the causal graph, then, aside from the observed variables, we should also define a python 
   :py:class:`list` containing these causal relations. See :ref:`causal_graph` for more details.

2. Identification of causal effect

   It is crucial to identify the causal effect when we want to estimate it from data. The first step for identifying the causal effect is identifying the causal estimand. This can be easily done in YLearn. For an instance, suppose that we are interested in identifying the causal estimand :math:`P(Y|do(X=x))` in the causal graph `cg`, then we should
   first define an instance of :class:`CausalModel` and call the :py:func:`identify()` method:

    .. code-block:: python

        cm = CausalModel(causal_graph=cg)
        cm.identify(treatment={'X'}, outcome={'Y'}, identify_method=('backdoor', 'simple'))

   where we use the *backdoor-adjustment* method here. YLearn also support front-door adjustment, finding instrumental variables, and, most importantly, the general identification method developed in [Pearl]_ which is able to identify any causal effect if it is identifiable.

3. Estimation of causal effect

   The estimation of causal effects in YLearn is also fairly easy. It follows the common approach of deploying a machine learning model since YLearn focuses on the intersection of machine learning and causal inference in this part. Given a dataset, one can apply any 
   :class:`EstimatorModel` in YLearn with a procedure composed of 3 steps:
   
    * Given data in the form of :class:`pandas.DataFrame`, find the names of `treatment, outcome, adjustment, covariate`.
    * Call :py:func:`fit()` method of :class:`EstimatorModel` to train the model.
    * Call :py:func:`estimate()` method of :class:`EstimatorModel` to estimate causal effects in test data.

   See :ref:`estimator_model` for more details.

4. Using the all-in-one API: Why

    For the purpose of applying YLearn in a unified and eaiser manner, YLearn provides the API :py:class:`Why`. :py:class:`Why` is an API which encapsulates almost everything in YLearn, such as identifying causal effects and scoring a trained estimator model. 
    To use :py:class:`Why`, one should first create an instance of :py:class:`Why` which needs to be trained by calling its method :py:func:`fit()`, after which other utilities, such as :py:func:`causal_effect()`, :py:func:`score()`, and :py:func:`whatif()`, 
    could be used. This procedure is illustrated in the following code example:

    .. code-block:: python

        from sklearn.datasets import fetch_california_housing

        from ylearn import Why

        housing = fetch_california_housing(as_frame=True)
        data = housing.frame
        outcome = housing.target_names[0]
        data[outcome] = housing.target

        why = Why()
        why.fit(data, outcome, treatment=['AveBedrms', 'AveRooms'])

        print(why.causal_effect())

