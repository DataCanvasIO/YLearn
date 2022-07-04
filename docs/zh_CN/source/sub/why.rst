***************************************
Why: An All-in-One Causal Learning API
***************************************

Want to use YLearn in a much eaiser way? Try the all-in-one API `Why`!

`Why` is an API which encapsulates almost everything in YLearn, such as *identifying causal effects* and *scoring a trained estimator model*. It provides to users a simple
and efficient way to use our package: one can directly pass the only thing you have, the data, into
`Why` and call various methods of it rather than learning multiple concepts such as adjustment set before being able to find interesting information hidden in your data. `Why`
is designed to enable the full-pipeline of causal inference: given data, it first tries to discover the causal graph
if not provided, then it attempts to find possible variables as treatments and identify the causal effects, after which
a suitable estimator model will be trained to estimate the causal effects, and, finally, the policy is evaluated to suggest the best option
for each individual.

.. figure:: flow.png

    `Why` can help almost every part of the whole pipeline of causal inference.

Class Structures
================

.. class:: ylearn._why.Why(discrete_outcome=None, discrete_treatment=None, identifier='auto', discovery_model=None, discovery_options=None, estimator='auto', estimator_options=None, random_state=None)

    An all-in-one API for causal learning.

    :param bool, default=infer from outcome discrete_outcome:
    :param bool, default=infer from the first treatment discrete_treatment:
    :param str, default=auto' identifier: Avaliable options: 'auto' or 'discovery'
    :param str, optional, default=None discovery_model:
    :param dict, optional, default=None discovery_options: Parameters (key-values) to initialize the discovery model
    :param str, optional, default='auto' estimator: Name of a valid EstimatorModel. One can also pass an instance of a valid estimator model.
    :param dict, optional, default=None estimator_options: Parameters (key-values) to initialize the estimator model
    :param int, optional, default=None random_state:
    
    .. py:attribute:: `feature_names_in_`
        
        list of feature names seen during `fit` 
    
    .. py:attribute:: outcome_

        name of outcome

    .. py:attribute:: treatment_

        list of treatment names identified during `fit`
    
    .. py:attribute:: adjustment_

        list of adjustment names identified during `fit`
    
    .. py:attribute:: covariate_

        list of covariate names identified during `fit`
    
    .. py:attribute:: instrument_

        list of instrument names identified during `fit`
    
    .. py:attribute:: identifier_

        `identifier` object or None. Used to identify treatment/adjustment/covariate/instrument if they were not specified during `fit`

    .. py:attribute:: y_encoder_

        `LabelEncoder` object or None. Used to encode outcome if its dtype was not numeric.
    
    .. py:attribute:: preprocessor_
        
        `Pipeline` object to preprocess data during `fit`

    .. py:attribute:: estimators_

        estimators dict for each treatment where key is the treatment name and value is the `EstimatorModel` object

    .. py:method:: fit(data, outcome, *, treatment=None, adjustment=None, covariate=None, instrument=None, treatment_count_limit=None, copy=True, **kwargs)

        Fit the Why object, steps:
            
            1. encode outcome if its dtype is not numeric
            2. identify treatment and adjustment/covariate/instrument
            3. preprocess data
            4. fit causal estimators
        
        :returns: The fitted :py:class:`Why`.
        :rtype: instance of :py:class:`Why`

    .. py:method:: identify(data, outcome, *, treatment=None, adjustment=None, covariate=None, instrument=None, treatment_count_limit=None)

        Identify treatment and adjustment/covariate/instrument. 

        :returns: identified treatment, adjustment, covariate, instrument
        :rtypes: tuple

    .. py:method:: causal_graph()

        Get identified causal graph.

        :returns: Identified causal graph
        :rtype: instance of :py:class:`CausalGraph`

    .. py:method:: causal_effect(test_data=None, treat=None, control=None)

        Estimate the causal effect.

        :returns: causal effect of all treatments
        :rtype: pandas.DataFrame
    
    .. py:method:: individual_causal_effect(test_data, treat=None, control=None)

        Estimate the causal effect for each individual.

        :returns: individual causal effect of each treatment
        :rtype: pandas.DataFrame
    
    .. py:method:: whatif(data, new_value, treatment=None)

        Get counterfactual predictions when treatment is changed to new_value from its observational counterpart.

        :returns: The counterfactual prediction
        :rtype: pandas.Series
 
    .. py:method:: score(test_data=None, treat=None, control=None, scorer='auto')

        :returns: Socre of the estimator models
        :rtype: float
   
    .. py:method:: policy_tree(data, control=None, **kwargs)

        Get the policy tree

        :returns: The fitted instance of :py:class:`PolicyTree`.
        :rtype: instance of :py:class:`PolicyTree`

    .. py:method:: policy_interpreter(data, control=None, **kwargsa)

        Get the policy interpreter

        :returns: The fitted instance of :py:class:`PolicyInterpreter`.
        :rtype: instance of :py:class:`PolicyInterpreter`

    .. py:method:: plot_causal_graph()

        Plot the causal graph.
    
    .. py:method:: plot_policy_tree(Xtest, control=None, **kwargs)

        Plot the policy tree.
    
    .. py:method:: plot_policy_interpreter(data, control=None, **kwargs)

        Plot the interpreter.