.. _ce_int:

*************
CEInterpreter
*************

For the CATE :math:`\tau(v)` estimated by a estimator model, e.g., double machine learning model, :class:`CEInterpreter` interprets the result
by building a decision tree to model the relationships between :math:`\tau(v)` and the covariates :math:`v`. Then one can use the decision rules 
of the fitted tree model to analyze :math:`\tau(v)`.

Class Structures
================

.. autoclass:: ylearn.effect_interpreter.ce_interpreter.CEInterpreter

    .. py:method:: fit(data, est_model, **kwargs)
        
        Fit the CEInterpreter model to interpret the causal effect estimated
        by the est_model on data.

        :param pandas.DataFrame data: The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.
        :param estimator_model est_model: est_model should be any valid estimator model of ylearn which was 
            already fitted and can estimate the CATE.
        
        :returns: Fitted CEInterpreter
        :rtype: instance of CEInterpreter

    .. py:method:: interpret(*, v=None, data=None)

        Interpret the fitted model in the test data.

        :param numpy.ndarray, optional, default=None v: The test covariates in the form of ndarray. If this is given, then data will be ignored and the model will use this as the test data.
        :param pandas.DataFrame, optional, default=None data: The test data in the form of the DataFrame. The model will only use this if v is set as None. In this case, if data is also None, then the data used for trainig will be used.

        :returns: The interpreted results for all examples.
        :rtype: dict

    .. py:method:: plot(*, feature_names=None, max_depth=None, class_names=None, label='all', filled=False, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None)

        Plot a policy tree.
        The sample counts that are shown are weighted with any sample_weights that
        might be present.
        The visualization is fit automatically to the size of the axis.
        Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
        the size of the rendering.

        :returns: List containing the artists for the annotation boxes making up the
            tree.
        :rtype: annotations : list of artists
    
.. topic:: Example

    pass