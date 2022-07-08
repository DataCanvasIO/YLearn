************************************
Nonparametric Instrumental Variables
************************************

Two-stage Least Squares
=======================
When the relationship between the outcome :math:`y`, treatment :math:`x` and covariate :math:`v` are assumed to be linear, e.g., [Angrist1996]_,

.. math::

    y & = \alpha x + \beta v + e \\
    x & = \gamma z + \lambda v + \eta,

then the IV framework becomes direct: it will first train a linear model for :math:`x` given :math:`z` and :math:`v`, then it replaces :math:`x`
with the predicted values :math:`\hat{x}` to train a linear model for :math:`y` in the second stage. This procedure is called the two-stage least-squares (2SLS).

Nonparametric IV
================
Removing the linear assumptions regarding the relationships between variables, the nonparametric IV can replace the linear regression with a linear projection 
onto a series of known basis functions [Newey2002]_. 

This method is similar to the conventional 2SLS and is also composed of 2 stages after finding new features of :math:`x`, :math:`v`, and :math:`z`,  

.. math:: 
    
    \tilde{z}_d & = f_d(z)\\
    \tilde{v}_{\mu} & = g_{\mu}(v),

which are represented by some non-linear functions (basis functions) :math:`f_d` and :math:`g_{\mu}`. After transforming into the new spaces, we then
    
    1. Fit the treatment model:
    
    .. math::

        \hat{x}(z, v, w) = \sum_{d, \mu} A_{d, \mu} \tilde{z}_d \tilde{v}_{\mu} + h(v, w) + \eta
    
    2. Generate new treatments x_hat, and then fit the outcome model

    .. math::
        
        y(\hat{x}, v, w) = \sum_{m, \mu} B_{m, \mu} \psi_m(\hat{x}) \tilde{v}_{\mu} + k(v, w) 
        + \epsilon.

The final causal effect can then be estimated. For an example, the CATE given :math:`v` is estimated as
    
    .. math::
        
        y(\hat{x_t}, v, w) - y(\hat{x_0}, v, w) = \sum_{m, \mu} B_{m, \mu} (\psi_m(\hat{x_t}) - \psi_m(\hat{x_0})) \tilde{v}_{\mu}.

YLearn implement this procedure in the class :class:`NP2SLS`.

Class structures
================

.. py:class:: ylearn.estimator_model.iv.NP2SLS(x_model=None, y_model=None, random_state=2022, is_discrete_treatment=False, is_discrete_outcome=False, categories='auto')

    :param estimator, optional, default=None x_model: The machine learning model to model the treatment. Any valid x_model should implement the `fit` and `predict` methods, by default None
    :param estimator, optional, default=None y_model: The machine learning model to model the outcome. Any valid y_model should implement the `fit` and `predict` methods, by default None
    :param int, default=2022 random_state:
    :param bool, default=False is_discrete_treatment: 
    :param bool, default=False is_discrete_outcome: 
    :param str, optional, default='auto' categories:

    .. py:method:: fit(data, outcome, treatment, instrument, is_discrete_instrument=False, treatment_basis=('Poly', 2), instrument_basis=('Poly', 2), covar_basis=('Poly', 2), adjustment=None, covariate=None, **kwargs)

        Fit a NP2SLS. Note that when both treatment_basis and instrument_basis have degree
        1 we are actually doing 2SLS.

        :param DataFrame data: Training data for the model.
        :param str or list of str, optional outcome: Names of the outcomes.
        :param str or list of str, optional treatment: Names of the treatment.
        :param str or list of str, optional, default=None covariate: Names of the covariate vectors.
        :param str or list of str, optional instrument: Names of the instrument variables.
        :param str or list of str, optional, default=None adjustment: Names of the adjustment variables.
        :param tuple of 2 elements, optional, default=('Poly', 2) treatment_basis: Option for transforming the original treatment vectors. The first element indicates the transformation basis function while the second one denotes the degree. Currently only support 'Poly' in the first element.
        :param tuple of 2 elements, optional, default=('Poly', 2) instrument_basis: Option for transforming the original instrument vectors. The first element indicates the transformation basis function while the second one denotes the degree. Currently only support 'Poly' in the first element.
        :param tuple of 2 elements, optional, default=('Poly', 2) covar_basis: Option for transforming the original covariate vectors. The first element indicates the transformation basis function while the second one denotes the degree. Currently only support 'Poly' in the first element.
        :param bool, default=False is_discrete_instrument:

    .. py:method:: estimate(data=None, treat=None, control=None, quantity=None)

        Estimate the causal effect of the treatment on the outcome in data.

        :param pandas.DataFrame, optional, default=None data: If None, data will be set as the training data.
        :param str, optional, default=None quantity: Option for returned estimation result. The possible values of quantity include:
                
                1. *'CATE'* : the estimator will evaluate the CATE;
                
                2. *'ATE'* : the estimator will evaluate the ATE;
                
                3. *None* : the estimator will evaluate the ITE or CITE.
        :param float, optional, default=None treat: Value of the treament when imposing intervention. If None, then `treat` will be set to 1.
        :param float, optional, default=None control: Value of the treament such that the treament effect is :math:`y(do(x=treat)) - y (do(x = control))`.

        :returns: The estimated causal effect with the type of the quantity.
        :rtype: ndarray or float, optional

    .. py:method:: effect_nji(data=None)

        Calculate causal effects with different treatment values. 
        
        :param pandas.DataFrame, optional, default=None data: The test data for the estimator to evaluate the causal effect, note
            that the estimator will use the training data if data is None.

        :returns: Causal effects with different treatment values.
        :rtype: ndarray

.. topic:: Example

    pass
