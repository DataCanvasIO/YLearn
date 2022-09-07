***********************
Double Machine Learning
***********************

.. topic:: Notation

    We use capital letters for matrices and small letters for vectors. The treatment is denoted by :math:`x`, the outcome is 
    denoted by :math:`y`, the covariate is denoted by :math:`v`, and other adjustment set variables are :math:`w`. Greek letters are for error terms.

The double machine learning (DML) model [Chern2016]_ can be applied when all confounders of the treatment and outcome, variables that
simultaneously influence the treatment and outcome, are observed. Let :math:`y` be the outcome and :math:`x` be the treatment, 
a DML model solves the following causal effect estimation (CATE estimation):

.. math::

    y & = F(v) x + g(v, w) + \epsilon \\
    x & = h(v, w) + \eta

where :math:`F(v)` is the CATE conditional on the condition :math:`v`. Furthermore, to estimate :math:`F(v)`, we note that

.. math::

    y - \mathbb{E}[y|w, v] = F(v) (x - \mathbb{E}[x|w, v]) + \epsilon. 
    
Thus by first estimating :math:`\mathbb{E}[y|w, v]` and :math:`\mathbb{E}[x|w,v]` as

.. math::

    m(v, w) & = \mathbb{E}[y|w, v]\\
    h(v, w) & = \mathbb{E}[x|w,v],

we can get a new dataset :math:`(\tilde{y}, \tilde{x})` where

.. math::

    \tilde{y} & = y - m(v, w) \\
    \tilde{x} & = x - h(v, w)

such that  the relation between :math:`\tilde{y}` and :math:`\tilde{x}` is linear

.. math::

    \tilde{y} = F(v) \tilde(x) + \epsilon

which can be simply modeled by the linear regression model. 

On the other hand,  in the current version, :math:`F(v)` takes the form 

.. math::

    F_{ij}(v) = \sum_k H_{ijk} \rho_k(v).
    
where :math:`H` can be seen as a 3-rank tensor and :math:`\rho_k` is a function of the covariate :math:`v`, e.g., 
:math:`\rho(v) = v` in the simplest case. Therefore, the outcome :math:`y` can now be represented as 

.. math::

    y_i & = \sum_j F_{ij}x_j + g(v, w)_j + \epsilon \\
        & = \sum_j \sum_k H_{ijk}\rho_k(v)x_j + g(v, w)_j + \epsilon

In this sense, the linear regression problem between :math:`\tilde{y}` and :math:`\tilde{x}`
now becomes

.. math::

    \tilde{y}_i = \sum_j \sum_k H_{ijk}\rho_k(v) \tilde{x}_j + \epsilon.

.. topic:: Implementation

    In YLearn, we implement a double machine learning as in the algorithm described in the [Chern2016]_:

        1. Let k (cf_folds in our class) be an int. Form a k-fold random
        partition {..., (train_data_i, test_data_i), ...,
        (train_data_k, test_data_k)}.

        2. For each i, train y_model and x_model on train_data_i, then evaluate
        their performances in test_data_i whoes results will be saved as
        :math:`(\hat{y}_k, \hat{x}_k)`. All :math:`(\hat{y}_k, \hat{x}_k)` will be combined to give the new dataset
        :math:`(\hat{y}, \hat{x})`.

        3. Define the differences

        .. math::

            \tilde{y}& = y - \hat{y}, \\
            \tilde{x}&= (x - \hat{x}) \otimes v.

        Then form the new dataset :math:`(\tilde{y}, \tilde{x})`.

        4. Perform linear regression on the dataset :math:`(\tilde{y}, \tilde{x})` whose
        coefficients will be saved in a vector :math:`f`. The estimated CATE given :math:`v`
        will just be

        .. math::

            f \cdot v.

.. topic:: Example

    .. code-block:: python
        
        from sklearn.ensemble import RandomForestRegressor

        from ylearn.exp_dataset.exp_data import single_continuous_treatment
        from ylearn.estimator_model.double_ml import DoubleML

        # build the dataset
        train, val, treatment_effect = single_continuous_treatment()
        adjustment = train.columns[:-4]
        covariate = 'c_0'
        outcome = 'outcome'
        treatment = 'treatment'

        dml = DoubleML(x_model=RandomForestRegressor(), y_model=RandomForestRegressor(), cf_fold=3,)
        dml.fit(train, outcome, treatment, adjustment, covariate,)

    >>> 06-23 14:02:36 I ylearn.e.double_ml.py 684 - _fit_1st_stage: fitting x_model RandomForestRegressor
    >>> 06-23 14:02:39 I ylearn.e.double_ml.py 690 - _fit_1st_stage: fitting y_model RandomForestRegressor
    >>> DoubleML(x_model=RandomForestRegressor(), y_model=RandomForestRegressor(), yx_model=LinearRegression(), cf_fold=3)      

Class Structures
================

.. py:class:: ylearn.estimator_model.double_ml.DoubleML(x_model, y_model, yx_model=None, cf_fold=1, adjustment_transformer=None, covariate_transformer=None, random_state=2022, is_discrete_treatment=False, categories='auto')

    :param estimator, optional x_model: Machine learning models for fitting x. Any such models should implement
            the :py:func:`fit` and :py:func:`predict`` (also :py:func:`predict_proba` if x is discrete) methods.
    :param estimator, optional y_model: The machine learning model which is trained to modeling the outcome. Any valid y_model should implement the :py:func:`fit()` and :py:func:`predict()` methods.
    :param estimator, optional yx_model: Machine learning models for fitting the residual of y on residual of x. *Only support linear regression model in the current version.*
    
    :param int, default=1 cf_fold: The number of folds for performing cross fit in the first stage.
    :param transormer, optional, default=None, adjustment_transformer: Transformer for adjustment variables which can be used to generate new features of adjustment variables.
    :param transormer, optional, default=None, covariate_transformer: Transformer for covariate variables which can be used to generate new features of covariate variables.
    :param int, default=2022 random_state:
    :param bool, default=False is_discrete_treatment: If the treatment variables are discrete, set this to True.
    :param str, optional, default='auto' categories:
    
    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, **kwargs)
        
        Fit the DoubleML estimator model. Note that the training of a DML has two stages, where we implement them in
        :py:func:`_fit_1st_stage` and :py:func:`_fit_2nd_stage`.

        :param pandas.DataFrame data: Training dataset for training the estimator.
        :param list of str, optional outcome: Names of the outcome.
        :param list of str, optional treatment: Names of the treatment.
        :param list of str, optional, default=None adjustment: Names of the adjustment set ensuring the unconfoundness,
        :param list of str, optional, default=None covariate: Names of the covariate.

        :returns: The fitted model
        :rtype: an instance of DoubleML

    .. py:method:: estimate(data=None, treat=None, control=None, quantity=None)
        
        Estimate the causal effect with the type of the quantity.

        :param pandas.DataFrame, optional, default=None data: The test data for the estimator to evaluate the causal effect, note
            that the estimator directly evaluate all quantities in the training
            data if data is None.
        :param float or numpy.ndarray, optional, default=None treat: In the case of single discrete treatment, treat should be an int or
            str of one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, treat should be a list
            or an ndarray where treat[i] indicates the value of the i-th intended
            treatment, for example, when there are multiple
            discrete treatments, array(['run', 'read']) means the treat value of
            the first treatment is taken as 'run' and that of the second treatment
            is taken as 'read';
            in the case of continuous treatment, treat should be a float or a
            ndarray.
        :param str, optional, default=None quantity: Option for returned estimation result. The possible values of quantity include:
                
                1. *'CATE'* : the estimator will evaluate the CATE;
                
                2. *'ATE'* : the estimator will evaluate the ATE;
                
                3. *None* : the estimator will evaluate the ITE or CITE.
        :param float or numpy.ndarray, optional, default=None control: This is similar to the cases of treat.

        :returns: The estimated causal effects 
        :rtype: ndarray

    .. py:method:: effect_nji(data=None)
        
        Calculate causal effects with different treatment values. 
        
        :param pandas.DataFrame, optional, default=None data: The test data for the estimator to evaluate the causal effect, note
            that the estimator will use the training data if data is None.

        :returns: Causal effects with different treatment values.
        :rtype: ndarray

    .. py:method:: comp_transormer(x, categories='auto')
        
        Transform the discrete treatment into one-hot vectors properly.

        :param numpy.ndarray, shape (n, x_d) x:  An array containing the information of the treatment variables.
        :param str or list, optional, default='auto' categories:

        :returns: The transformed one-hot vectors.
        :rtype: numpy.ndarray

