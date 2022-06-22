*************
Doubly Robust
*************

The doubly robust method (see [Funk2010]_) estimates the causal effects when the treatment is discrete and the unconfoundness condition is satisified.
Training a doubly robust model is composed of 3 steps.

1. Let :math:`k` be an int. Form a :math:`K`-fold random
   partition for the data :math:`\{(X_i, W_i, V_i, Y_i)\}_{i = 1}^n` such that

   .. math::

        \{(x_i, w_i, v_i, y_i)\}_{i = 1}^n = D_j \cup T_k

   where :math:`D_k` stands for the trainig data while :math:`T_k` stands for the test data and :math:`\cup_{k = 1}^K T_k = \{(X_i, W_i, V_i, Y_i)\}_{i = 1}^n`.

2. For each :math:`k`, train two models :math:`f(X, W, V)` and :math:`g(W, V)` on :math:`D_k` to predict :math:`y` and :math:`x`, respectively. Then evaluate
   their performances in :math:`T_k` whoes results will be saved as :math:`\{(\hat{X}, \hat{Y})\}_k`. All :math:`\{(\hat{X}, \hat{Y})\}_k` will be combined to
   give the new dataset :math:`\{(\hat{X}_i, \hat{Y}_i(X, W, V))\}_{i = 1}^n`. 

3. For any given pair of treat group where :math:`X=x` and control group where :math:`X = x_0`, we build the final dataset :math:`\{(V, \tilde{Y}_x - \tilde{Y}_0)\}` where :math:`\tilde{Y}_x`
   is difined as

   .. math::

        \tilde{Y}_x & = \hat{Y}(X=x, W, V) + \frac{(Y - \hat{Y}(X=x, W, V)) * \mathbb{I}(X=x)}{P[X=x| W, V]} \\
        \tilde{Y}_0 & = \hat{Y}(X=x_0, W, V) + \frac{(Y - \hat{Y}(X=x_0, W, V)) * \mathbb{I}(X=x_0)}{P[X=x_0| W, V]}
    
   and train the final machine learing model :math:`h(W, V)` on this dataset to predict the causal effect :math:`\tau(V)`

   .. math::

       \tau(V) =  \tilde{Y}_x - \tilde{Y}_0 = h(V).
    
   Then we can directly estimate the causal effects by passing the covariate :math:`V` to
   the model :math:`h(V)`.

Class Structures
================

.. py:class:: ylearn.estimator_model.doubly_robust.DoublyRobust(x_model, y_model, yx_model, cf_fold=1, random_state=2022, categories='auto')

    :param estimator, optional x_model: The machine learning model which is trained to modeling the treatment. Any valid x_model should implement the :py:func:`fit()` and :py:func:`predict_proba()` methods.
    :param estimator, optional y_model: The machine learning model which is trained to modeling the outcome with covariates (possibly adjustment) and the  treatment. Any valid y_model should implement the :py:func:`fit()` and :py:func:`predict()` methods.
    :param estimator, optional yx_model: The machine learning model which is trained in the final stage of doubly robust method to modeling the causal effects with covariates (possibly adjustment). Any valid yx_model should implement the :py:func:`fit()` and :py:func:`predict()` methods.
    
    :param int, default=1 cf_fold: The nubmer of folds for performing cross fit in the first stage.
    :param int, default=2022 random_state:
    :param str, optional, default='auto' categories:
    
    .. py:method:: fit(data, outcome, treatment, adjustment=None, covariate=None, treat=None, control=None, combined_treatment=True, **kwargs)
        
        Fit the DoublyRobust estimator model. Note that the trainig of a doubly robust model has three stages, where we implement them in 
        :py:func:`_fit_1st_stage` and :py:func:`_fit_2nd_stage`.

        :param pandas.DataFrame data: Training dataset for training the estimator.
        :param list of str, optional outcome: Names of the outcome.
        :param list of str, optional treatment: Names of the treatment.
        :param list of str, optional, default=None adjustment: Names of the adjustment set ensuring the unconfoundness,
        :param list of str, optional, default=None covariate: Names of the covariate.
        :param int, optional treat: Label of the intended treatment group. If None, then :py:attr:`treat` will be set as 1. In the case of single discrete treatment, treat should be an int or
            str in one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, treat should be a list
            or an ndarray where treat[i] indicates the value of the i-th intended
            treatment. For example, when there are multiple
            discrete treatments, array(['run', 'read']) means the treat value of
            the first treatment is taken as 'run' and that of the second treatment
            is taken as 'read'.
        :param int, optional control: Label of the intended control group. This is similar to the cases of treat. If None, then :py:attr:`control` will be set as 0.


        :returns: The fitted instance of DoublyRobust.
        :rtype: instance of DoublyRobust

    .. py:method:: estimate(data=None, quantity=None, treat=None, all_tr_effects=False)
        
        Estimate the causal effect with the type of the quantity.

        :param pandas.DataFrame, optional, default=None data: Test data. The model will use the training data if set as None.
        :param str, optional, default=None quantity: Option for returned estimation result. The possible values of quantity include:
                
                1. *'CATE'* : the estimator will evaluate the CATE;
                
                2. *'ATE'* : the estimator will evaluate the ATE;
                
                3. *None* : the estimator will evaluate the ITE or CITE.
        :param float or numpy.ndarray, optional, default=None treat: In the case of single discrete treatment, treat should be an int or
            str in one of all possible treatment values which indicates the
            value of the intended treatment;
            in the case of multiple discrete treatment, treat should be a list
            or an ndarray where treat[i] indicates the value of the i-th intended
            treatment. For example, when there are multiple
            discrete treatments, array(['run', 'read']) means the treat value of
            the first treatment is taken as 'run' and that of the second treatment
            is taken as 'read'.
        :param bool, default=False, all_tr_effects: If True, return all causal effects with all values of :py:attr:`treatments`, otherwise
            only return the causal effect of the treatment with the value of 
            :py:attr:`treat` if it is provided. If :py:attr:`treat` is not provided, then the value of
            treatment is taken as the value of that when fitting the estimator model.

        :returns: The estimated causal effects 
        :rtype: ndarray

    .. py:method:: effect_nji(data=None)
        
        Calculate causal effects with different treatment values. Note that this method only will convert any 
        problem with discrete treatment into that with binary treatment. One can use :py:func:`_effect_nji_all` to get casual effects with all
        values of :py:attr:`treat` taken by :py:attr:`treatment`.

        :returns: Causal effects with different treatment values.
        :rtype: ndarray

    .. py:method:: comp_transormer(x, categories='auto')
        
        Transform the discrete treatment into one-hot vectors properly.

        :param numpy.ndarray, shape (n, x_d) x:  An array containing the information of the treatment variables.
        :param str or list, optional, default='auto' categories:

        :returns: The transformed one-hot vectors.
        :rtype: numpy.ndarray

.. topic:: Example

    pass