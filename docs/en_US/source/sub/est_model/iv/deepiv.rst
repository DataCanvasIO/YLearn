******
DeepIV
******

DeepIV, developed in [Hartford]_, is a method for estimating the causal effects in the presence of the unobserved confounder
between treatment and outcome variables. It applies deep learning methods to accurately characterize the causal relationships
between the treatment and outcome when the instrumental variables (IV) are present. Due to the representation powers of deep learning
models, it does not assume any parametric forms for the causal relationships. 

Training a DeepIV has two steps and resembles the estimation procedure of a normal IV method. Specifically, we

    1. train a neural network, which we refer to as the *treatment network* :math:`F(Z, V)`, to estimate the distribution of the treatment :math:`X` given the IV :math:`Z` and covariate variables :math:`V`

    2. train another neural network, which we refer to as the *outcome network* :math:`H(X, V)`, to estimate the outcome :math:`Y` given treatment :math:`X` and covariate variables :math:`V`.

The final causal effect can then be estimated by the outcome network :math:`H(X, W)`. For an instance, the CATE :math:`\tau(v)` is estimated as

.. math::

    \tau(v) = H(X=x_t, V=v) - H(X=x_0, W=v).


Class Structures
================

.. py:class:: ylearn.estimator_model.deepiv.DeepIV(x_net=None, y_net=None, x_hidden_d=None, y_hidden_d=None, num_gaussian=5, is_discrete_treatment=False, is_discrete_outcome=False, is_discrete_instrument=False, categories='auto', random_state=2022)

    :param ylearn.estimator_model.deepiv.Net, optional, default=None x_net: Representation of the mixture density network for continuous
            treatment or an usual classification net for discrete treatment. If None, the default neural network will be used. See :py:class:`ylearn.estimator_model.deepiv.Net` for reference.
    :param ylearn.estimator_model.deepiv.Net, optional, default=None y_net: Representation of the outcome network. If None, the default neural network will be used.
    :param int, optional, default=None x_hidden_d: Dimension of the hidden layer of the default x_net of DeepIV.
    :param int, optional, default=None y_hidden_d: Dimension of the hidden layer of the default y_net of DeepIV.
    :param bool, default=False is_discrete_treatment:
    :param bool, default=False is_discrete_instrument:
    :param bool, default=False is_discrete_outcome:

    :param int, default=5 num_gaussian: Number of gaussians when using the mixture density network which will be directly ignored when the treatment is discrete.
    :param int, default=2022 random_state:
    :param str, optional, default='auto' categories:
    
    .. py:method:: fit(data, outcome, treatment, instrument=None, adjustment=None, approx_grad=True, sample_n=None, y_net_config=None, x_net_config=None, **kwargs)
        
        Train the DeepIV model.

        :param pandas.DataFrame data: Training dataset for training the estimator.
        :param list of str, optional outcome: Names of the outcome.
        :param list of str, optional treatment: Names of the treatment.
        :param list of str, optional instrument: Names of the IV. Must provide for DeepIV.
        :param list of str, optional, default=None adjustment: Names of the adjustment set ensuring the unconfoundness, which can also be seen as the covariates in the current version.
        :param bool, default=True approx_grad:  Whether use the approximated gradient as in [Hartford]_.
        :param int, optional, default=None sample_n: Times of new samples when using the approx_grad technique.
        :param dict, optional, default=None x_net_config: Configuration of the x_net.
        :param dict, optional, default=None y_net_config: Configuration of the y_net.
        
        :returns: The trained DeepIV model
        :rtype: instance of DeepIV 

    .. py:method:: estimate(data=None, treat=None, control=None, quantity=None, marginal_effect=False, *args, **kwargs)
        
        Estimate the causal effect with the type of the quantity.

        :param pandas.DataFrame, optional, default=None data: Test data. The model will use the training data if set as None.
        :param str, optional, default=None quantity: Option for returned estimation result. The possible values of quantity include:
                
                1. *'CATE'* : the estimator will evaluate the CATE;
                
                2. *'ATE'* : the estimator will evaluate the ATE;
                
                3. *None* : the estimator will evaluate the ITE or CITE.
        :param int, optional, default=None treat: Value of the treatment, by default None. If None, then the model will set treat=1.
        :param int, optional, default=None control: Value of the control, by default None. If None, then the model will set control=1.

        :returns: Estimated causal effects 
        :rtype: torch.tensor

    .. py:method:: effect_nji(data=None)
        
        Calculate causal effects with different treatment values.

        :returns: Causal effects with different treatment values.
        :rtype: ndarray

    .. py:method:: comp_transormer(x, categories='auto')
        
        Transform the discrete treatment into one-hot vectors properly.

        :param numpy.ndarray, shape (n, x_d) x:  An array containing the information of the treatment variables.
        :param str or list, optional, default='auto' categories:

        :returns: The transformed one-hot vectors.
        :rtype: numpy.ndarray