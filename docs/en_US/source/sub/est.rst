.. _estimator_model:

**********************************************
Estimator Model: Estimating the Causal Effects
**********************************************

For a causal effect with :math:`do`-operator, after converting it into the corresponding statistical estimand with the approach called 
:ref:`identification`, the task of causal inference now becomes estimating the statistical estimand, the converted causal effect. 
Before diving into any specific estimation methods for causal effects, we briefly introduce the problem settings of the estimation
of causal effects. 

Problem Setting
===============

It is introduced in :ref:`causal_model` that every causal structure has a corresponding DAG called causal graph. Furthermore, 
each child-parent family in a DAG :math:`G` represents a deterministic function

.. math::

    X_i = F_i (pa_i, \eta_i), i = 1, \dots, n,

where :math:`pa_i` are parents of :math:`x_i` in :math:`G` and :math:`\eta_i` are random disturbances representing exogeneous not present in the
analysis. We call these functions **Structural Equation Model** related to the causal structures. For a set of variables :math:`W` that satisfies
the back-door criterion (see :ref:`identification`), the causal effect of :math:`X` on :math:`Y` is given by the formula

.. math::

    P(y|do(x))  = \sum_w P(y| x, w)P(w).

In such case, variables :math:`X` for which the above equality is valid are also named *"conditionally ignorable given* :math:`W`" in 
the *potential outcome* framework. The set of variables :math:`W` satisfying this condition is called **adjustment set**. And in
the language of structural equation model, these relations are encoded by

.. math::

    X & = F_1 (W, \epsilon),\\
    Y & = F_2 (W, X, \eta).

Our problems can be expressed with the structural equation model.

.. topic:: ATE

    Specifically, one particular important causal quantity in YLearn is the difference

    .. math::

        \mathbb{E}(Y|do(X=X_1)) - \mathbb{E}(Y|do(X=X_0))

    which is also called **average treatment effect (ATE)**, where :math:`Y` is called the *outcome* and :math:`X` is called the *treatment*. Furthermore,
    when the conditional independence (conditional ignorability) holds given a set of variables :math:`W` potentially having effects on both outcome
    :math:`Y`` and treatment :math:`X`, the ATE can be evaluated as

    .. math::

        E(Y|X=x_1, w) - E(Y|X=x_0, w).

    Using structural equation model we can describe the above relation as

    .. math::
        
        X & = F_1 (W, \epsilon) \\
        Y & = F_2 (X, W, \eta) \\
        \text{ATE} & = \mathbb{E}\left[ F_2(x_1, W, \eta) - F_2(x_0, W, \eta)\right]. 

.. topic:: CATE

    Suppose that we assign special roles to a subset of variables in the adjustment set :math:`W`` and name them as **covariates** :math:`V`, 
    then, in the structural equation model, the **CATE** (also called **heterogeneous treatment effect**) is defined by
    
    .. math::
    
        X & = F_1 (W, V, \epsilon) \\
        Y & = F_2 (X, W, V, \eta) \\
        \text{CATE} & = \mathbb{E}\left[ f_2(x_1, W, V, \eta) - f_2(x_0, W, V, \eta)| V =v\right].

.. topic:: Counterfactual

    Besides casual estimands which are differences of effects, there is also a causal quantity **counterfactual**.
    For such quantity, we estimate the following causal estimand:

    .. math::

        \mathbb{E} [Y|do(x), V=v].


Estimator Models
==========================
YLearn implements several estimator models for the estimation of causal effects:

.. toctree::
    :maxdepth: 2

    est_model/approx_bound
    est_model/meta
    est_model/dml
    est_model/dr
    est_model/causal_tree
    est_model/iv
    est_model/score
    
The evaluations of 

.. math::
    
    \mathbb{E}[F_2(x_1, W, \eta) - F_2(x_0, W, \eta)]
    
in ATE and 

.. math::
    
    \mathbb{E}[F_2(x_1, W, V, \eta) - F_2(x_0, W, V, \eta)]

in CATE will be the tasks of various suitable **estimator models** in YLearn. The concept :class:`EstimatorModel` in YLearn is designed for this purpose.

A typical :class:`EstimatorModel` should have the following structure:

.. code-block:: python

    class BaseEstModel:
        """
        Base class for various estimator model.

        Parameters
        ----------
        random_state : int, default=2022
        is_discrete_treatment : bool, default=False
            Set this to True if the treatment is discrete.
        is_discrete_outcome : bool, default=False
            Set this to True if the outcome is discrete.            
        categories : str, optional, default='auto'

        """
        def fit(
            self,
            data,
            outcome,
            treatment,
            **kwargs,
        ):
            """Fit the estimator model.

            Parameters
            ----------
            data : pandas.DataFrame
                The dataset used for training the model

            outcome : str or list of str, optional
                Names of the outcome variables

            treatment : str or list of str
                Names of the treatment variables

            Returns
            -------
            instance of BaseEstModel
                The fitted estimator model.
            """

        def estimate(
            self,
            data=None,
            quantity=None,
            **kwargs
        ):
            """Estimate the causal effect.

            Parameters
            ----------
            data : pd.DataFrame, optional
                The test data for the estimator to evaluate the causal effect, note
                that the estimator directly evaluate all quantities in the training
                data if data is None, by default None

            quantity : str, optional
                The possible values of quantity include:
                    'CATE' : the estimator will evaluate the CATE;
                    'ATE' : the estimator will evaluate the ATE;
                    None : the estimator will evaluate the ITE or CITE, by default None
    
            Returns
            -------
            ndarray
                The estimated causal effect with the type of the quantity.
            """

        def effect_nji(self, data=None, *args, **kwargs):
            """Return causal effects for all possible values of treatments.

            Parameters
            ----------
            data : pd.DataFrame, optional
                The test data for the estimator to evaluate the causal effect, note
                that the estimator directly evaluate all quantities in the training
                data if data is None, by default None
            """

.. topic:: Usage

    One can apply any :class:`EstimatorModel` in the following procedure:

    1. For the data in the form of :class:`pandas.DataFrame`, find the names of *treatment*, *outcome*, *adjustment*, and *covariate*.

    2. Pass the data along with names of treatments, outcomes, adjustment set, and covariates into the :meth:`fit()` method of :class:`EstimatorModel` and call it.

    3. Call the :meth:`estimate()` method to use the fitted :class:`EstimatorModel` on test data.