.. _causal_model:

************
Causal Model
************

:class:`CausalModel` is a core object for performing :ref:`identification` and finding
Instrumental Variables. 

Before introducing the causal model, we should clarify the **Interventions** first.
Interventions would be to take the whole population and give every one some operation. 
[Pearl]_ defined the :math:`do`-operator to describe such operations. Probabilistic models can not serve 
to predict the effect of interventions which leads to the need for causal model. 

The formal definition of **causal model** is due to [Pearl]_. A causal model is a triple

.. math::
    
    M = \left< U, V, F\right>

where

* :math:`U` are **exogenous** (variables that are determined by factors outside the model);
* :math:`V` are **endogenous** that are determined by :math:`U \cup V`, and :math:`F` is a set of functions such that

.. math::
        
        V_i = F_i(pa_i, U_i)

with :math:`pa_i \subset V \backslash V_i`. 

For example, :math:`M = \left< U, V, F\right>` is a causal model where

.. math::
    
    V = \{V_1, V_2\}, 
    
    U = \{ U_1, U_2, I, J\},
    
    F = \{F_1, F_2 \}

such that

.. math::

    V_1 = \theta_1 I + U_1\\
    V_2 = \phi V_1 + \theta_2 J + U_2.

Note that every causal model can be associated with a DAG and encodes necessary information of the causal relationships between variables.
YLearn uses :class:`CausalModel` to represent a causal model and support many operations related to the causal
model such as :ref:`identification`.

.. _identification:

Identification
==============

To characterize the effect of the intervention, one needs to consider the **causal effect** which is a 
causal estimand including the :math:`do`-operator. The action which converts the causal effect into corresponding 
statistical estimands is called :ref:`identification` in YLearn and is implemented in :class:`CausalModel`. Note that not 
all causal effect can be converted to statistical estimands. We refer such causal effects as not identifiable.

.. topic:: Backdoor adjustment

    The causal effect of :math:`X` on :math:`Y` is given by

    .. math::

        P(y|do(x)) = \sum_w P(y|x, w)P(w)
    
    if the set of variables :math:`W` satisfies the back-door criterion relative to :math:`(X, Y)`.

.. topic:: Frontdoor adjustment

    The causal effect of :math:`X` on :math:`Y` is given by

    .. math::

        P(y|do(x)) = \sum_w P(w|x) \sum_{x'}P(y|x', w)P(x')
    
    if the set of variables :math:`W` satisfies the front-door criterion relative to :math:`(X, Y)` and if
    :math:`P(x, w) > 0`.

.. topic:: General identification

    [Shpitser2006]_ gives a necessary and sufficient graphical condition such that the causal effect
    of an arbitrary set of variables on another arbitrary set can be identified uniquely whenever its identifiable. We 
    call the correspondig action of verifying this condition as **general identification**.

.. topic:: Finding Instrumental Variables

    Instrumental variables are useful to identify and estimate the causal effect of :math:`X` on :math:`Y` when there are 
    unobserved confoundings of :math:`X` and :math:`Y`. A set of variables :math:`Z` is said to be a set of **instrumental variables**
    if for any :math:`z` in :math:`Z`:
    
    1. :math:`z` has a causal effect on :math:`X`.
    
    2. The causal effect of :math:`z` on :math:`Y` is fully mediated by :math:`X`.
    
    3. There are no back-door paths from :math:`z` to :math:`Y`.


Class Structures
================

.. py:class:: CausalModel(causal_graph=None, data=None)

    :param CausalGraph, optional, default=None causal_graph: An instance of CausalGraph which encodes the causal structures.
    :param pandas.DataFrame, optional, default=None data: The data used to discover the causal structures if causal_graph is not provided.

    .. py:method:: id(y, x, prob=None, graph=None)
        
        Identify the causal quantity :math:`P(y|do(x))` if identifiable else return
        raise :class:`IdentificationError`. 
        Note that here we only consider semi-Markovian causal model, where
        each unobserved variable is a parent of exactly two nodes. This is
        because any causal model with unobserved variables can be converted
        to a semi-Markovian causal model encoding the same set of conditional
        independences.

        :param set of str y: Set of names of outcomes.
        :param set of str x: Set of names of treatments.
        :param Prob, optional, default=None prob: Probability distribution encoded in the graph.
        :param CausalGraph graph: CausalGraph encodes the information of correspondig causal structures.

        :returns: The probabiity distribution of the converted casual effect.
        :rtype: Prob
        :raises IdentificationError: If the interested causal effect is not identifiable, then raise IdentificationError.

    .. py:method:: is_valid_backdoor_set(set_, treatment, outcome)

        Determine if a given set is a valid backdoor adjustment set for
        causal effect of treatments on the outcomes.

        :param set set_: The adjustment set.
        :param set or list of str treatment: Names of the treatment. str is also acceptable for single treatment.
        :param set or list of str outcome: Names of the outcome. str is also acceptable for single outcome.

        :returns: True if the given set is a valid backdoor adjustment set for the 
                causal effect of treatment on outcome in the current causal graph.
        :rtype: bool

    .. py:method::  get_backdoor_set(treatment, outcome, adjust='simple', print_info=False)
        
        Return the backdoor adjustment set for the given treatment and outcome.

        :param set or list of str treatment: Names of the treatment. str is also acceptable for single treatment.
        :param set or list of str outcome: Names of the outcome. str is also acceptable for single outcome.
        :param str adjust: Set style of the backdoor set. Avaliable options are
                
                simple: directly return the parent set of treatment
                
                minimal: return the minimal backdoor adjustment set
                
                all: return all valid backdoor adjustment set.
        
        :param bool, default=False print_info: If True, print the identified results.

        :returns: The first element is the adjustment list, while the second is the
                encoded Prob.
        :rtype: tuple of two element
        :raises IdentificationError: Raise error if the style is not in simple, minimal or all or no
                set can satisfy the backdoor criterion.

    .. py:method:: get_backdoor_path(treatment, outcome)

        Return all backdoor paths connecting treatment and outcome.

        :param str treatment: Name of the treatment.
        :param str outcome: Name of the outcome

        :returns: A list containing all valid backdoor paths between the treatment and
                outcome in the graph.
        :rtype: list

    .. py:method:: has_collider(path, backdoor_path=True)

        If the path in the current graph has a collider, return True, else
        return False.

        :param list of str path: A list containing nodes in the path.
        :param bool, default=True backdoor_path: Whether the path is a backdoor path.

        :returns: True if the path has a colider.
        :rtype: bool

    .. py:method:: is_connected_backdoor_path(path)

        Test whether a backdoor path is connected.

        :param list of str path: A list describing the path.

        :returns: True if path is a d-connected backdoor path and False otherwise.
        :rtype: bool

    .. py:method:: is_frontdoor_set(set_, treatment, outcome)

        Determine if the given set is a valid frontdoor adjustment set for the
        causal effect of treatment on outcome.

        :param set set_: The set waited to be determined as a valid front-door adjustment set.
        :param str treatment: Name of the treatment.
        :param str outcome: Name of the outcome.

        :returns: True if the given set is a valid frontdoor adjustment set for causal effects
                of treatemtns on outcomes.
        :rtype: bool

    .. py:method:: get_frontdoor_set(treatment, outcome, adjust='simple')

        Return the frontdoor set for adjusting the causal effect between
        treatment and outcome.

        :param set of str or str treatment: Name of the treatment. Should contain only one element.
        :param set of str or str outcome: Name of the outcome. Should contain only one element.
        :param str, default='simple' adjust: Avaliable options include 
                'simple': Return the frontdoor set with minimal number of elements.
                
                'minimal': Return the frontdoor set with minimal number of elements.
                
                'all': Return all possible frontdoor sets.
        
        :returns: 2 elements (adjustment_set, Prob)
        :rtype: tuple
        :raises IdentificationError: Raise error if the style is not in simple, minimal or all or no
                set can satisfy the backdoor criterion.

    .. py:method:: get_iv(treatment, outcome)

        Find the instrumental variables for the causal effect of the
        treatment on the outcome.

        :param iterable treatment: Name(s) of the treatment.
        :param iterable outcome: Name(s) of the outcome.

        :returns: A valid instrumental variable set which will be an empty one if
                there is no such set.
        :rtype: set

    .. py:method:: is_valid_iv(treatment, outcome, set_)

        Determine whether a given set is a valid instrumental variable set.

        :param iterable treatment: Name(s) of the treatment.
        :param iterable outcome: Name(s) of the outcome.
        :param set set_: The set waited to be tested.

        :returns: True if the set is a valid instrumental variable set and False
                otherwise.
        :rtype: bool

    .. py:method:: identify(treatment, outcome, identify_method='auto')
        
        Identify the causal effect expression. Identification is an operation that
        converts any causal effect quantity, e.g., quantities with the do operator, into
        the corresponding statistical quantity such that it is then possible
        to estimate the causal effect in some given data. However, note that not all
        causal quantities are identifiable, in which case an IdentificationError
        will be raised.

        :param set or list of str treatment: Set of names of treatments.
        :param set or list of str outcome: Set of names of outcomes.
        :param tuple of str or str, optional, default='auto' identify_method: If the passed value is a tuple or list, then it should have two
                elements where the first one is for the identification methods
                and the second is for the returned set style.

                Available options:
                
                    'auto' : Perform identification with all possible methods
                    
                    'general': The general identification method, see id()
                    
                    *('backdoor', 'simple')*: Return the set of all direct confounders of
                    both treatments and outcomes as a backdoor
                    adjustment set.
                    
                    *('backdoor', 'minimal')*: Return all possible backdoor adjustment sets with
                    minial number of elements.
                    
                    *('backdoor', 'all')*: Return all possible backdoor adjustment sets.
                    
                    *('frontdoor', 'simple')*: Return all possible frontdoor adjustment sets with
                    minial number of elements.
                    
                    *('frontdoor', 'minimal')*: Return all possible frontdoor adjustment sets with
                    minial number of elements.
                    
                    *('frontdoor', 'all')*: Return all possible frontdoor adjustment sets.

        :returns: A python dict where keys of the dict are identify methods while the values are the
                corresponding results.
        :rtype: dict
        :raises IdentificationError: If the causal effect is not identifiable or if the identify_method was not given properly.

    .. py:method:: estimate(estimator_model, data=None, *, treatment=None, outcome=None, adjustment=None, covariate=None, quantity=None, **kwargs)

        Estimate the identified causal effect in a new dataset.

        :param EstimatorModel estimator_model: Any suitable estimator models implemented in the EstimatorModel can
                be applied here. 
        :param pandas.DataFrame, optional, default=None data: The data set for causal effect to be estimated. If None, use the data
                which is used for discovering causal graph.
        :param  set or list, optional, default=None treatment: Names of the treatment. If None, the treatment used for backdoor adjustment
                will be taken as the treatment.
        :param set or list, optional, default=None outcome: Names of the outcome. If None, the treatment used for backdoor adjustment
                will be taken as the outcome.
        :param set or list, optional, default=None adjustment: Names of the adjustment set. If None, the ajustment set is given by
                the simplest backdoor set found by CausalModel.
        :param set or list, optional, default=None covariate: Names of covariate set. Ignored if set as None.
        :param str, optional, default=None quantity: The interested quantity when evaluating causal effects.

        :returns: The estimated causal effect in data.
        :rtype: np.ndarray or float

    .. py:method:: identify_estimate(data, outcome, treatment, estimator_model=None, quantity=None, identify_method='auto', **kwargs)

        Combination of the identifiy method and the estimate method. However,
        since current implemented estimator models assume (conditionally)
        unconfoundness automatically (except for methods related to iv), we may
        only consider using backdoor set adjustment to fullfill the unconfoundness
        condition.

        :param set or list of str, optional treatment: Set of names of treatments.
        :param set or list of str, optional outcome: Set of names of outcome.
        :param tuple of str or str, optional, default='auto' identify_method: If the passed value is a tuple or list, then it should have two
                elements where the first one is for the identification methods
                and the second is for the returned set style.

                Available options:
                
                    'auto' : Perform identification with all possible methods
                    
                    'general': The general identification method, see id()
                    
                    *('backdoor', 'simple')*: Return the set of all direct confounders of
                    both treatments and outcomes as a backdoor adjustment set.
                    
                    *('backdoor', 'minimal')*: Return all possible backdoor adjustment sets with minial number of elements.
                    
                    *('backdoor', 'all')*: Return all possible backdoor adjustment sets.
                    
                    *('frontdoor', 'simple')*: Return all possible frontdoor adjustment sets with minial number of elements.
                    
                    *('frontdoor', 'minimal')*: Return all possible frontdoor adjustment sets with minial number of elements.
                    
                    *('frontdoor', 'all')*: Return all possible frontdoor adjustment sets.
        
        :param str, optional, default=None quantity: The interested quantity when evaluating causal effects.

        :returns: The estimated causal effect in data.
        :rtype: np.ndarray or float


.. topic:: Example 1: Identify the causal effect with the general identification method

    .. figure:: graph_un_arc.png
        
        Causal structures where all unobserved variables are removed and their related causations are replaced by
        the confounding arcs (black doted lines with two arrows).
    
    For the causal structure in the figure, we want to identify the causal effect of :math:`X` on :math:`Y` using the *general identification* method. The first
    step is to represent the causal structure with :class:`CausalModel`.
    
    .. code-block:: python
        
        from ylearn.causal_model.graph import CausalGraph
        
        causation = {
            'X': ['Z2'],
            'Z1': ['X', 'Z2'],
            'Y': ['Z1', 'Z3'],
            'Z3': ['Z2'],
            'Z2': [], 
        }
        arcs = [('X', 'Z2'), ('X', 'Z3'), ('X', 'Y'), ('Z2', 'Y')]
        cg = CausalGraph(causation=causation, latent_confounding_arcs=arcs)

    Then we need to define an instance of :class:`CausalModel` for the causal structure encoded in :py:attr:`cg` to preform the identification.

    .. code-block:: python

        from ylearn.causal_model.model import CausalModel
        cm = CausalModel(causal_model=cg)
        stat_estimand = cm.id(y={'Y'}, x={'X'})
        stat_estimand.show_latex_expression()

    >>> :math:`\sum_{Z3, Z1, Z2}[P(Z2)P(Y|Z3, Z2)][P(Z1|Z2, X)][P(Z3|Z2)]`

    The result is the desired identified causal effect of :math:`X` on :math:`Y` in the given causal structure.

.. topic:: Example 2: Identify the causal effect with the back-door adjustment

    .. figure:: backdoor.png

        All nodes are observed variables.
    
    For the causal structure in the figure, we want to identify the causal effect of :math:`X` on :math:`Y` using the *back-door adjustment* method.
    
    .. code-block:: python
        
        from ylearn.causal_model.graph import CausalGraph
        from ylearn.causal_model.model import CausalModel

        causation = {
            'X1': [], 
            'X2': [], 
            'X3': ['X1'], 
            'X4': ['X1', 'X2'], 
            'X5': ['X2'], 
            'X6': ['X'], 
            'X': ['X3', 'X4'], 
            'Y': ['X6', 'X4', 'X5', 'X'], 
        } 

        cg = CausalGraph(causation=causation)
        cm = CausalModel(causal_graph=cg)
        backdoor_set, prob = cm3.identify(treatment={'X'}, outcome={'Y'}, identify_method=('backdoor', 'simple'))['backdoor']

        print(backdoor_set)


    >>> ['X3', 'X4']

.. topic:: Example 3: Find the valid instrumental variables

    .. figure:: iv1.png

        Causal structure for the variables :math:`p, t, l, g`

    We want to find the valid instrumental variables for the causal effect of :math:`t` on :math:`g`.

    .. code-block:: python

        causation = {
            'p':[],
            't': ['p'],
            'l': ['p'],
            'g': ['t', 'l']
        }
        arc = [('t', 'g')]
        cg = CausalGraph(causation=causation, latent_confounding_arcs=arc)
        cm = CausalModel(causal_graph=cg)

        cm.get_iv('t', 'g')

    >>> No valid instrument variable has been found.

    .. figure:: iv2.png

        Another causal structure for the variables :math:`p, t, l, g`

    We still want to find the valid instrumental variables for the causal effect of :math:`t` on :math:`g`
    in this new causal structure.

    .. code-block:: python

        causation = {
            'p':[],
            't': ['p', 'l'],
            'l': [],
            'g': ['t', 'l']
        }
        arc = [('t', 'g')]
        cg = CausalGraph(causation=causation, latent_confounding_arcs=arc)
        cm = CausalModel(causal_graph=cg)

        cm.get_iv('t', 'g')
    
    >>> {'p'}