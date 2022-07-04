*****************************************************
Causal Model: The Representation of Causal Structures
*****************************************************

.. toctree::
    :maxdepth: 1

    causal_model/graph
    causal_model/causal_model
    causal_model/prob
    

For a set of variables :math:`V`, its **causal structure** can be represented by a directed acylic 
graph (DAG), where each node corresponds to an element of :math:`V` while each direct functional 
relationship among the corresponding variables can be represented by a link in the DAG. A causal
structure guides the precise specification of how each variable is influenced by its parents in the
DAG. For an instance, :math:`X \leftarrow W \rightarrow Y` denotes that :math:`W` is a parent, thus 
also a common cause, of :math:`X` and :math:`Y`. More specifically, for two distinct variables :math:`V_i` 
and :math:`V_j`, if their functional relationship is

.. math::
    
    V_j = f(V_i, \eta_{ij})

for some function :math:`f` and noise :math:`\eta`, then in the DAG representing the causal structure of the set of variables 
:math:`V`, there should be an arrow pointing to :math:`V_i` from :math:`V_j`. A detailed introduction to
such DAGs for causal structures can be found in [Pearl]_.

A causal effect, also named as causal estimand, can be expressed with the :math:`do`-operator according to
[Pearl]_. As an example,

.. math::

    P(y|do(x))

denotes the probability function of :math:`y` after imposing the intervention :math:`x`. Causal structures
are crucial to expressing and estimating interested causal estimands. YLearn implements an object, 
``CausalGraph``, to support representations for causal structures and related operations of the 
causal structures. Please see :ref:`causal_graph` for details.

YLearn concerns the intersection of causal inference and machine learning. Therefore we assume that we have
abundant observational data rather than having the access to design randomized experiments. Then Given a DAG
for some causal structure, the causal estimands, e.g., the average treatment effects (ATEs), usually can not
be directly estimated from the data due to the counterfactuals which can never be observed. Thus it is 
necessary to convert these causal estimands into other quantities, which can be called as statistical estimands
and can be estimated from data, before proceeding to any estimation. The procedure of converting a causal
estimand into the corresponding statistical estimand is called **identification**.

The object for supporting identification and other related operations of causal structures is ``CausalModel``.
More details can be found in :ref:`causal_model`.

In the language of Pearl's causal inference, it is also necessary to represent the results
in the language of probability. For this purpose, YLearn also implements an object :class:`Prob` which is introduced in 
:ref:`prob`.