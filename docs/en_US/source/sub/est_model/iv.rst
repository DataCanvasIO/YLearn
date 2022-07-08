**********************
Instrumental Variables
**********************


Instrumental Variables (IV) deal with the case for estimating causal effects in the presence of unobserved confounding variables that simultaneously
have effects on the treatment :math:`X` and the outcome :math:`Y`. A set of variables :math:`Z` is said to be a set of **instrumental variables**
if for any :math:`z` in :math:`Z`:
    
    1. :math:`z` has a causal effect on :math:`X`.

    2. The causal effect of :math:`z` on :math:`Y` is fully mediated by :math:`X`.

    3. There are no back-door paths from :math:`z` to :math:`Y`.

In such case, we must first find the IV (which can be done by using the :class:`CausalModel`, see :ref:`identification`). For an instance, the variable
:math:`Z` in the following figure can serve as a valid IV for estimating the causal effects of :math:`X` on :math:`Y` in the presence of the unobserved confounder
:math:`U`.

.. figure:: iv3.png

    Causal graph with IV

YLearn implements two different methods related to IV: deepiv [Hartford]_, which utilizes the deep learning models to IV, and IV of nonparametric models [Newey2002]_.

The IV Framework and Problem Setting
====================================
The IV framework aims to predict the value of the outcome :math:`y` when the treatment :math:`x` is given. Besides, there also exist some covariates vectors :math:`v` that
simultaneously affect both :math:`y` and :math:`x`. There also are some unobserved confounders :math:`e` that potentially also affect :math:`y`, :math:`x` and :math:`v`. The core part
of causal questions lies in estimating the causal quantity

.. math::

    \mathbb{E}[y| do(x)]

in the following causal graph, where the set of causal relationships are determined by the set of functions

.. math::

    y & = f(x, v) + e\\
    x & = h(v, z) + \eta\\
    \mathbb{E}[e] & = 0.

.. figure:: iv4.png

    Causal graph with IV and both observed and unobserved confounders

The IV framework solves this problem by doing a two-stage estimation:

    1. Estimate :math:`\hat{H}(z, v)` that captures the relationship between :math:`x` and the variables :math:`(z, v)`.

    2. Replace :math:`x` with the predicted result of :math:`\hat{H}(z, v)` given :math:`(v, z)`. Then estimate :math:`\hat{G}(x, v)` to build the relationship between :math:`y` and :math:`(x, v)`.

The final casual effects can then be calculated.

IV Classes
===========

.. toctree::
    :maxdepth: 2
    
    iv/iv_normal
    iv/deepiv