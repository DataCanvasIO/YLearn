**********************
Instrumental Variables
**********************


Instrumental Variables (IV) deal with the case for estimating causal effects in the presense of unobserved confounding, variables that simultaneously
have effects on the treatment :math:`X` and the outcome :math:`Y`. A set of variables :math:`Z` is said to be a set of **instrumental variables**
if for any :math:`z` in :math:`Z`:
    
    1. :math:`z` has a causal effect on :math:`X`.

    2. The causal effect of :math:`z` on :math:`Y` is fully mediated by :math:`X`.

    3. There are no back-door paths from :math:`z` to :math:`Y`.

In such case, we must first find the IV (which can be done by using the :class:`CausalModel`, see :ref:`identification`). For an instance, the variable
:math:`Z` in the following figure can serve as a valid IV for estimating the causal effects of :math:`X` on :math:`Y` in the presense of the unobserved confounder
:math:`U`.

.. figure:: iv3.png

    Causal graph with IV

YLearn implements two different methods related to IV: deepiv [Hartford]_, which utilizes the deep learning models to IV, and IV of nonparametric models [Newey2002]_.

.. toctree::
    
    iv/iv_normal
    iv/deepiv