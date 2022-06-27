***************************************
Why: An All-in-One Causal Learning API
***************************************

Want to use YLearn in a much eaiser way? Try the all-in-one API `Why`!

`Why` is an API which encapsulates almost everything in YLearn, such as *identifying causal effects* and *scoring a trained estimator model*. It provides to users a simple
and efficient way to use our package: one can directly pass the only thing you have, the data, into
`Why` and call various methods of it rather than learning multiple concepts such as adjustment set before being able to find interesting information hidden in your data. `Why`
is designed to enable the full-pipeline of causal inference: given data, it first tries to discover the causal graph
if not provided, then it attempts to find possible variables as treatments and identify the causal effects, after which
a suitable estimator model will be trained to estimate the causal effects, and, finally, the policy is evaluated to suggest the best option
for each individual.

.. figure:: flow.png

    `Why` can help almost every part of the whole pipeline of causal inference.

Class Structures
================

.. autoclass:: ylearn._why.Why