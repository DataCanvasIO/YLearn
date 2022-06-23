Causal Discovery: Exploring the Causal Structures in Data
=========================================================

.. toctree::
    discovery/notears

A fundamental task in causal learning is to find the underlying causal relationships, the so-called "causal structures", and apply
them. Traditionally, these relationships might be revealed by designing randomized experiments or imposing interventions. However,
such methods are too expansive or even impossible. Therefore, many techniques, e.g., the PC algorithm (see [Spirtes2001]_), have been suggested recently to analyze the causal
structures by directly utilizing observational data. These techniques are named as *causal discovery*.

The current version of YLearn implement a score-based method for causal discovery [Zheng2018]_. More methods will be added in later versions.

