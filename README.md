
**YLearn**, a pun of "learn why", is a python package for causal inference which supports various aspects of causal inference ranging from causal effect identification, estimation, and causal graph discovery, etc.

Documentation website: <https://ylearn.readthedocs.io/en/latest/index.html>

## Installation

### Pip

Install YLearn with `pip`:

```bash
pip install ylearn
```

## Overview of YLearn

Machine learning has made great achievements in recent years.
The areas in which machine learning succeeds are mainly for prediction,
e.g., the classification of pictures of cats and dogs. However, machine learning is incapable of answering some
questions that naturally arise in many scenarios. One example is for the **counterfactual questions** in policy
evaluations: what would have happened if the policy had changed? Due to the fact that these counterfactuals can
not be observed, machine learning models, the prediction tools, can not be used. These incapabilities of machine
learning partly give rise to applications of causal inference in these days.

Causal inference directly models the outcome of interventions and formalizes the counterfactual reasoning.
With the aid of machine learning, causal inference can draw causal conclusions from observational data in
vairous manners nowdays, rather than relying on conducting craftly designed experiments.

A typical complete causal inference procedure is composed of three parts. First, it learns causal relationships
using the technique called causal discovery. These relationships are then expressed either in the form of Structural
Causal Models or Directed Acyclic Graphs (DAG). Second, it expresses the causal estimands, which are clarified by the
interested causal questions such as the average treatment effects, in terms of the observed data. This process is
known as identification. Finally, once the causal estimand is identified, causal inference proceeds to focus on
estimating the causal estimand from observational data. Then policy evaluation problems and counterfactual questions
can also be answered.

YLearn, equiped with many techniques developed in recent literatures, is implemented to support the whole causal inference pipeline from causal discovery to causal estimand estimation with the help of machine learning. This is more promising especially when there are abundant observational data.

### Concepts in YLearn

![Concepts in YLearn](./fig/structure_ylearn.png)

There are 5 main concepts in YLearn corresponding to the causal inference pipeline.

1. *Causal Discovery*. Discovering the causal relationships in the observational data.

2. *Causal Model*. Representing the causal relationships in the form of ``CausalGraph`` and doing other related operations such as identification with ``CausalModel``.

3. *Estimator Model*. Estimating the causal estimand with vairous techniques.

4. *Policy Model*. Selecting the best policy for each individual.

5. *Interpreters*. Explaining the causal effects and polices.

These components are conneted to give a full pipeline of causal inference, which are also encapsulated into a single API `Why`.

![A typical pipeline of YLearn](./fig/flow.png)
*The pipeline of causal inference in YLearn. Starting from the training data, one first uses the `CausalDiscovery` to reveal
the causal structures in data, which will usually output a `CausalGraph`. The causal graph is then passed into the `CausalModel`, where
the interested causal effects are identified and converted into statistical estimands. An `EstimatorModel` is then trained with the training data
to model relationships between causal effects and other variables, i.e., estimating causal effects in training data. One can then
use the trained `EstimatorModel` to predict causal effects in some new test dataset and evaluate the policy assigned to each individual or interpret
the estiamted causal effects.*

## Quick Start

In this part, we first show several simple example usages of YLearn. These examples cover the most common functionalities. Then we present a case stuy with :class:`Why` to unveil the hidden
causal relations in data.

### Example usages

We present several necessary example usages of YLearn in this section. Please see their specifical documentations to for more details.

1. Representation of causal graph
   
   For a given causal graph `X <- W -> Y`, the causal graph is represented by `CausalGraph`

    ```python
        causation = {'X': ['W'], 'W':[], 'Y':['W']}
        cg = CausalGraph(causation=causation)
    ```
   :py:attr:`cg` will be the causal graph represented in YLearn.

2. Identification of causal effect

   Suppose that we are interested in identifying the causal estimand `P(Y|do(X=x))` in the causal graph `cg`, then we should
   first define an instance of :class:`CausalModel` and call the :py:func:`identify()` method:

    ```python

        cm = CausalModel(causal_graph=cg)
        cm.identify(treatment={'X'}, outcome={'Y'}, identify_method=('backdoor', 'simple'))

    ```

3. Estimation of causal effect

   The estimation of causal effect with an `EstimatorModel` is composed of 4 steps:

    * Given data in the form of  `pandas.DataFrame`, find the names of `treatment, outcome, adjustment, covariate`.
    * Call  `fit()` method of  `EstimatorModel` to train the model.
    * Call  `estimate()` method of  `EstimatorModel` to estimate causal effects in test data.

### Case Study

## References

J. Pearl. Causality: models, reasoing, and inference.

S. Shpister and J. Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models. *AAAI 2006*.

B. Neal. Introduction to Causal Inference.

M. Funk, et al. Doubly Robust Estimation of Causal Effects. *Am J Epidemiol. 2011 Apr 1;173(7):761-7.*

V. Chernozhukov, et al. Double Machine Learning for Treatment and Causal Parameters. *arXiv:1608.00060.*

S. Athey and G. Imbens. Recursive Partitioning for Heterogeneous Causal Effects. *arXiv: 1504.01132.*

A. Schuler, et al. A comparison of methods for model selection when estimating individual treatment effects. *arXiv:1804.05146.*

X. Nie, et al. Quasi-Oracle estimation of heterogeneous treatment effects. *arXiv: 1712.04912.*

J. Hartford, et al. Deep IV: A Flexible Approach for Counterfactual Prediction. *ICML 2017.*

W. Newey and J. Powell. Instrumental Variable Estimation of Nonparametric Models. *Econometrica 71, no. 5 (2003): 1565–78.*

S. Kunzel2019, et al. Meta-Learners for Estimating Heterogeneous Treatment Effects using Machine Learning. *arXiv: 1706.03461.*

J. Angrist, et al. Identification of causal effects using instrumental variables. *Journal of the American Statistical Association*.

S. Athey and S. Wager. Policy Learning with Observational Data. *arXiv: 1702.02896.*

P. Spirtes, et al. Causation, Prediction, and Search.

X. Zheng, et al. DAGs with NO TEARS: Continuous Optimization for Structure Learning. *arXiv: 1803.01422.*
