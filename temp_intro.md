# Definitions of concepts in YLearn

## Causal Model

1. *Causal Model*
    All causal models are represented by the class ***CausalModel*** in YLearn.
    A causal model is a triple
    $$
        M = \left< U, V, F\right>
    $$
    where
    - $U$ are **exogenous** (variables that are determined by factors outside the model);
    - $V$ are **endogenous** that are determined by $U \cup V$, and $F$ is a set of functions such that
        $$
            V_i = F_i(pa_i, U_i)
        $$
        with $pa_i \subset V \backslash V_i$.
    - Example: $M = \left< U, V, F\right>$ is a causal model where
        $$
        V = \{V_1, V_2\}, \\
        U = \{ U_1, U_2, I, J\}, \\
         F = \{F_1, F_2 \}
        $$
        such that
        $$
            V_1 = \theta_1 I + U_1\\
            V_2 = \phi V_1 + \theta_2 J + U_2.
        $$

2. *Causal Graph*

    A **causal graph** associated with a causal model $M$ is a directed graph $G(M)$ where each node corresponds to a variable and the directed edges point from members of $pa_i$ and $U_i$ toward $V_i$. All causal graphs can be represented by the class ***CausalGraph*** in YLearn.
    - Example:

         $X \longrightarrow V \longrightarrow W \longrightarrow Z$

         $X \longleftarrow V \longrightarrow W \longrightarrow Z$.

3. *Structural Equation Model*

   Each child-parent family in a DAG $G$ representas a deterministic function
   $$
        x_i = f_i(pa_i, \epsilon_i), i = 1, \dots, n,
   $$
    where $pa_i$ are the parents of $x_i$ in G and $\epsilon_i$ are random disturbance representing independent exogeneous not present in the analysis.

4. *Causal Effect*

   The **causal effect** of $X$ on $Y$ is denoted by $P(y|do(x))$ where $do(x)$ is called an intervention.

5. *Adjustment*

    If a set of variables $W$ satisfies the back-door criterion relative to $(X, Y)$, then the causal effect of $X$ on $Y$ is given by the formula
    $$
        P(y|do(x)) = \sum_w P(y| x, w)P(w).
    $$
    - Variables $X$ for which the above equality is valid are also named *"conditionally ignorable given $W$"* in the *potential outcome* framework.
    - The set of variables $W$ satisfying this condition is called **adjustment set**. These variables are named as **adjustment** and represented by the letter ***w*** in YLearn.
    - In the language of strucutral equation model, these relations are encoded by
    $$
        x = f_1 (w, \epsilon),\\
        y = f_2 (w, x, \eta).
    $$

## Estimator Model

1. *Average Treatment Effect (ATE)*

    In YLearn, we are interested in the difference
    $$
        E(Y|do(X=x_1)) - E(Y|do(X=X_0))
    $$
    which is also called **average treatment effect (ATE)**, where
    - $Y$ is called **outcome**
    - and $X$ is called **treatment**.

    When the conditional independence (conditional ignorability) holds given a set of variables $W$ potentially having effects on both outcome $Y$ and treatment $X$, then the ATE can be evaluated as
    $$
        E(Y|X=x_1, w) - E(Y|X=x_0, w).
    $$
    Using structural equation model we can describe the above relation as
    $$
        X = f_1 (W, \epsilon) \\
        Y = f_2 (X, W, \eta)\\
        \text{ATE} = E\left[ f_2(x_1, W, \eta) - f_2(x_0, W, \eta)\right].
    $$

2. *Conditional Average Treatment Effect (CATE) and Covariates*

    Suppose that we assign special roles to a subset of variables in the adjustment set $W$ and name them as **covariates** $V$, then, in the structural equation model, the **CATE** (also called **heterogeneous treatment effect**) is defined by
    $$
        X = f_1 (W, V, \epsilon) \\
        Y = f_2 (X, W, V, \eta)\\
        \text{CATE} = E\left[ f_2(x_1, W, V, \eta) - f_2(x_0, W, V, \eta)| V =v\right].
    $$

3. *Estimator Model.*

    The evaluations of $E(f_2(X, W, \eta))$ in ATE and $E(f_2(x_1, W, V, \eta))$ in CATE will be the tasks of machine learning in YLearn. The concept ***EstimatorModel*** in YLearn is designed for this purpose. A common usage should be composed of two steps:
    - Define and train an EstimatorModel:

        '''python

            model = EstimatorModel()

            model.fit(
                data,
                outcome,
                treatment,
                adjustment,
                covariate,
                instrument,
            )
        '''
    - Estimate the ATE, CATE, or other quantities in new dataset

        '''python

            model.estimate(test_data)
        '''

4. *Parameter names in YLearn.*

    Below we specify the corresponding parameter names of the aforementioned concepts in YLearn:
    | Concepts | Paramter names in dataset | Parameter names in code|
    |---- | ----| ---- |
    | Treatment | treatment | x |
    | Outcome | outcome | y |
    | Adjustment set |  adjustment | w |
    | Covariates | covariate | v |
    | Instruments | instrument | z |

## Causal Discovery

1.


## Combined treatment and separate treatment
