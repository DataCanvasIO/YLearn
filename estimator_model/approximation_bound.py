import numpy as np

from copy import deepcopy


class CausalEffectBound:
    def __init__(
        self,
        y_model,
        data,
        outcome,
        treatment,
        condition_set=None,
        x_prob=None,
        x_model=None
    ):
        """
        Parameters
        ----------
        y_model : MLModel, optional.
            Machine learning models for fitting the relation between treatment
            and outcome.
        data : DataFrame
        outcome : str
        treatment : str
        condition_set : list of str, optional. Defaults to None.
            Specify this for estiamte CATE.
        x_prob : list of float, optional. Defaults to None.
            The probability of taking a specific treatment.
        x_model : MLModel, optional. Defaults to None.
            Machine learning models for fitting the relation between treatment
            and condition set if condition set is True.
        """
        self.y_model = y_model
        self.x = data[treatment]
        self.y = data[outcome]
        self.condition_set = data[condition_set] if condition_set else None
        # self.x_model = x_model
        self.x_prob = x_prob

    def fit(self):
        """Fit all relevant models and calculate necessary quantites.
        """
        if self.condition_set:
            pass
        else:
            if not self.x_prob:
                self.x_prob = (
                    self.x.value_counts().sort_index() / self.x.shape[0]
                ).values

            self.y_model.fit(
                self.x.values.reshape(-1, 1), self.y.values.reshape(-1, 1)
            )

    def effect_bound(
        self,
        y_upper=None,
        y_lower=None,
        treatment_value=None,
        assump='non-negative'
    ):
        """Calculate the approximation bound of causal effects.

        Parameters
        ----------
        y_upper : float.
            The upper bound of the outcome
        y_lower : float.
            The lower bound of the outocme.
        treatment_value : float, optional.
            Specify the which treatment group is selected. Defaults to None.
        assump : str, optional.  Defaults to 'no-assump'.
            Should be in one of
                1. no-assump: calculate the no assumption bound whose result
                    will always contain 0.
                2. non-negative: The treatment is always positive.
                3. non-positive: The treatment is always negative.
                4. optimal: The treatment is taken if its effect is positive.

        Raises
        ----------
        Exception

        Returns
        ----------
        tuple of float
            The first element is the lower bound while the second element is
            the upper bound.
        """
        if not treatment_value:
            treatment_value = 1
        if y_upper is None:
            y_upper = self.y.max()
        if y_lower is None:
            y_lower = self.y.min()

        xt = treatment_value * np.array([[1]])
        x0 = np.array([[0]])
        xt_prob = self.x_prob[treatment_value]
        x0_prob = self.x_prob[0]
        yt, y0 = self.y_model.predict(xt), self.y_model.predict(x0)

        upper = xt_prob * yt + (1 - xt_prob) * y_upper \
            - (1 - x0_prob) * y_lower - x0_prob * y0
        lower = xt_prob * yt + (1 - xt_prob) * y_lower \
            - (1 - x0_prob) * y_upper - x0_prob * y0

        if assump == 'no-assump':
            return (lower, upper)
        elif assump == 'non-negative':
            return (0, upper)
        elif assump == 'non-positive':
            return (lower, 0)
        elif assump == 'optimal':
            # TODO: need to update this for discrete treatment
            optimal_upper1 = xt_prob * yt - (1 - x0_prob) * y_lower \
                + (1 - xt_prob - x0_prob) * y_upper
            optimal_lower1 = x0_prob * lower - x0_prob * y0
            optimal_upper2 = yt - xt_prob * lower - x0_prob * y0
            optimal_lower2 = xt_prob * yt + x0_prob * lower - y0
            return (
                (optimal_lower1, optimal_upper1),
                (optimal_lower2, optimal_upper2)
            )
        else:
            raise Exception(
                'Only support assumptions in no-assump, non-negative, and'
                'non-positive'
            )
