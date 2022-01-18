class Prob:
    """Probability distribution.

    Attributes
    ----------
    variable : str
    conditional : str

    Methods
    ----------
    parse()
        Return the expression of the probability distribution.
    """

    def __init__(self, variable, conditional, marginal=None):
        """Represent the probability distribution P(variable|conditional).

        Parameters
        ----------
        variable : str, optional
        conditional : str, optional
        marginal : list
            elements are strings, summing over these elements will return the
            marginal distribution
        """
        self.variable = variable
        self.conditional = conditional
        self.marginal = marginal

    def parse(self):
        """Return the expression of the probability distribution.

        Returns
        ----------
        expression : str
        """
        # TODO
        expression = None
        return expression


class ProbProduct:
    """Product of several single probability distribution.

    Attributes
    ----------

    Methods
    ----------
    """

    def __init__(self, prob_list):
        """
        Parameters
        ----------
        prob_list : list
            each element is a Prob object representing a probability
            distribution
        """
        pass