class Prob:
    """Probability distribution.

    Attributes
    ----------
    variables : str
    conditional : str

    Methods
    ----------
    parse()
        Return the expression of the probability distribution.
    """

    def __init__(self,
                 variables=None,
                 conditional=None,
                 marginal=None,
                 divisor=None,
                 product=None):
        """Represent the probability distribution P(variable|conditional).

        Parameters
        ----------
        variables : set
        conditional : set
        marginal : set
            elements are strings, summing over these elements will return the
            marginal distribution
        divisor : set
        product : set
        """
        self.variables = variables
        self.conditional = conditional
        self.marginal = marginal
        self.divisor = divisor
        self.product = product

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
