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
                 variables=set(),
                 conditional=set(),
                 marginal=set(),
                 product=set()):
        """Represent the probability distribution P(variable|conditional).

        Parameters
        ----------
        variables : set
        conditional : set
        marginal : set
            elements are strings, summing over these elements will return the
            marginal distribution
        product : set
            set of Prob
        """
        self.variables = variables
        self.conditional = conditional
        self.marginal = marginal
        self.product = product

    def parse(self):
        """Return the expression of the probability distribution.

        Returns
        ----------
        expression : str
        """
        # TODO
        expression = 'Not implemented yet.'
        return expression