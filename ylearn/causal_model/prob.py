from IPython.display import Latex


class Prob:
    r"""
    Probability distribution, e.g., the probability expression
    \sum_{w}P(v|y)[P(w|z)P(x|y)P(u)]. We will clarify below the meanings
    of our variables.


    Attributes
    ----------
    variables : set
        The variables (v in the above example) of the probability.
    
    conditional : set
        The conditional set (y in the above example).
    
    divisor : set
        Not defined yet.
    
    marginal : set
        The sum set (w in the above example) for marginalizing the probability.
    
    product : set
        If not set(), then the probability is composed of the first probability
        object (P(v|y)) and several other probabiity objects that are all saved
        in the set product, e.g., product = {P1, P2, P3} where P1 for P(w|z),
        P2 for P(x|y), and P3 for P(u) in the above example.

    Methods
    ----------
    parse()
        Return the expression of the probability distribution.
    
    show_latex_expression()
        Show the latex expression.
    """

    def __init__(self,
                 variables=set(),
                 conditional=set(),
                 divisor=set(),
                 marginal=set(),
                 product=set()):
        """
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
        self.divisor = divisor
        self.variables = variables
        self.conditional = conditional
        self.marginal = marginal
        self.product = product

    def parse(self):
        """
        Return the expression of the probability distribution.

        Returns
        ----------
        expression : str
        """
        # TODO
        expression = ''
        # First find the marginal set, -> \sum_{marginal set}
        if self.marginal:
            mar = ', '
            mar = mar.join(self.marginal)
            expression = expression + '\\sum_{' + f'{mar}' + '}'

        # Find the variables, -> \sum_{marginal} P(variables|conditional)
        if self.variables:
            var = ', '
            var = var.join(self.variables)
            if self.conditional:
                cond = ', '
                cond = cond.join(self.conditional)
                expression = expression + f'P({var}|{cond})'
            else:
                expression += f'P({var})'

        # Find the products, ->
        # \sum_{marginal} P(var|cond) \sum_{prod1.mar}P(prod1.var|prod1.cond)..
        if self.product:
            for p in self.product:
                expression = expression + '\\left[' + p.parse() + '\\right]'

        return expression

    def show_latex_expression(self):
        """
        Show the latex expression.
        """
        return Latex(f'${self.parse()}$')

    # def __repr__(self) -> str:
    #     return 'Prob'