from copy import deepcopy


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

    def __init__(self, variable, conditional):
        """Represent the probability distribution P(variable|conditional).

        Parameters
        ----------
        variable : str, optional
        conditional : str, optional
        """
        self.variable = variable
        self.conditional = conditional

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

    def __init__(self, prob_list) -> None:
        """
        Parameters
        ----------
        prob_list : list
            each element is a Prob object representing a probability
            distribution
        """
        pass


class Do:
    """Implementation of the do-operator.

    Attributes
    ----------
    causation : CausalGraph or CausalStructureModel
    treatment : str

    Methods
    ----------
    """

    def __init__(self, causation):
        """
        Parameters
        ----------
        causation : CausalGraph or CausalStructureModel
            the causation structures in which the do-operator will be applied
        """
        self.causation = causation

    def do(self, treatment, value):
        """Do the treatment=value in the graph or structural models.

        Parameters
        ----------
        treatment : str
        value : float, optional
            the value of the treatment
        """
        self.clear_incoming_edges(treatment)
        # TODO

    def clear_incoming_edges(self, node):
        """Clear the incoming edges to node

        Parameters
        ----------
        node : str
        """
        pass


class DoCalculus:
    # TODO: Rules of do-calculus are implemented as follows
    def rule_one(self):
        pass

    def rule_two(self):
        pass

    def rule_three(self):
        pass
