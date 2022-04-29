

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
        self.causation.remove_incoming_edges(treatment)
        # TODO

    # def remove_incoming_edges(self, node):
    #     """remove the incoming edges to node

    #     Parameters
    #     ----------
    #     node : str
    #     """
    #     pass


class DoCalculus:
    # TODO: Rules of do-calculus are implemented as follows
    def rule_one(self):
        pass

    def rule_two(self):
        pass

    def rule_three(self):
        pass
