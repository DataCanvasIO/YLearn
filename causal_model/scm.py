import networkx as nx


class CausalStructuralModel:
    def __init__(self, causation):
        self.edges = self.build_edges(causation)
        pass

    def build_edges(self, causation):
        pass
