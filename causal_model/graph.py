import networkx as nx


class CausalGraph:

    def __init__(self, causation):
        self.causation = causation
        self.edges = []
        for k, v in causation.items():
            # self.order_list.append(k)
            for para in v:
                self.edges.append((para, k))
        self.DG = nx.DiGraph()
        self.DG.add_edges_from(self.edges)

    def is_dag(self):
        # TODO: determin if the graph is a DAG, can use tr(e^{W\circledot W}-d)=0
        return nx.is_directed_acyclic_graph(self.DG)

    def add_edges(self, edge_list):
        for i in edge_list:
            self.edges.append(i)
        self.DG.add_edges_from(edge_list)

    def to_adj_matrix(self):
        W = nx.to_numpy_matrix(self.DG)
        return W

    def to_adj_list(self):
        pass
