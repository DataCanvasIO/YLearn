import base64

import networkx as nx
import pandas as pd

from ylearn.causal_model import CausalGraph
from ylearn.utils import to_list, drop_none


class DiGraph(nx.DiGraph):
    def __init__(self, graph, acyclic=None):
        if isinstance(graph, CausalGraph):
            g = graph.dag
            graph_data = g.edges
        elif isinstance(graph, nx.Graph):
            g = graph
            graph_data = graph.edges(data=True)
        elif isinstance(graph, pd.DataFrame):
            g = nx.from_pandas_adjacency(graph, create_using=nx.DiGraph)
            graph_data = g
        else:
            g = nx.DiGraph(graph)
            graph_data = g

        if acyclic:
            if not nx.is_directed_acyclic_graph(g):
                raise ValueError('The given structure graph is not acyclic.')

            n_graph = nx.number_weakly_connected_components(g)
            if n_graph > 1:
                raise ValueError(f'The given structure graph include {n_graph} subgraph.')

        super().__init__(graph_data)

    @property
    def is_dag(self):
        return (nx.is_directed_acyclic_graph(self)
                and nx.number_weakly_connected_components(self) == 1
                )

    @property
    def topo_sorted_nodes(self):
        # result = []
        #
        # def push_node(node):
        #     if node not in result:
        #         for p in self.graph.predecessors(node):
        #             if p not in result:
        #                 push_node(p)
        #         result.append(node)
        #
        # nodes = self.nodes
        # for n in nodes:
        #     push_node(n)
        #
        # assert len(result) == len(nodes) and set(result) == set(nodes)
        #
        # return result
        if not self.is_dag:
            raise ValueError('"topo_sorted_nodes" only be supported by DAG.')

        return nx.topological_sort(self)

    def get_edges(self):
        return list(self.edges)

    def get_nodes(self, topo_sorted=False):
        return self.sort_nodes(self.nodes, topo_sorted=topo_sorted)

    def get_parents(self, node, topo_sorted=False):
        return self.sort_nodes(self.predecessors(node), topo_sorted=topo_sorted)

    def get_children(self, node, topo_sorted=False):
        return self.sort_nodes(self.successors(node), topo_sorted=topo_sorted)

    def get_upstream_blanket(self, nodes, return_self=True, topo_sorted=False):
        nodes = to_list(nodes, name='nodes')

        parents = []
        for node in nodes:
            parents.extend(self.get_parents(node))
        blanket = set(parents)

        if return_self:
            blanket.update(nodes)

        return self.sort_nodes(blanket, topo_sorted=topo_sorted)

    def get_downstream_blanket(self, nodes, return_self=True, topo_sorted=False):
        nodes = to_list(nodes, name='nodes')

        children = []
        for node in nodes:
            children.extend(self.get_children(node))
        blanket = set(children)

        if return_self:
            blanket.update(nodes)

        return self.sort_nodes(blanket, topo_sorted=topo_sorted)

    def get_markov_blanket(self, nodes, return_self=True, topo_sorted=False):
        """
        Returns a markov blanket for nodes.
        In Bayesian Networks, the markov blanket is the set of node's parents,
        its children and its children's other parents.
        """
        nodes = to_list(nodes, name='nodes')

        parents = []
        children = []
        for node in nodes:
            parents.extend(self.get_parents(node))
            children.extend(self.get_children(node))
        blanket = parents + children
        for child in set(children):
            blanket.extend(self.get_parents(child))
        blanket = set(blanket)

        if return_self:
            blanket.update(nodes)
        else:
            for node in nodes:
                blanket.discard(node)

        return self.sort_nodes(blanket, topo_sorted=topo_sorted)

    def get_blanket(self, nodes, kind='markov', return_self=True, topo_sorted=False):
        if kind == 'markov':
            blanket = self.get_markov_blanket(nodes, topo_sorted=topo_sorted)
        elif kind == 'downstream':
            blanket = self.get_downstream_blanket(nodes, topo_sorted=topo_sorted)
        else:
            blanket = self.get_upstream_blanket(nodes, topo_sorted=topo_sorted)

        if not return_self:
            for n in nodes:
                if n in blanket:
                    blanket.remove(n)
        return blanket

    def compress(self, keeps):
        """
        trim edges
        """
        raise NotImplementedError()

    def sort_nodes(self, nodes, topo_sorted=False):
        if topo_sorted:
            nodes = set(nodes)
            return [n for n in self.topo_sorted_nodes if n in nodes]
        else:
            return list(sorted(nodes))

    def to_pydot(self, dot_options=None, node_options=None, edge_options=None, node_pos=None):
        import pydot

        g = pydot.Subgraph('DAG', label="DAG")

        for node in self.get_nodes():
            node_attrs = self.nodes[node]
            assert isinstance(node_attrs, dict)

            if 'shape' in node_attrs.keys():
                shape = node_attrs['shape']
            else:
                shape = 'box'
            kwargs = {'shape': shape}
            if node_pos is not None and node in node_pos.keys():
                node_x, node_y = node_pos[node]
                kwargs['pos'] = f'{node_x / 72:.2f},{node_y / 72:.2f}!'
            if node_options is not None:
                kwargs.update(node_options)
            g.add_node(pydot.Node(node, lable=node, **kwargs))

        for s, d in self.get_edges():
            kwargs = {}
            if edge_options is not None:
                kwargs.update(edge_options)
            g.add_edge(pydot.Edge(s, d, **kwargs))

        kwargs = {'strict': True}
        if dot_options is not None:
            kwargs.update(dot_options)
        D = pydot.Dot("main", graph_type="digraph", **kwargs)
        D.add_subgraph(g)
        return D

    def pydot_layout(self, prog="neato", node_pos=None, dot_options=None):
        import pydot
        from locale import getpreferredencoding

        P = self.to_pydot(dot_options=dot_options, node_pos=node_pos)
        D_bytes = P.create(format='dot', prog=prog)
        assert len(D_bytes) > 0, f"Graphviz layout with {prog} failed."

        D = str(D_bytes, encoding=getpreferredencoding())
        Q_list = pydot.graph_from_dot_data(D)
        assert len(Q_list) == 1

        Q = Q_list[0].get_subgraph('DAG')[0]
        node_layout = {}
        for n in self.nodes():
            node_info = {}
            str_n = str(n)
            pydot_node = pydot.Node(str_n).get_name()
            node = Q.get_node(pydot_node)

            if isinstance(node, list):
                node = node[0]
            pos = node.get_pos()
            if pos is not None:
                xx, yy = pos.strip('"').split(",")
                node_info['x'] = float(xx)
                node_info['y'] = float(yy)
            width = node.get_width()
            if width is not None:
                node_info['width'] = float(width) * 72
            height = node.get_height()
            if height is not None:
                node_info['height'] = float(height) * 72
            shape = node.get_shape()
            if shape is not None:
                node_info['shape'] = shape
            node_layout[n] = node_info

        edge_layout = {}
        for es, ee in self.edges():
            edge = Q.get_edge(str(es), str(ee))
            if isinstance(edge, (list, tuple)):
                edge = edge[0]
            pos = edge.get_pos().replace('\\\n', '').strip('"')
            xs = []
            ys = []
            for xy in pos.strip('e,').split(' '):
                x, y = xy.split(',')
                xs.append(float(x))
                ys.append(float(y))

            # see: https://graphviz.org/docs/attr-types/splineType/
            if pos.startswith('e'):
                xs = xs[1:] + xs[:1]
                ys = ys[1:] + ys[:1]
            elif pos.startswith('s'):
                xs = xs[:1] + xs[2:] + xs[1:2]
                ys = ys[:1] + ys[2:] + ys[1:2]
            edge_layout[(es, ee)] = dict(x=xs, y=ys)

        return node_layout, edge_layout

    def plot(self, prog='dot', fmt='svg', width=None, height=None, **kwargs):
        try:
            from IPython.display import Image, display, SVG, HTML
            P = self.to_pydot()
            img = P.create(format=fmt, prog=prog)
            if fmt == 'svg':
                img = SVG(img)
                if width is not None or height is not None:
                    img_b64 = base64.b64encode(img.data.encode('utf-8')).decode('utf-8')
                    img_html = '<img'
                    if width is not None:
                        img_html += f' width="{width}"'
                    if height is not None:
                        img_html += f' height="{height}"'
                    img_html += f' src="data:image/svg+xml;base64, {img_b64}" >'
                    img = HTML(img_html)
            else:
                img = Image(img, **drop_none(width=width, height=height), **kwargs)
            display(img)
        except Exception as e:
            import warnings
            warnings.warn(f'Failed to display pydot image: {e}.')


class DAG(DiGraph):
    def __init__(self, graph):
        super().__init__(graph, acyclic=True)
        nx.freeze(self)
