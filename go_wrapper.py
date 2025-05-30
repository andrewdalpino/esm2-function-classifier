"""
A class that wraps the the GO ontology that constructs a graph from the GO OBO file.
It provides methods to construct the GO graph, compute various GO-related metrics,
propagate terms, and plot GO subgraphs in various ways.
"""

from pathlib import Path
from typing import Optional, Union, Tuple, Container, List, Callable

import networkx as nx
from networkx.drawing import layout
from matplotlib import pyplot as plt
from goatools.obo_parser import GODag


MF_ROOT_ID = 'GO:0003674'
CC_ROOT_ID = 'GO:0005575'
BP_ROOT_ID = 'GO:0008150'


class GoWrapper:
    """
    A wrapper for the the GO ontology that constructs a graph from the GO OBO file.
    It provides methods to construct the GO graph, compute various GO-related metrics,
    propagate terms, and plot GO subgraphs in various ways.

    Args:
        go_path: The path to the GO OBO file.
        root_ids: The root IDs of the GO graph.
        discard_go_dag: Whether to discard the GOATools GODag object after construction.
            If True, the GODag object will not be available after construction.
            If False, the GODag object will be available as self.go_dag.
    """
    def __init__(self, go_fpath: Union[Path, str],
                 root_ids: Optional[Tuple[str, str, str]] = (MF_ROOT_ID, CC_ROOT_ID, BP_ROOT_ID),
                 discard_go_dag: bool = False):
        self.go_fpath = go_fpath
        go_dag = GODag(go_fpath)
        self.ids_to_terms = {go_id: term_node.name for go_id, term_node in go_dag.items()}
        self.root_ids = root_ids
        self.go_graph = self.construct_graph_from_godag(go_dag)
        if not discard_go_dag:
            self.go_dag = go_dag

    def construct_graph_from_godag(self,
        godag: GODag,
        max_depth: Optional[int] = None,
        direction: str = 'both'
    ) -> nx.DiGraph:
        """
        Construct a NetworkX directed graph from a GOATools GODag object.
        
        Args:
            godag: A GOATools GODag object containing the Gene Ontology hierarchy
            root_node: Optional root node ID to start subgraph construction from
            max_depth: Optional maximum depth to traverse from root node
            direction: Direction to traverse from root ('up', 'down', or 'both')
            
        Returns:
            A NetworkX directed graph with both 'PARENT' and 'CHILD' edges
        """
        G = nx.DiGraph()
        queue = []
        visited = set()

        def add_node_and_edges(go_id: str, current_depth: int = 0):
            """Helper function to add nodes and edges"""
            if go_id not in godag or go_id in visited:
                return

            term = godag[go_id]
            # print(term.id, term.name)
            visited.add(go_id)

            # Add the node if not already present
            # print('Checking if node exists', go_id, go_id in G)
            if go_id not in G:
                # print('Adding node', go_id)
                G.add_node(go_id,
                        name=term.name,
                        namespace=term.namespace,
                        level=term.level,
                        depth=term.depth)

            # If we've reached max depth, stop traversing
            if max_depth is not None and current_depth >= max_depth:
                return

            # Add parent relationships (traverse up)
            if direction in ['up', 'both']:
                for parent in term.parents:
                    if parent.id not in visited:
                        queue.append((parent.id, current_depth + 1))
                        G.add_node(go_id,
                                name=parent.name,
                                namespace=parent.namespace,
                                level=parent.level,
                                depth=parent.depth)
                    # print('Adding edge', go_id, parent.id)
                    G.add_edge(go_id, parent.id, type='PARENT')
                    G.add_edge(parent.id, go_id, type='CHILD')

            # Add child relationships (traverse down)
            if direction in ['down', 'both']:
                for child in term.children:
                    if child.id not in visited:
                        queue.append((child.id, current_depth + 1))
                        G.add_node(child.id,
                                name=child.name,
                                namespace=child.namespace,
                                level=child.level,
                                depth=child.depth)
                    G.add_edge(go_id, child.id, type='CHILD')
                    G.add_edge(child.id, go_id, type='PARENT')

        for go_id in self.root_ids:
            queue.append((go_id, 0))

        while queue:
            go_id, current_depth = queue.pop(0)
            add_node_and_edges(go_id, current_depth)

        return G

    def compute_excess_components(self,
                                  predicted_terms: Container[str],
                                  expected_num_components: Union[int, None]=None) -> int:
        """
        Computes the number of excess disconnected components in the predicted subgraph.

        Args:
            predicted_terms: A container of GO IDs representing the predicted terms.
            expected_num_components: The expected number of components. If None the expected number of components 
                is the number of root IDs used to construct the graph.
        
        Returns:
            The number of excess disconnected components in the predicted subgraph.
        """
        if expected_num_components is None:
            expected_num_components = len(self.root_ids)
        predicted_subgraph = self.go_graph.subgraph(predicted_terms)
        count_weak = nx.number_weakly_connected_components(predicted_subgraph)
        return max(0, count_weak - expected_num_components)

    def compute_excess_components_per_term(self,
                                           predicted_terms: Container[str],
                                           expected_num_components: Union[int, None]=None) -> float:
        """
        Computes the number of excess disconnected components in the predicted subgraph normalized by the number of terms.

        Args:
            predicted_terms: A container of GO IDs representing the predicted terms.
            expected_num_components: The expected number of components. If None the expected number of components 
                is the number of root IDs used to construct the graph.

        Returns:
            The number of excess disconnected components in the predicted subgraph normalized by the number of terms.
        """
        return self.compute_excess_components(predicted_terms=predicted_terms,
                                            expected_num_components=expected_num_components) / len(predicted_terms)

    @staticmethod
    def compute_depths(G: nx.DiGraph, root_id: str) -> List[int]:
        """
        Returns a list of depth values for each node in the graph.
        The depth of a node is the number of edges from the root to the node as found by breadth-first search 
            (this may be different from the "depth" attribute of the nodes).
        The root is the node with the given root_id.

        Args:
            G: A NetworkX directed graph.
            root_id: The ID of the root node to start the breadth-first search from.

        Returns:
            A list of depth values for each node in the graph.
        """
        depths = []
        queue = [(root_id, 0)]
        while queue:
            node_id, depth = queue.pop(0)
            depths.append(depth)
            for _, dest, data in G.edges(node_id, data=True):
                if data['type'] == 'CHILD':
                    queue.append((dest, depth + 1))
        return depths

    def propagate_terms(self, source_terms: Container[str], direction: str='up') -> List[str]:
        """
        Returns a list of node ids representing the subgraph induced by following edges from the source terms.
        If the direction is 'up', only 'PARENT' edges are followed.
        If the direction is 'down', only 'CHILD' edges are followed.
        If the direction is 'both', both 'PARENT' and 'CHILD' edges are followed.

        Args:
            source_terms: A container of GO IDs representing the source terms.
            direction: The direction to propagate the terms. Allowed values are "up" and "down" and "both"

        Returns:
            A list of GO IDs representing the propagated terms.
        """
        if direction == 'up':
            edge_types = ('PARENT',)
        elif direction == 'down':
            edge_types = ('CHILD',)
        elif direction == 'both':
            edge_types = ('PARENT', 'CHILD')
        else:
            raise ValueError(f'Invalid direction: {direction}. Allowed directions are "up", "down", and "both".')

        propagated_nodes = []

        for term in source_terms:
            parent_queue = [term]

            while parent_queue:
                this_node = parent_queue.pop()
                propagated_nodes.append(this_node)
                for dest_node, edge_data in self.go_graph[this_node].items():
                    if edge_data['type'] in edge_types:
                        parent_queue.append(dest_node)
        return propagated_nodes

    @staticmethod
    def plot_go_subgraph(G: nx.DiGraph,
                           ax: Optional[plt.Axes]=None,
                           label_by_name: bool=False,
                           layout_func: Callable=layout.spring_layout,
                           fig_width: float=12,
                           node_size: float=50,
                           edge_width: float=0.5,
                           node_color: str='lightblue',
                           edge_color: str='gray',
                           label_size: float=6) -> Tuple[plt.Figure, plt.Axes]:
        '''
        Plots a subgraph of G containing only the nodes in node_list and only 'CHILD' edges.
        '''
        pos = layout_func(G)
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(fig_width, fig_width), dpi=300)
        else:
            fig = ax.get_figure()

        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            ax=ax,
            node_color=node_color,
            node_size=node_size
        )
        # Filter edges to only those labeled as 'CHILD'
        child_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get('type') == 'CHILD'
        ]
        nx.draw_networkx_edges(
            G=G,
            pos=pos,
            edgelist=child_edges,
            ax=ax,
            edge_color=edge_color,
            arrows=True,
            width=edge_width
        )
        if label_by_name:
            node_labels = {node: G.nodes[node]['name'] for node in G.nodes()}
        else:
            node_labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(
            G=G,
            pos=pos,
            ax=ax,
            labels=node_labels,
            font_size=label_size
        )
        ax.set_axis_off()
        return fig, ax

    def plot_go_subgraphs_true_vs_predicted(self,
                                            true_terms: Container[str],
                                            predicted_terms: Container[str],
                                            ax: Optional[plt.Axes]=None,
                                            label_by_name: bool=False,
                                            layout_func: Callable=layout.spring_layout,
                                            fig_width: float=8,
                                            node_size: float=50,
                                            edge_width: float=0.5,
                                            label_size: float=6,
                                            tp_color: str='green',
                                            fp_color: str='red',
                                            fn_color: str='blue',
                                            edge_color: str='gray') -> Tuple[plt.Figure, plt.Axes]:
        '''
        Plots a subgraph of the GO graph containing the nodes representing the true and predicted terms.
        The nodes are colored green, red, and blue (default colors) for true positives, false positives,
            and false negatives, respectively.
        The edges are colored gray (default). Only the 'CHILD' edges are shown.
        The nodes are labeled by their GO IDs, or optionally by the term names.
        
        Args:
            true_terms: A container of GO IDs representing the true terms.
            predicted_terms: A container of GO IDs representing the predicted terms.
            ax: An optional matplotlib Axes object to plot the graph on.
            label_by_name: Whether to label the nodes by the term names instead of their GO IDs.
            layout_func: A function to layout the graph.
            fig_width: The width of the figure.
            node_size: The size of the nodes.
            edge_width: The width of the edges.
            label_size: The size of the labels.
            
        '''
        true_terms = set(true_terms)
        predicted_terms = set(predicted_terms)
        tp_terms = true_terms.intersection(predicted_terms)
        fp_terms = predicted_terms.difference(tp_terms)
        fn_terms = true_terms.difference(tp_terms)
        joint_terms = true_terms.union(predicted_terms)

        joint_network = self.go_graph.subgraph(joint_terms)

        pos = layout_func(joint_network)

        if ax is None:
            fig, ax = plt.subplots(1, figsize=(fig_width, fig_width), dpi=300)
        else:
            fig = ax.get_figure()

        nx.draw_networkx_nodes(
            G=self.go_graph.subgraph(tp_terms),
            pos=pos,
            ax=ax,
            node_color=tp_color,
            node_size=node_size
        )

        nx.draw_networkx_nodes(
            G=self.go_graph.subgraph(fp_terms),
            pos=pos,
            ax=ax,
            node_color=fp_color,
            node_size=node_size
        )

        nx.draw_networkx_nodes(
            G=self.go_graph.subgraph(fn_terms),
            pos=pos,
            ax=ax,
            node_color=fn_color,
            node_size=node_size
        )

        # Filter edges to only those labeled as 'CHILD'
        child_edges = [
            (u, v) for u, v, d in joint_network.edges(data=True)
            if d.get('type') == 'CHILD'
        ]
        nx.draw_networkx_edges(
            G=joint_network,
            pos=pos,
            edgelist=child_edges,
            ax=ax,
            edge_color=edge_color,
            arrows=True,
            width=edge_width
        )
        if label_by_name:
            node_labels = {node: joint_network.nodes[node]['name'] for node in joint_network.nodes()}
        else:
            node_labels = {node: node for node in joint_network.nodes()}
        nx.draw_networkx_labels(
            G=joint_network,
            pos=pos,
            ax=ax,
            labels=node_labels,
            font_size=label_size
        )
        ax.set_axis_off()
        return fig, ax
