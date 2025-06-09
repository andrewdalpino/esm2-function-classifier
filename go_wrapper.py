"""
A class that wraps the the GO ontology that constructs a graph from the GO OBO file.
It provides methods to construct the GO graph, compute various GO-related metrics,
propagate terms, and plot GO subgraphs in various ways.
"""
import json
from pathlib import Path
from typing import Optional, Union, Tuple, Container, List, Callable, Iterable

import networkx as nx
from networkx.drawing import layout
from matplotlib import pyplot as plt


GO_BASIC_MULTIVALUED_KEYS = ('is_a', 'relationship', 'synonym', 'xref', 'alt_id', 'subset', 'consider')
GO_BASIC_NAMESPACES = ('cellular_component', 'biological_process', 'molecular_function')
MF_ROOT_ID = 'GO:0003674' # Molecular Function
CC_ROOT_ID = 'GO:0005575' # Cellular Component
BP_ROOT_ID = 'GO:0008150' # Biological Process

def parse_obo_content(obo_fpath: Union[Path, str], multivalued_keys: Container[str]=GO_BASIC_MULTIVALUED_KEYS) -> dict:
    """
    Parse an OBO file and return a dictionary of header keys and values, and a dictionary of atom types and their values.
    Currently defined "atoms" in the go-basic.obo file are: "Term" and "Typedef". The Typedef entries define relationship types between Terms
    """
    def parse_kv(line: str) -> tuple[str, str]:
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        return key, value
    
    multivalued_keys = set(multivalued_keys)

    header = {}
    content = {}
    current_atom = {}
    current_atom_type = None
    print('Parsing OBO file', obo_fpath)
    with open(obo_fpath, 'r') as obo_file:
        for line_num, line in enumerate(obo_file):
            line = line.strip()
            # print(line_num, line)
            if not line:
                continue
            if line[0] == '[' and line[-1] == ']':
                if current_atom:
                    # print(f"Adding new atom of type: {current_atom_type}, {current_atom}")
                    content[current_atom_type].append(current_atom)
                    current_atom = {}
                current_atom_type = line[1:-1]
                if current_atom_type not in content:
                    # print(f"Adding new atom type: {current_atom_type}")
                    content[current_atom_type] = []

                continue
            try:
                key, value = parse_kv(line)
            except ValueError as e:
                print(f"Error parsing line {line_num}: {line} with error {e}")
                continue
            if current_atom_type is None:
                # print(f"Adding header: {key}, {value}")
                header[key] = value
            else:
                if key in multivalued_keys:
                    if key not in current_atom:
                        current_atom[key] = []
                    current_atom[key].append(value)
                else:
                    current_atom[key] = value
                
    if current_atom:
        # print(f"Adding new atom of type: {current_atom_type}, {current_atom}")
        content[current_atom_type].append(current_atom)

    return header, content


def construct_nx_graph_from_obo_content(obo_terms: Iterable[dict], 
                                        namespaces_to_include: Union[Container[str], None]=GO_BASIC_NAMESPACES,
                                        relationship_types_to_include: Union[Container[str], None]=None,
                                        exclude_obsolete: bool=True) -> nx.DiGraph:
    """
    Construct a networkx graph from the OBO content.
    """
    def parse_is_a(is_a: str) -> tuple[str, str]:
        parent_id, parent_name = is_a.split(' ! ')
        return parent_id, parent_name
    
    def parse_relationship(relationship: str) -> tuple[str, str]:
        relationship_type, target = relationship.split(' ', 1)
        target_id, target_name = target.split(' ! ')
        return relationship_type, target_id, target_name
    
    if relationship_types_to_include is None:
        valid_relationship_types = None
        print('Adding all relationship types')
    else:
        valid_relationship_types = set(relationship_types_to_include)
        print('Adding relationship types', valid_relationship_types)

    namespaces_to_include = set(namespaces_to_include)
    print('Namespaces to include', namespaces_to_include)

    graph = nx.DiGraph()
    # Add all non-obsolete terms to the graph
    for term in obo_terms:
        if exclude_obsolete and 'is_obsolete' in term and term['is_obsolete'] == 'true':
            continue
        if namespaces_to_include is not None and term['namespace'] not in namespaces_to_include:
            # print('Skipping term', term['id'], ' with namespace', term['namespace'], 'because it is not in the allowed namespaces', namespaces_to_include)
            continue
        # print('Adding term', term['id'])
        graph.add_node(term['id'], **term)

        # Add the "is_a" relationships in both parent and child directions
        if 'is_a' in term:
            for is_a in term['is_a']:
                parent_id, _ = parse_is_a(is_a)
                graph.add_edge(parent_id, term['id'], relationship_type='CHILD')
                graph.add_edge(term['id'], parent_id, relationship_type='PARENT')

        # Add the other relationships
        if 'relationship' in term:
            for relationship in term['relationship']:
                try:
                    relationship_type, target_id, _ = parse_relationship(relationship)
                except ValueError:
                    print('Error parsing relationship', relationship)
                    continue
                if valid_relationship_types is not None and relationship_type in valid_relationship_types:
                    graph.add_edge(term['id'], target_id, relationship_type=relationship_type)
    
    return graph


class GOGraph:
    """
    A wrapper for the the GO ontology that constructs a graph from the GO OBO file.
    It provides methods to construct the GO graph, compute various GO-related metrics,
    propagate terms, and plot GO subgraphs in various ways.

    Args:
        go_fpath: The path to the GO OBO file.
        relationship_types_to_include: The relationship types to include in the graph. If None (default), includes all relationship types.
        namespaces_to_include: The namespaces to include in the graph. Default is all namespaces ('cellular_component', 'biological_process', 'molecular_function').
        exclude_obsolete: Whether to exclude obsolete terms from the graph. Default is True.
    """
    def __init__(self, go_fpath: Union[Path, str], namespaces_to_include: Union[Container[str], None]=GO_BASIC_NAMESPACES,
                 relationship_types_to_include: Union[Container[str], None]=None, 
                 exclude_obsolete: bool=True):
        self.go_fpath = go_fpath
        self.exclude_obsolete = exclude_obsolete

        self.header, obo_content = parse_obo_content(go_fpath)
        print('Header', json.dumps(self.header, indent=2))

        if relationship_types_to_include:
            self.relationship_types = set(relationship_types_to_include)
        else:
            # Get the relationship types from the typedefs in the OBO file
            self.relationship_types = set().union([type_def['id'] for type_def in obo_content['Typedef']])
        self.namespaces = set(namespaces_to_include)

        self.G = construct_nx_graph_from_obo_content(obo_content['Term'], namespaces_to_include=self.namespaces, 
                                                            relationship_types_to_include=self.relationship_types,
                                                            exclude_obsolete=self.exclude_obsolete) # pylint: disable=invalid-name
        self.ids_to_terms = {go_id: term_node['name'] for go_id, term_node in self.G.nodes.items() if 'name' in term_node}
        
        print('Generated graph with', len(self.G.nodes), 'nodes and', len(self.G.edges), 'edges')
        edge_counts_by_type = {}
        for _, _, edge_data in self.G.edges(data=True):
            if edge_data['relationship_type'] not in edge_counts_by_type:
                edge_counts_by_type[edge_data['relationship_type']] = 0
            edge_counts_by_type[edge_data['relationship_type']] += 1

        self.edge_counts_by_type = edge_counts_by_type
        print('Edge counts by type', json.dumps(self.edge_counts_by_type, indent=2))

    def compute_excess_components(self,
                                  predicted_terms: Container[str],
                                  expected_num_components: Union[int, None]=None) -> int:
        """
        Computes the number of excess disconnected components in the predicted subgraph.

        Args:
            predicted_terms: A container of GO IDs representing the predicted terms.
            expected_num_components: The expected number of components. If None the expected number of components 
                is the number of namespaces used to construct the graph.
        
        Returns:
            The number of excess disconnected components in the predicted subgraph.
        """
        if expected_num_components is None:
            expected_num_components = len(self.namespaces)
        predicted_subgraph = self.G.subgraph(predicted_terms)
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
                is the number of namespaces used to construct the graph.

        Returns:
            The number of excess disconnected components in the predicted subgraph normalized by the number of terms.
        """
        return self.compute_excess_components(predicted_terms=predicted_terms,
                                            expected_num_components=expected_num_components) / len(predicted_terms)

    def compute_depths(self, root_id: str) -> List[int]:
        """
        Returns a list of depth values for each node in the graph.
        The depth of a node is the number of edges from the root to the node as found by breadth-first search 
            (this may be different from the "depth" attribute of the nodes).
        The root is the node with the given root_id.

        Args:
            root_id: The ID of the root node to start the breadth-first search from.

        Returns:
            A list of depth values for each node in the graph.
        """
        depths = []
        queue = [(root_id, 0)]
        while queue:
            node_id, depth = queue.pop(0)
            depths.append(depth)
            for _, dest, data in self.G.edges(node_id, data=True):
                if data['relationship_type'] == 'CHILD':
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
                for dest_node, edge_data in self.G[this_node].items():
                    if edge_data['relationship_type'] in edge_types:
                        parent_queue.append(dest_node)
        return propagated_nodes

    def plot_go_subgraph(self,
                           node_list: Container[str],
                           ax: Optional[plt.Axes]=None,
                           label_by_name: bool=False,
                           layout_func: Callable=layout.spring_layout,
                           fig_width: float=12,
                           node_size: float=50,
                           edge_width: float=0.5,
                           node_color: str='lightblue',
                           edge_color: str='gray',
                           label_size: float=6,
                           ) -> Tuple[plt.Figure, plt.Axes]:
        '''
        Plots a subgraph of G containing only the nodes in node_list and only 'CHILD' edges.

        Args:
            node_list: A container of GO IDs representing the nodes to include in the subgraph.
            ax: An optional matplotlib Axes object to plot the graph on.
            label_by_name: Whether to label the nodes by the term names instead of their GO IDs.
            layout_func: A function to layout the graph.
            fig_width: The width of the figure.
            node_size: The size of the nodes.
            edge_width: The width of the edges.
            node_color: The color of the nodes.
            edge_color: The color of the edges.
            label_size: The size of the labels.

        Returns:
            A tuple containing the figure and axes objects.
        '''
        subgraph = self.G.subgraph(node_list)
        pos = layout_func(subgraph)
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(fig_width, fig_width), dpi=300)
        else:
            fig = ax.get_figure()

        nx.draw_networkx_nodes(
            G=subgraph,
            pos=pos,
            ax=ax,
            node_color=node_color,
            node_size=node_size
        )
        # Filter edges to only those labeled as 'CHILD'
        child_edges = [
            (u, v) for u, v, d in subgraph.edges(data=True)
            if d.get('relationship_type') == 'CHILD'
        ]
        nx.draw_networkx_edges(
            G=subgraph,
            pos=pos,
            edgelist=child_edges,
            ax=ax,
            edge_color=edge_color,
            arrows=True,
            width=edge_width
        )
        if label_by_name:
            node_labels = {node: subgraph.nodes[node]['name'] for node in subgraph.nodes()}
        else:
            node_labels = {node: node for node in subgraph.nodes()}
        nx.draw_networkx_labels(
            G=subgraph,
            pos=pos,
            ax=ax,
            labels=node_labels,
            verticalalignment='bottom',
            horizontalalignment='center',
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
                                            edge_color: str='gray',
                                            ) -> Tuple[plt.Figure, plt.Axes]:
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

        joint_network = self.G.subgraph(joint_terms)

        pos = layout_func(joint_network)

        if ax is None:
            fig, ax = plt.subplots(1, figsize=(fig_width, fig_width), dpi=300)
        else:
            fig = ax.get_figure()

        nx.draw_networkx_nodes(
            G=self.G.subgraph(tp_terms),
            pos=pos,
            ax=ax,
            node_color=tp_color,
            node_size=node_size
        )

        nx.draw_networkx_nodes(
            G=self.G.subgraph(fp_terms),
            pos=pos,
            ax=ax,
            node_color=fp_color,
            node_size=node_size
        )

        nx.draw_networkx_nodes(
            G=self.G.subgraph(fn_terms),
            pos=pos,
            ax=ax,
            node_color=fn_color,
            node_size=node_size
        )

        # Filter edges to only those labeled as 'CHILD'
        child_edges = [
            (u, v) for u, v, d in joint_network.edges(data=True)
            if d.get('relationship_type') == 'CHILD'
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
            verticalalignment='bottom',
            horizontalalignment='center',
            font_size=label_size
        )
        ax.set_axis_off()
        return fig, ax