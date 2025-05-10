import obonet

from networkx import is_directed_acyclic_graph

class GOInterpreter:
    """Class to interpret Gene Ontology (GO) terms and their relationships."""

    def __init__(self, obo_path: str):
        graph = obonet.read_obo(obo_path)

        if not is_directed_acyclic_graph(graph):
            raise ValueError("Invalid gene ontology network.")
        
        self.graph = graph

    def get_names(self, go_terms: list[str]) -> list[str]:
        """Get the names of the GO terms."""

        return [self.graph.nodes[go_term]["name"] for go_term in go_terms]
