"""
Module for MAX-CUT related utilities.
"""
import numpy as np
import networkx as nx
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.problems import QuadraticProgram


def gen_graph(n_nodes: int, edges: list[tuple[int, int]]) -> nx.Graph:
    """
    Generate a grap from a # of nodes and a list of edges.
    """
    G = nx.Graph()
    G.add_nodes_from(np.arange(0, n_nodes, 1))
    elist = [(u,v,1.0) for u,v in edges]
    # tuple is (i,j,weight) where (i,j) is the edge
    G.add_weighted_edges_from(elist)

    return G



def compute_weight_matrix(graph: nx.Graph) -> np.ndarray:
    """Transform a graph to an adjacency matrix.

    Args:
        graph (nx.Graph): input graph.

    Returns:
        np.ndarray: weight matrix.
    """
    n = graph.number_of_nodes()
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = graph.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = 1
    return w

def create_quadratic_program(graph: nx.Graph) -> QuadraticProgram:
    """
    Intermediate form used to generate a MAX-CUT Hamiltonian.
    """
    w = compute_weight_matrix(graph)
    max_cut = Maxcut(w)
    qp = max_cut.to_quadratic_program()
    return qp