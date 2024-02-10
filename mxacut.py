import numpy as np
import networkx as nx
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer


# Code for Max-Cut Hamiltonians
def get_graph_example(n_notes: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(np.arange(0, n, 1))
    elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    # tuple is (i,j,weight) where (i,j) is the edge
    G.add_weighted_edges_from(elist)

    colors = ["r" for node in G.nodes()]
    pos = nx.spring_layout(G)
    return G



def get_weight_matrix(G: nx.Graph) -> np.ndarray:
    # Computing the weight matrix from the random graph
    n = G.number_of_nodes()
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp["weight"]
    return w



def solve_maxcut_brute_force(G: nx.Graph) -> tuple[int, list[int]]:
    n = G.number_of_nodes()
    w = get_weight_matrix(G)
    best_cost_brute = 0
    for b in range(2**n):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
        cost = 0
        for i in range(n):
            for j in range(n):
                cost = cost + w[i, j] * x[i] * (1 - x[j])
        if best_cost_brute < cost:
            best_cost_brute = cost
            xbest_brute = x
        print("case = " + str(x) + " cost = " + str(cost))

    return xbest_brute, best_cost_brute