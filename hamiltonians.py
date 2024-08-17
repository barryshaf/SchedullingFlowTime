"""
Module for generating useful types of Hamiltonians.
"""
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.states import Statevector

import numpy as np
from numpy.linalg import eig
import networkx as nx

from maxcut import gen_graph, create_quadratic_program


def gen_trans_ising_op(num_qubits: int,
                        zz_coeff: float,
                        x_coeff: float,
                        toric_bounds: bool=False) -> SparsePauliOp:
    """Generate a Transverse-Field Ising Model Hamiltonian.

    Args:
        num_qubits (int): number of qubis for the operator.
        zz_coeff (float): coefficient for all the ZZ terms in the Hamiltonian.
        x_coeff (float): coefficient for all the X terms in the Hamiltonian.
        toric_bounds (bool, optional): Whether to add a ZZ term between the first and last qubits. Defaults to False.

    Returns:
        SparsePauliOp: The hamiltonian.
    """
    terms = []
    coeffs = []
    # Adding the ZZ terms
    for i in range(num_qubits if toric_bounds else num_qubits-1):
        curr_term = ['I'] * num_qubits
        curr_term[i] = 'Z'
        curr_term[(i+1)%num_qubits] = 'Z'
        terms.append(''.join(curr_term))
        coeffs.append(zz_coeff)
    # Adding the X terms
    for i in range(num_qubits):
        curr_term = ['I'] * num_qubits
        curr_term[i] = 'X'
        terms.append(''.join(curr_term))
        coeffs.append(x_coeff)
    return SparsePauliOp(terms, coeffs)



def gen_maxcut_op(n_nodes: int,  edges: list[tuple[int, int]]) -> SparsePauliOp:
    """Generate a Hamiltonian reduction of a MAX-CUT optimization problem.

    Args:
        n_nodes (int): number of nodes in the graph, also the number of qubits.
        edges (list[tuple[int, int]]): edges in the graph.

    Returns:
        SparsePauliOp: The hamiltonian.
    """
    g = gen_graph(n_nodes, edges)
    qp = create_quadratic_program(g)
    op, offset = qp.to_ising()
    return op + SparsePauliOp(['I'*n_nodes], [offset])

def gen_maxcut_op_from_graph(g: nx.Graph) -> SparsePauliOp:
    """Instead of feeding the edges manually, make the Hamiltonian using a networkx graph.

    Args:
        g (nx.Graph): the networkx graph.

    Returns:
        SparsePauliOp: The hamiltonian.
    """
    qp = create_quadratic_program(g)
    op, offset = qp.to_ising()
    return op + SparsePauliOp(['I'*g.number_of_nodes()], [offset])

def get_exact_ground(op: SparsePauliOp) -> np.float64:
    """
    Diagonalize a Hamiltonian to get its exact ground state.
    """
    eig_res = eig(op.to_matrix())
    return min(eig_res.eigenvalues).real


def get_expectation_value(state: Statevector, op: SparsePauliOp) -> np.float64:
    """
    Calculate the expectation value of a state vector over some Hamiltonian.
    """
    return np.round(state.expectation_value(op).real, 10)
