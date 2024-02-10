from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.states import Statevector

import numpy as np
from numpy.linalg import eig, norm



def gen_trans_ising_op(num_qubits: int,
                        zz_coeff: float,
                        x_coeff: float,
                        toric_bounds: bool=False) -> SparsePauliOp:
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


def get_exact_ground(op: SparsePauliOp) -> np.float64:
    eig_res = eig(op.to_matrix())
    return min(eig_res.eigenvalues).real


def get_expectation_value(state: Statevector, op: SparsePauliOp) -> np.float64:
    return np.round(state.expectation_value(op).real, 10)
