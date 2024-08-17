"""
Module for generating MUB states and partial MUB states.
"""
from qiskit.circuit import QuantumCircuit
import numpy as np
from itertools import combinations


def generate_mub_state_circ(state_idx: int,
                            mub_idx: int,
                            num_qubits: int,
                            qubit_subset: list[int],
                            plus_for_non_mub: bool = False)-> QuantumCircuit:
    """Generate a circuit that outputs a MUB (or partial MUB) state. \
        The actual generation functions are separated for each # of qubits.

    Args:
        state_idx (int): index of the state inside its basis.
        mub_idx (int): index of the MUB, out of the set of MUBs for a number of qubits.
        num_qubits (int): number of qubits in the entire register.
        qubit_subset (list[int]): list of qubits in the register to put in the MUB state. \
            If this list does not cover the entirety of range(num_qubits), the result is called a *Partial MUB*. \
            This feature exists for cases where we don't know how to find the MUBs for a large \
            number of qubits, or there are too many, but we still want something *like* a MUB state.
        plus_for_non_mub (bool, optional): puts the non-MUB qubits in a partial-MUB state in |+> instead of |0>. Defaults to False.

    Raises:
        Exception: if we don;t know how to generate the MUBs for a certain size of MUB subset.

    Returns:
        QuantumCircuit: the circuit that generates the state.
    """
    circuit = QuantumCircuit(num_qubits)
    
    if len(qubit_subset) == 1:
        prep_MUB1(circuit, state_idx, mub_idx, qubit_subset)
    elif len(qubit_subset) == 2:
        prep_MUB2(circuit, state_idx, mub_idx, qubit_subset)
    elif len(qubit_subset) == 3:
        prep_MUB3(circuit, state_idx, mub_idx, qubit_subset)
    else:
        raise Exception("We do not support this size of MUB states.")
    
    if plus_for_non_mub:
        for qubit in range(num_qubits):
            if qubit not in qubit_subset:
                circuit.h(qubit)
    return circuit



def generate_all_subsets(n_mub_qubits: int, n_qubits: int) -> list[tuple[int]]:
    """
    Generates all subsets of size n_mub_qubits from the list [1 ... n_qubits].
    """
    return list(combinations(np.linspace(0, n_qubits-1, n_qubits, dtype=int), n_mub_qubits))


# Prepares a MUB state of 1 qubit over a circuit.
def prep_MUB1(circ: QuantumCircuit,
              state_idx: int,
              mub_idx: int,
              qubit_subset: list[int]) -> None:
    assert len(qubit_subset) == 1
    assert 0 <= state_idx <= 1
    assert 0 <= mub_idx <= 2
    # act according to in-basis index
    if state_idx == 1:
        circ.x(qubit_subset[0])
    # act to choose MUB
    if mub_idx == 1:
        circ.h(qubit_subset[0])
    elif mub_idx == 2:
        yh(circ, qubit_subset[0])

# Prepares a MUB state of 2 qubits over a circuit.
def prep_MUB2(circ: QuantumCircuit,
                state_idx: int,
                mub_idx: int,
                qubit_subset: int) -> None:
    assert len(qubit_subset) == 2
    assert (0 <= state_idx <= 3)
    assert (0 <= mub_idx <= 4)
    # state_idx chooses the state in the basis (MUB)
    if state_idx == 1:
        circ.x(qubit_subset[0])
    elif state_idx == 2:
        circ.x(qubit_subset[1])
    elif state_idx == 3:
        circ.x(qubit_subset[0])
        circ.x(qubit_subset[1])
    # mub_idx chooses the basis (MUB) itself
    if mub_idx == 1:
        circ.h(qubit_subset[0])
        circ.h(qubit_subset[1])
    elif mub_idx == 2:
        circ.h(qubit_subset[0])
        yh(circ,qubit_subset[1])
        circ.cz(qubit_subset[0], qubit_subset[1])
    elif mub_idx == 3:
        yh(circ,qubit_subset[0])
        yh(circ,qubit_subset[1])
    elif mub_idx == 4:
        yh(circ,qubit_subset[0])
        circ.h(qubit_subset[1])
        circ.cz(qubit_subset[0], qubit_subset[1])


# Prepares a MUB state of 3 qubits over a circuit.
def prep_MUB3(circ: QuantumCircuit,
                state_idx: int,
                mub_idx: int,
                qubit_subset: int) -> None:
    # state_idx chooses the state in the basis (MUB)
    if state_idx == 1:
        circ.x(qubit_subset[0])
    elif state_idx == 2:
        circ.x(qubit_subset[1])
    elif state_idx == 3:
        circ.x(qubit_subset[0])
        circ.x(qubit_subset[1])
    elif state_idx == 4:
        circ.x(qubit_subset[2])
    elif state_idx == 5:
        circ.x(qubit_subset[0])
        circ.x(qubit_subset[2])
    elif state_idx == 6:
        circ.x(qubit_subset[1])
        circ.x(qubit_subset[2])
    elif state_idx == 7:
        circ.x(qubit_subset[0])
        circ.x(qubit_subset[1])
        circ.x(qubit_subset[2])
    # mub_idx chooses the basis (MUB) itself
    if mub_idx == 1:
        circ.h(qubit_subset[0])
        circ.h(qubit_subset[1])
        circ.h(qubit_subset[2])
    elif mub_idx == 2:
        yh(circ, qubit_subset[0])
        yh(circ, qubit_subset[1])
        yh(circ, qubit_subset[2])
    elif mub_idx == 3:
        yh(circ, qubit_subset[0])
        circ.h(qubit_subset[1])
        circ.h(qubit_subset[2])
        circ.cz(qubit_subset[1], qubit_subset[2])
        circ.cz(qubit_subset[0], qubit_subset[1])
    elif mub_idx == 4:
        circ.h(qubit_subset[0])
        yh(circ, qubit_subset[1])
        circ.h(qubit_subset[2])
        circ.cz(qubit_subset[1], qubit_subset[2])
        circ.cz(qubit_subset[0], qubit_subset[2])
    elif mub_idx == 5:
        circ.h(qubit_subset[0])
        circ.h(qubit_subset[1])
        yh(circ, qubit_subset[2])
        circ.cz(qubit_subset[0], qubit_subset[1])
        circ.cz(qubit_subset[0], qubit_subset[2])
    elif mub_idx == 6:
        yh(circ, qubit_subset[0])
        yh(circ, qubit_subset[1])
        circ.h(qubit_subset[2])
        circ.cz(qubit_subset[0], qubit_subset[1])
        circ.cz(qubit_subset[0], qubit_subset[2])
    elif mub_idx == 7:
        yh(circ, qubit_subset[0])
        circ.h(qubit_subset[1])
        yh(circ, qubit_subset[2])
        circ.cz(qubit_subset[1], qubit_subset[2])
        circ.cz(qubit_subset[0], qubit_subset[2])
    elif mub_idx == 8:
        circ.h(qubit_subset[0])
        yh(circ, qubit_subset[1])
        yh(circ, qubit_subset[2])
        circ.cz(qubit_subset[0], qubit_subset[1])
        circ.cz(qubit_subset[1], qubit_subset[2])


# Adds Y-Hadamard gate on qubit q in circ
def yh(circ: QuantumCircuit, q: int) -> None:
    circ.h(q)
    circ.s(q)