"""
module for calculation of energy landscapes.
"""
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.states import Statevector
from dataclasses import dataclass

import numpy as np

from hamiltonians import get_expectation_value
from mub_state_gen import generate_mub_state_circ

# Probably should be moved to `mub_state_gen.py`
@dataclass
class MUBIndex:
    """
    Small datacless idenitfying a MUB / partial MUB state.
    """
    subset_idx: int  # Index of subset of the register. Can be changed to a list, probably.
    mub_idx: int  # Index of the MUB in the list of MUBs.
    basis_state_idx: int  # Index of the state inside the basis.
    
    def __str__(self) -> str:
        return f"subset {self.subset_idx}, MUB {self.mub_idx}, state {self.basis_state_idx}"


@dataclass
class LandscapeResult:
    """
    Result of calculating the energy of a specific MUB state.
    """
    index: MUBIndex  # Index of the MUB state
    state_circuit: QuantumCircuit  # Circuit to generate the state
    value: np.float64  # Expectation value

# Type aliases.
# Result heirarchy is:
# the total result has a list of the results from each different MUB.
# for each MUB, we have a list of the results on each possible subset of the register.
# for each MUB and subset, we have a list of the results from each state of the MUB.
SubsetLandscapeResult = list[LandscapeResult]
MUBLandscapeResult = list[SubsetLandscapeResult]

@dataclass
class TotalLandscapeResult:
    """
    Result of calculating the entire landscape of a Hamiltonian.
    """
    op: SparsePauliOp                       # Hamiltonian we landscape.
    mub_results: list[MUBLandscapeResult]   # Collection of all results.
    n_mub_qubits: int                       # number of qubits we calculate MUB states on.
    exact_value: np.float64 | None          # The exact result. Used for graphing the result.
    appended_ansatz: QuantumCircuit | None  # An ansatz to append to each MUB state circuit. Used to calculate shifted MUBs.
    n_qubits: int                           # number of qubits in the register.
    desc: str | None
    
    def __init__(self, op: SparsePauliOp, mub_results: list[MUBLandscapeResult],
                 n_mub_qubits: int, appended_ansatz: QuantumCircuit | None = None, 
                 exact_value: np.float64 | None = None):
        self.op = op
        self.mub_results = mub_results
        self.n_mub_qubits = n_mub_qubits
        self.ground_energy = exact_value
        self.appended_ansatz = appended_ansatz
        self.n_qubits = op.num_qubits
        
    @property
    def basis_size(self) -> int:
        return 2 ** self.n_mub_qubits
    
    @property
    def subset_num(self) -> int:
        return len(self.mub_results[0])


def calculate_energy_landscape(op: SparsePauliOp, n_mub_qubits: int, subset_list: list[tuple],
                                appended_ansatz: QuantumCircuit | None = None,
                                plus_for_non_mub: bool = False) -> TotalLandscapeResult:
    """Run an energy landscape calculation.

    Args:
        op (SparsePauliOp): Hamiltonian to check.
        n_mub_qubits (int): number of qubits to generate the MUB states on.
        subset_list (list[tuple]): the specification of all the subsets to add into the calculation. Should probably be calculated inside.
        appended_ansatz (QuantumCircuit | None, optional): *non-parametric* ansatz to add in case we want to calculate shifted MUBs. \
            Since ansatzes *are* usually parametric, make sure to set the parameters to a zero vector before you run this function. \
            Defaults to None, meaning that there is no appended ansatz.
        plus_for_non_mub (bool, optional): set the non-MUB qubits to |+> instead of |0>. Defaults to False.

    Raises:
        Exception: if the number of qubits is not supported.

    Returns:
        TotalLandscapeResult: result of the landscape calculation.
    """
    if n_mub_qubits < 1 or n_mub_qubits > 3:
        raise Exception("We do not support this size of MUB states.")
    num_states = 2 ** n_mub_qubits
    num_mubs = num_states + 1
       
    total_res: list[MUBLandscapeResult] = []
    for mub_idx in range(num_mubs):
        mub_res: MUBLandscapeResult = []
        for subset_idx, subset in enumerate(subset_list):
            subset_res: SubsetLandscapeResult = []
            for state_idx in range(num_states):
                circuit = generate_mub_state_circ(state_idx, mub_idx, op.num_qubits, subset, plus_for_non_mub)
                if appended_ansatz is not None:
                    final_circuit = circuit.compose(appended_ansatz, range(circuit.num_qubits), inplace=False)
                else:
                    final_circuit = circuit
                state = Statevector.from_instruction(final_circuit)
                value = get_expectation_value(state, op)
                full_index = MUBIndex(subset_idx, mub_idx, state_idx)
                state_result = LandscapeResult(full_index, circuit, value)
                subset_res.append(state_result)
            mub_res.append(subset_res)
        total_res.append(mub_res)
    return TotalLandscapeResult(op, total_res, n_mub_qubits, appended_ansatz)



def flatten_results(results: TotalLandscapeResult) -> list[LandscapeResult]:
    """
    Flatten the list-heirarchy of the landscape result to a simple list.
    """
    flat_res: list[LandscapeResult] = []
    for mub_res in results.mub_results:
        for subset_res in mub_res:
            flat_res += subset_res
    return flat_res



def flatten_energies(results: TotalLandscapeResult) -> list[np.float64]:
    """
    Return the list of all energies found in the landscape calculation.
    """
    flat_res = flatten_results(results)
    return [result.value for result in flat_res]



def find_k_best_points(results: TotalLandscapeResult, k: int) -> list[LandscapeResult]:
    """Return the $k$ landscape points with the lowest energy value.

    Args:
        results (TotalLandscapeResult): landscape result strcture.
        k (int): number of points to return.
    """
    results = flatten_results(results)
    return list(sorted(results, key=(lambda x: x.value))[:k])


def find_k_worst_points(results: TotalLandscapeResult, k: int) -> list[LandscapeResult]:
    """Return the $k$ landscape points with the highest energy value.

    Args:
        results (TotalLandscapeResult): landscape result strcture.
        k (int): number of points to return.
    """
    results = flatten_results(results)
    return list(reversed(sorted(results, key=(lambda x: x.value))))[:k]

