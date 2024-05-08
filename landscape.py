"""
## Result heirarchy
The list heirarchy is defined by the following indexes, in this order:
1. MUB used.
2. Subset of qubits on which this MUB was applied.
3. The specific MUB states.
"""
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.states import Statevector
from dataclasses import dataclass

import numpy as np

from hamiltonians import get_expectation_value
from mub_state_gen import generate_mub_state_circ


@dataclass
class MUBIndex:
    subset_idx: int
    mub_idx: int
    basis_state_idx: int
    
    def __str__(self) -> str:
        return f"subset {self.subset_idx}, MUB {self.mub_idx}, state {self.basis_state_idx}"

@dataclass
class LandscapeResult:
    index: MUBIndex
    state_circuit: QuantumCircuit
    value: np.float64
    
SubsetLandscapeResult = list[LandscapeResult]
MUBLandscapeResult = list[SubsetLandscapeResult]

@dataclass
class TotalLandscapeResult:
    op: SparsePauliOp
    mub_results: list[MUBLandscapeResult]
    n_mub_qubits: int
    exact_value: np.float64 | None
    appended_ansatz: QuantumCircuit | None
    # op_desc: str | None
    n_qubits: int
    
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
    
# TotalLandscapeResult = list[MUBLandscapeResult]


def calculate_energy_landscape(op: SparsePauliOp, n_mub_qubits: int, subset_list: list[tuple],
                                appended_ansatz: QuantumCircuit | None = None,
                                plus_for_non_mub: bool = False) -> TotalLandscapeResult:
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
    flat_res: list[LandscapeResult] = []
    for mub_res in results.mub_results:
        for subset_res in mub_res:
            flat_res += subset_res
    return flat_res



def flatten_energies(results: TotalLandscapeResult) -> list[np.float64]:
    flat_res = flatten_results(results)
    return [result.value for result in flat_res]



def find_k_best_points(results: TotalLandscapeResult, k: int) -> list[LandscapeResult]:
    results = flatten_results(results)
    return list(sorted(results, key=(lambda x: x.value))[:k])


def find_k_worst_points(results: TotalLandscapeResult, k: int) -> list[LandscapeResult]:
    results = flatten_results(results)
    return list(reversed(sorted(results, key=(lambda x: x.value))))[:k]

