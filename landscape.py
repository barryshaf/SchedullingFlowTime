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

import numpy as np
import matplotlib.pyplot as plt

from .hamiltonians import get_expectation_value
from .mub_state_gen import generate_mub_state_circ

NUM_STATES_2_QUBITS = 4
NUM_MUBS_2_QUBITS = 5
NUM_STATES_3_QUITS = 8
NUM_MUBS_3_QUBTIS = 9
FIG_SIZE = (8,5)

ResultTuple = tuple[QuantumCircuit, np.float64]
LandscapeResultsType = list[list[list[ResultTuple]]]



def calculate_energy_landscape(op: SparsePauliOp, n_mub_qubits: int, subset_list: list[tuple],
                                appended_ansatz: QuantumCircuit | None = None,
                                plus_for_non_mub: bool = False) -> LandscapeResultsType:
    num_states = num_mubs = 0
    if n_mub_qubits == 2:
        num_states = NUM_STATES_2_QUBITS
        num_mubs = NUM_MUBS_2_QUBITS
    elif n_mub_qubits == 3:
        num_states = NUM_STATES_3_QUITS
        num_mubs = NUM_MUBS_3_QUBTIS
    else:
        raise Exception("We do not support this size of MUB states.")
    total_res = []
    for mub_idx in range(num_mubs):
        mub_res = []
        for subset in subset_list:
            subset_res = []
            for state_idx in range(num_states):
                circuit = generate_mub_state_circ(state_idx, mub_idx, op.num_qubits, subset, plus_for_non_mub)
                if appended_ansatz is not None:
                    final_circuit = circuit.compose(appended_ansatz, range(circuit.num_qubits), inplace=False)
                else:
                    final_circuit = circuit
                state = Statevector.from_instruction(final_circuit)
                res = get_expectation_value(state, op)
                subset_res.append((circuit, res))
            mub_res.append(subset_res)
        total_res.append(mub_res)
    return total_res



def flatten_results(results: LandscapeResultsType) -> list[ResultTuple]:
    flat_res = []
    for mub_res in results:
        for subset_res in mub_res:
            flat_res += subset_res
    return flat_res



def flatten_energies(results: LandscapeResultsType) -> list[np.float64]:
    flat_res = flatten_results(results)
    return [energy for circuit, energy in flat_res]



def find_k_best_results(results: LandscapeResultsType, k: int) -> list[ResultTuple]:
    results = flatten_results(results)
    return sorted(results, key=(lambda x: x[1]))[:k]



def display_energy_landscape(energy_landscape_results: LandscapeResultsType, exact_result: np.float64, graph_title="Energy landscape",
                                show_legend=False):
    fig = plt.figure(figsize=FIG_SIZE)
    idx_counter = 0
    basis_size = len(energy_landscape_results[0][0])
    mub_results_size = basis_size * len(energy_landscape_results[0])
    for i, mub_res in enumerate(energy_landscape_results):
        for j, subset_res in enumerate(mub_res):
            energies_only = [energy for circuit, energy in subset_res]
            plt.plot(list(range(idx_counter, idx_counter+basis_size)), energies_only, 'o', lw=0.4, label=f"MUB {i}, subset {j}")
            idx_counter += basis_size
        # Show separation between different MUBs
        plt.axvspan(idx_counter - mub_results_size, idx_counter, alpha=0.1, color=f"C{i}")
    # Show exact result
    plt.axhline(y=exact_result, lw=0.6, color='red')
    # Show comp. basis specifically
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(x=basis_size*0.25, y=ymin + (ymax-ymin)*0.8, s='COMP', fontsize=10)
    
    plt.xlabel("MUB state index")
    plt.ylabel("Cost function result")
    if show_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(graph_title)
    plt.show()



def display_energy_histogram(energy_landscape_results: LandscapeResultsType, exact_result: np.float64, bins=100,
                                graph_title="Energy landscape histogram", show_legend=False):
    fig = plt.figure(figsize=FIG_SIZE)
    plt.locator_params(axis='x', nbins=min(bins//2, 30), tight=True)
    plt.xticks(fontsize=10, rotation=60)
    plt.locator_params(axis='y', nbins=10)

    flat_results = flatten_energies(energy_landscape_results)
    plt.hist(flat_results, bins)
    # Show exact result
    plt.axvline(x=exact_result, lw=1, color='red')

    plt.xlabel("Cost function result")
    plt.ylabel("number of results")
    if show_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(graph_title)
    plt.show()