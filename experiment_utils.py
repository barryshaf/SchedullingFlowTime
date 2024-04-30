from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
import numpy as np

from hamiltonians import get_exact_ground
from mub_state_gen import generate_all_subsets
from landscape import calculate_energy_landscape, find_k_best_points, flatten_results, TotalLandscapeResult
from graphing import display_energy_landscape, display_energy_histogram
from vqe import get_standard_params, run_vqe_experiment, MyVQEResult

from ipynb.fs.full.ansatz import params_MUB_1q, params_MUB_2q, gen_expressive_ansatz_1qubit, gen_expressive_ansatz_2qubits

def run_and_record_landscape(ham: SparsePauliOp, n_mub_qubits: int) -> TotalLandscapeResult:
    n_qubits = ham.num_qubits
    assert n_mub_qubits <= n_qubits
    mub_subsets = generate_all_subsets(n_mub_qubits, n_qubits)
    print(f"attempting all MUB states over the operator {ham}")
    results = calculate_energy_landscape(ham, n_mub_qubits, mub_subsets)
    results.exact_value = get_exact_ground(ham)
    print("Energy Landscape:")
    display_energy_landscape(results)
    print("Energy Histogram:")
    display_energy_histogram(results)
    return results

def run_and_record_landscape_list(hams: list[SparsePauliOp], n_mub_qubits: int) -> list[TotalLandscapeResult]:
    return [run_and_record_landscape(ham, n_mub_qubits) for ham in hams]

def run_and_record_landscape_shifted(ham: SparsePauliOp, n_mub_qubits: int, ansatz: QuantumCircuit) -> TotalLandscapeResult:
    n_qubits = ham.num_qubits
    assert n_mub_qubits <= n_qubits
    mub_subsets = generate_all_subsets(n_mub_qubits, n_qubits)
    zeroset_anastz = ansatz.assign_parameters([0.0]*ansatz.num_parameters)
    print(f"attempting all MUB states over the operator {ham}")
    results = calculate_energy_landscape(ham, n_mub_qubits, mub_subsets, appended_ansatz=zeroset_anastz)
    results.exact_value = get_exact_ground(ham)
    print("Energy Landscape:")
    display_energy_landscape(results)
    print("Energy Histogram:")
    display_energy_histogram(results)
    return results

def run_and_record_landscape_shifted_list(hams: list[SparsePauliOp], n_mub_qubits: int, ansatz: QuantumCircuit) -> list[TotalLandscapeResult]:
    return [run_and_record_landscape_shifted(ham, n_mub_qubits, ansatz) for ham in hams]


def run_and_record_vqe_expressive_1q(landscape: TotalLandscapeResult, record_progress: bool = False) -> list[MyVQEResult]:
    points = flatten_results(landscape)
    vqe_results = []
    print(f"The operator {landscape.op} has the exact value {landscape.exact_value}.")
    print(f"Now trying to reach the value from different MUB points.")
    for point in points:
        initial_thetas = params_MUB_1q(point.index.basis_state_idx, point.index.mub_idx)
        ansatz = gen_expressive_ansatz_1qubit()
        # run VQE from all starting points
        params = get_standard_params(landscape.n_qubits)
        params.exact_result = get_exact_ground(landscape.op)
        params.report_period = 10
        params.success_bound = 1e-3
        params.num_of_starting_points = 6
        params.max_iter = 50
        params.record_progress = record_progress
        print(f"running from state of index {point.index} and value {point.value}")
        vqe_res = run_vqe_experiment(hamiltonian=landscape.op, ansatz=ansatz, initial_thetas=initial_thetas,
                        prepened_state_circ=None, params=params)
        vqe_res.desc = str(point.index)
        vqe_results.append(vqe_res)
    return vqe_results


def run_and_record_vqe_expressive_2q(landscape: TotalLandscapeResult, record_progress: bool = False) -> list[MyVQEResult]:
    best_5_points = find_k_best_points(landscape, 5)
    params = get_standard_params(landscape.n_qubits)
    params.exact_result = get_exact_ground(landscape.op)
    params.success_bound = 5e-3
    params.max_iter = 100
    params.record_progress = record_progress
    print(f"The operator {landscape.op} has the exact value {landscape.exact_value}.")
    print(f"Now trying to reach the value from different MUB points.")
    results = []
    for point in best_5_points:
        initial_thetas = params_MUB_2q(point.index.basis_state_idx, point.index.mub_idx)
        ansatz = gen_expressive_ansatz_2qubits()
        # run VQE from the best 5 examples
        print(f"running from state of index {point.index} and value {point.value}")
        res = run_vqe_experiment(hamiltonian=landscape.op, ansatz=ansatz, initial_thetas=initial_thetas,
                        prepened_state_circ=None, params=params)
        res.desc = str(point.index)
        results.append(res)
    return results

def run_and_record_vqe_expressive_2q_list(landscapes: list[TotalLandscapeResult], record_progress: bool = False) -> list[list[MyVQEResult]]:
    return [run_and_record_vqe_expressive_2q(landscape, record_progress) for landscape in landscapes]


def run_and_record_vqe_shifted(landscape: TotalLandscapeResult, ansatz: QuantumCircuit, record_progress: bool = False) -> list[MyVQEResult]:
    best_5_points = find_k_best_points(landscape, 5)
    initial_thetas = [0.0]*ansatz.num_parameters
    params = get_standard_params(landscape.n_qubits)
    params.exact_result = landscape.exact_result
    params.success_bound = 5e-3
    params.record_progress = record_progress
    landscape.exact_result = get_exact_ground(landscape.op)
    print(f"The operator {landscape.op} has the exact value {landscape.exact_value}.")
    print(f"Now trying to reach the value from different MUB points.")
    results = []
    for point in best_5_points:
        print(f"running from state of index {point.index} and value {point.value}")
        res = run_vqe_experiment(hamiltonian=landscape.op, ansatz=ansatz, initial_thetas=initial_thetas, prepened_state_circ=point.state_circuit, params=params)
        res.desc = str(point.index)
        results.append(res)
    return results
        
def run_and_record_vqe_shifted_list(landscapes: list[TotalLandscapeResult], ansatz: QuantumCircuit) -> list[list[MyVQEResult]]:
    return [run_and_record_vqe_shifted(landscape, ansatz) for landscape in landscapes]

