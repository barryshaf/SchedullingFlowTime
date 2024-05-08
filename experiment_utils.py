from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
import numpy as np

from hamiltonians import get_exact_ground
from mub_state_gen import generate_all_subsets
from landscape import (
    calculate_energy_landscape,
    find_k_best_points,
    find_k_worst_points,
    flatten_results,
    TotalLandscapeResult,
)
from graphing import display_energy_landscape, display_energy_histogram, plot_VQE_evals
from vqe import get_standard_params, run_vqe_experiment, MyVQEResult, Parameters

from ipynb.fs.full.ansatz import (
    params_MUB_1q,
    params_MUB_2q,
    gen_expressive_ansatz_1qubit,
    gen_expressive_ansatz_2qubits,
)

## Functions for landscape experiments ##


def run_and_record_landscape(ham: SparsePauliOp, n_mub_qubits: int) -> TotalLandscapeResult:
    n_qubits = ham.num_qubits
    assert n_mub_qubits <= n_qubits
    mub_subsets = generate_all_subsets(n_mub_qubits, n_qubits)
    print(f"attempting all MUB states over the operator {ham}")
    results = calculate_energy_landscape(ham, n_mub_qubits, mub_subsets)
    results.ground_energy = get_exact_ground(ham)
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
    zeroset_anastz = ansatz.assign_parameters([0.0] * ansatz.num_parameters)
    print(f"attempting all MUB states over the operator {ham}")
    results = calculate_energy_landscape(ham, n_mub_qubits, mub_subsets, appended_ansatz=zeroset_anastz)
    results.ground_energy = get_exact_ground(ham)
    print("Energy Landscape:")
    display_energy_landscape(results)
    print("Energy Histogram:")
    display_energy_histogram(results)
    return results


def run_and_record_landscape_shifted_list(hams: list[SparsePauliOp], n_mub_qubits: int, ansatz: QuantumCircuit) -> list[TotalLandscapeResult]:
    return [run_and_record_landscape_shifted(ham, n_mub_qubits, ansatz) for ham in hams]


## Parameter definitions for VQE experiments ##


def get_expressive_1q_params(n_qubits: int, ground_energy: float, record_progress: bool = True) -> Parameters:
    std_params = get_standard_params(n_qubits)
    std_params.ground_energy = ground_energy
    std_params.report_period = 10
    std_params.success_bound = 1e-3
    std_params.num_of_starting_points = 6
    std_params.max_iter = 50
    std_params.record_progress = record_progress
    return std_params


def get_expressive_2q_params(n_qubits: int, ground_energy: float, record_progress: bool = True) -> Parameters:
    std_params = get_standard_params(n_qubits)
    std_params.ground_energy = ground_energy
    std_params.report_period = 10
    std_params.success_bound = 5e-3
    std_params.max_iter = 100
    std_params.record_progress = record_progress
    return std_params


def get_shifted_params(n_qubits: int, ground_energy: float, record_progress: bool = True) -> Parameters:
    std_params = get_standard_params(n_qubits)
    std_params.ground_energy = ground_energy
    std_params.report_period = 10
    std_params.success_bound = 5e-3
    std_params.record_progress = record_progress
    return std_params


## Functions for VQE Experiments from landscape points ##


def run_and_record_vqe_expressive_1q(landscape: TotalLandscapeResult, record_progress: bool = True) -> list[MyVQEResult]:
    points = flatten_results(landscape)
    vqe_results = []
    print(f"The operator {landscape.op} has the exact value {landscape.ground_energy}.")
    print(f"Now trying to reach the value from different MUB points.")
    for point in points:
        initial_thetas = params_MUB_1q(point.index.basis_state_idx, point.index.mub_idx)
        ansatz = gen_expressive_ansatz_1qubit()
        # run VQE from all starting points
        params = get_expressive_1q_params(landscape.n_qubits, get_exact_ground(landscape.op), record_progress)
        print(f"running from state of index {point.index} and value {point.value}")
        vqe_res = run_vqe_experiment(
            hamiltonian=landscape.op,
            ansatz=ansatz,
            initial_thetas=initial_thetas,
            prepened_state_circ=None,
            params=params,
        )
        vqe_res.desc = str(point.index)
        vqe_results.append(vqe_res)
    return vqe_results


def run_and_record_vqe_expressive_2q(landscape: TotalLandscapeResult, record_progress: bool = True) -> list[MyVQEResult]:
    best_5_points = find_k_best_points(landscape, 5)
    params = get_expressive_2q_params(landscape.n_qubits, get_exact_ground(landscape.op), record_progress)
    print(f"The operator {landscape.op} has the exact value {landscape.ground_energy}.")
    print(f"Now trying to reach the value from different MUB points.")
    results = []
    for point in best_5_points:
        initial_thetas = params_MUB_2q(point.index.basis_state_idx, point.index.mub_idx)
        ansatz = gen_expressive_ansatz_2qubits()
        # run VQE from the best 5 examples
        print(f"running from state of index {point.index} and value {point.value}")
        res = run_vqe_experiment(
            hamiltonian=landscape.op,
            ansatz=ansatz,
            initial_thetas=initial_thetas,
            prepened_state_circ=None,
            params=params,
        )
        res.desc = str(point.index)
        results.append(res)
    return results


def run_and_record_vqe_expressive_2q_list(landscapes: list[TotalLandscapeResult], record_progress: bool = True) -> list[list[MyVQEResult]]:
    return [run_and_record_vqe_expressive_2q(landscape, record_progress) for landscape in landscapes]


def run_and_record_vqe_shifted(
    landscape: TotalLandscapeResult,
    ansatz: QuantumCircuit,
    record_progress: bool = True,
) -> list[MyVQEResult]:
    best_5_points = find_k_best_points(landscape, 5)
    params = get_shifted_params(landscape.n_qubits, landscape.ground_energy, record_progress)
    initial_thetas = [0.0] * ansatz.num_parameters
    print(f"The operator {landscape.op} has the exact value {landscape.ground_energy}.")
    print(f"Now trying to reach the value from different MUB points.")
    results = []
    for point in best_5_points:
        print(f"running from state of index {point.index} and value {point.value}")
        res = run_vqe_experiment(
            hamiltonian=landscape.op,
            ansatz=ansatz,
            initial_thetas=initial_thetas,
            prepened_state_circ=point.state_circuit,
            params=params,
        )
        res.desc = str(point.index)
        results.append(res)
    return results


def run_and_record_vqe_shifted_list(landscapes: list[TotalLandscapeResult], ansatz: QuantumCircuit) -> list[list[MyVQEResult]]:
    return [run_and_record_vqe_shifted(landscape, ansatz) for landscape in landscapes]


## Functions for comparing VQE runs from different points ##


def run_and_display_vqe_best_and_worst_expressive_2q(
    landscape: TotalLandscapeResult,
) -> None:
    best_point = find_k_best_points(landscape, 1)[0]
    worst_point = find_k_worst_points(landscape, 1)[0]
    params = get_expressive_2q_params(landscape.n_qubits, get_exact_ground(landscape.op), True)
    print(f"The operator {landscape.op} has the exact value {landscape.ground_energy}.")
    print(f"Now trying to reach the value from the best and worst landscape points.")
    best_initial_thetas = params_MUB_2q(best_point.index.basis_state_idx, best_point.index.mub_idx)
    worst_initial_thetas = params_MUB_2q(worst_point.index.basis_state_idx, worst_point.index.mub_idx)
    ansatz = gen_expressive_ansatz_2qubits()
    best_vqe_run = run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=best_initial_thetas,
        prepened_state_circ=None,
        params=params,
    )
    worst_vqe_run = run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=worst_initial_thetas,
        prepened_state_circ=None,
        params=params,
    )
    best_vqe_run.desc = f"Best starting point: {best_point.index}"
    worst_vqe_run.desc = f"Worst starting point: {worst_point.index}"
    plot_VQE_evals([best_vqe_run, worst_vqe_run])


def run_and_display_vqe_best_and_worst_shifted(landscape: TotalLandscapeResult, ansatz: QuantumCircuit) -> None:
    best_point = find_k_best_points(landscape, 1)[0]
    worst_point = find_k_worst_points(landscape, 1)[0]
    params = get_shifted_params(landscape.n_qubits, landscape.ground_energy, True)
    print(f"The operator {landscape.op} has the exact value {landscape.ground_energy}.")
    print(f"Now trying to reach the value from the best and worst landscape points.")
    initial_thetas = [0.0] * ansatz.num_parameters
    best_vqe_run = run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=initial_thetas,
        prepened_state_circ=best_point.state_circuit,
        params=params,
    )
    worst_vqe_run = run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=initial_thetas,
        prepened_state_circ=worst_point.state_circuit,
        params=params,
    )
    best_vqe_run.desc = f"Best starting point: {best_point.index}"
    worst_vqe_run.desc = f"Worst starting point: {worst_point.index}"
    plot_VQE_evals([best_vqe_run, worst_vqe_run])


def run_and_display_vqe_best_vs_random_shifted(landscape: TotalLandscapeResult, ansatz: QuantumCircuit, num_random: int | None = None) -> None:
    best_point = find_k_best_points(landscape, 1)[0]
    params = get_shifted_params(landscape.n_qubits, landscape.ground_energy, True)
    zero_thetas = [0.0] * ansatz.num_parameters
    best_vqe_run = run_vqe_experiment(
        hamiltonian=landscape.op, ansatz=ansatz, initial_thetas=zero_thetas, prepened_state_circ=best_point.state_circuit, params=params
    )
    best_vqe_run.desc = f"Best starting point: {best_point.index}"
    if num_random is None:
        num_random = landscape.subset_num * landscape.basis_size
    random_thetas_list = [np.random.uniform(low=0.0, high=2 * np.pi, size=(ansatz.num_parameters,)) for _ in range(num_random)]
    random_thetas_vqe_list = []
    for i, random_thetas_entry in enumerate(random_thetas_list):
        curr_res = run_vqe_experiment(hamiltonian=landscape.op, ansatz=ansatz, initial_thetas=random_thetas_entry, prepened_state_circ=None, params=params)
        random_thetas_vqe_list.append(curr_res)
    best_random_theta_vqe_run = max(random_thetas_vqe_list, key=lambda res:res.final_cost)
    best_random_theta_vqe_run.desc = "Best random starting point"
    plot_VQE_evals([best_vqe_run, best_random_theta_vqe_run])
    
