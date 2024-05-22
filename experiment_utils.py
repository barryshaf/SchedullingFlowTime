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
from vqe import get_standard_params, run_vqe_experiment, sample_single_vqe_value, MyVQEResult, Parameters

from ipynb.fs.full.ansatz import (
    params_MUB_1q,
    params_MUB_2q,
    gen_expressive_ansatz_1qubit,
    gen_expressive_ansatz_2qubits,
)

## Functions for landscape experiments ##


def run_and_record_landscape(ham: SparsePauliOp, n_mub_qubits: int, desc: str = "") -> TotalLandscapeResult:
    n_qubits = ham.num_qubits
    assert n_mub_qubits <= n_qubits
    mub_subsets = generate_all_subsets(n_mub_qubits, n_qubits)
    print(f"attempting all MUB states over the operator {ham if desc == "" else desc}")
    results = calculate_energy_landscape(ham, n_mub_qubits, mub_subsets)
    results.ground_energy = get_exact_ground(ham)
    print("Energy Landscape:")
    display_energy_landscape(results, graph_title="Energy Landscape" if desc == "" else f"Energy Landscape of {desc}")
    print("Energy Histogram:")
    display_energy_histogram(results, graph_title="Energy Histogram" if desc == "" else f"Energy Histogram of {desc}")
    return results


def run_and_record_landscape_list(hams: list[SparsePauliOp], n_mub_qubits: int, descs: list[str] = None) -> list[TotalLandscapeResult]:
    if descs is None:
        descs = [""] * len(hams)
    return [run_and_record_landscape(ham, n_mub_qubits, desc) for ham,desc  in zip(hams,descs)]


def run_and_record_landscape_shifted(ham: SparsePauliOp, n_mub_qubits: int, ansatz: QuantumCircuit, desc: str = "") -> TotalLandscapeResult:
    n_qubits = ham.num_qubits
    assert n_mub_qubits <= n_qubits
    mub_subsets = generate_all_subsets(n_mub_qubits, n_qubits)
    zeroset_anastz = ansatz.assign_parameters([0.0] * ansatz.num_parameters)
    print(f"attempting all MUB states over the operator {ham if desc == "" else desc}")
    results = calculate_energy_landscape(ham, n_mub_qubits, mub_subsets, appended_ansatz=zeroset_anastz)
    results.ground_energy = get_exact_ground(ham)
    print("Energy Landscape:")
    display_energy_landscape(results, graph_title="Energy Landscape" if desc == "" else f"Energy Landscape of {desc}")
    print("Energy Histogram:")
    display_energy_histogram(results, graph_title="Energy Histogram" if desc == "" else f"Energy Histogram of {desc}")
    return results


def run_and_record_landscape_shifted_list(hams: list[SparsePauliOp], n_mub_qubits: int, ansatz: QuantumCircuit, descs: list[str] = None) -> list[TotalLandscapeResult]:
    if descs is None:
        descs = [""] * len(hams)
    return [run_and_record_landscape_shifted(ham, n_mub_qubits, ansatz, desc) for ham,desc in zip(hams,descs)]


## Parameter definitions for VQE experiments ##


def get_expressive_1q_params(n_qubits: int, ground_energy: float, record_progress: bool = True) -> Parameters:
    std_params = get_standard_params(n_qubits)
    std_params.ground_energy = ground_energy
    std_params.report_period = 10
    std_params.success_bound = 1e-3
    std_params.num_of_starting_points = 6
    std_params.max_iter = 50
    std_params.record_progress = record_progress
    std_params.tol=1e-1
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
    landscape: TotalLandscapeResult, k: int = 3
) -> None:
    best_points = find_k_best_points(landscape, k)
    worst_points = find_k_worst_points(landscape, k)
    params = get_expressive_2q_params(landscape.n_qubits, get_exact_ground(landscape.op), True)
    print(f"The operator {landscape.op} has the exact value {landscape.ground_energy}.")
    print(f"Now trying to reach the value from the best and worst landscape points.")
    best_initial_thetas = [params_MUB_2q(point.index.basis_state_idx, point.index.mub_idx) for point in best_points]
    worst_initial_thetas = [params_MUB_2q(point.index.basis_state_idx, point.index.mub_idx) for point in worst_points]
    ansatz = gen_expressive_ansatz_2qubits()
    
    best_vqe_runs = [run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=initial_thetas,
        prepened_state_circ=None,
        params=params,
        desc=f"index {point.index}, #{i+1} best choice"
    ) for i, point, initial_thetas in zip(range(k), best_points, best_initial_thetas)]
    worst_vqe_runs = [run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=initial_thetas,
        prepened_state_circ=None,
        params=params,
        desc=f"index {point.index}, #{i+1} worst choice"
    ) for i, point, initial_thetas in zip(range(k), worst_points, worst_initial_thetas)]
    
    plot_VQE_evals(best_vqe_runs + worst_vqe_runs)


def run_and_display_vqe_best_and_worst_shifted(landscape: TotalLandscapeResult, ansatz: QuantumCircuit, k: int = 3) -> None:
    best_points = find_k_best_points(landscape, k)
    worst_points = find_k_worst_points(landscape, k)
    params = get_shifted_params(landscape.n_qubits, landscape.ground_energy, True)
    print(f"The operator {landscape.op} has the exact value {landscape.ground_energy}.")
    print(f"Now trying to reach the value from the best and worst landscape points.")
    initial_thetas = [0.0] * ansatz.num_parameters
    best_vqe_runs = [run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=initial_thetas,
        prepened_state_circ=point.state_circuit,
        params=params,
        desc=f"MUB index {point.index}, #{i+1} best choice"
    ) for i, point in zip(range(k), best_points)]
    worst_vqe_runs = [run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=initial_thetas,
        prepened_state_circ=point.state_circuit,
        params=params,
        desc=f"MUB index {point.index}, #{i+1} worst choice"
    ) for i, point in zip(range(k), worst_points)]
    plot_VQE_evals(best_vqe_runs + worst_vqe_runs)


def run_and_display_vqe_best_vs_random_expressive_2q(landscape: TotalLandscapeResult, ansatz: QuantumCircuit, k: int=3) -> None:
    best_points = find_k_best_points(landscape, k)
    params = get_expressive_2q_params(landscape.n_qubits, get_exact_ground(landscape.op), True)
    mub_initial_thetas = [params_MUB_2q(point.index.basis_state_idx, point.index.mub_idx) for point in best_points]

    mub_vqe_runs = [run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=initial_thetas,
        prepened_state_circ=None,
        params=params,
        desc=f"MUB index {point.index}, #{i+1} best choice"
    ) for i, point, initial_thetas in zip(range(k), best_points, mub_initial_thetas)]
    
    num_random = landscape.subset_num * landscape.basis_size
    random_thetas_list = [np.random.uniform(low=0.0, high=2 * np.pi, size=(ansatz.num_parameters,)) for _ in range(num_random)]
    best_random_thetas = sorted(random_thetas_list, key=lambda thetas: sample_single_vqe_value(landscape.op, ansatz, thetas, None, params))[:k]
    best_random_theta_vqe_runs = [run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=random_thetas_entry,
        prepened_state_circ=None,
        params=params,
        desc = f"Random theta, #{i+1} best choice"
    ) for i, random_thetas_entry in enumerate(best_random_thetas)
    ]
    plot_VQE_evals(mub_vqe_runs + best_random_theta_vqe_runs)
    



def run_and_display_vqe_best_vs_random_shifted(landscape: TotalLandscapeResult, ansatz: QuantumCircuit, k: int = 3) -> None:
    best_points = find_k_best_points(landscape, k)
    params = get_shifted_params(landscape.n_qubits, landscape.ground_energy, True)
    zero_thetas = [0.0] * ansatz.num_parameters
    best_vqe_runs = [run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=zero_thetas,
        prepened_state_circ=point.state_circuit,
        params=params,
        desc=f"MUB index {point.index}, good choice #{i}"
    ) for i, point in enumerate(best_points)]
        
    num_random = landscape.subset_num * landscape.basis_size
    random_thetas_list = [np.random.uniform(low=0.0, high=2 * np.pi, size=(ansatz.num_parameters,)) for _ in range(num_random)]
    best_random_thetas = sorted(random_thetas_list, key=lambda thetas: sample_single_vqe_value(landscape.op, ansatz, thetas, None, params))[:k]

    best_random_theta_vqe_runs = [run_vqe_experiment(
        hamiltonian=landscape.op,
        ansatz=ansatz,
        initial_thetas=random_thetas_entry,
        prepened_state_circ=None,
        params=params,
        desc = f"Random theta, good choice #{i}"
    ) for i, random_thetas_entry in enumerate(best_random_thetas)
    ]
    plot_VQE_evals(best_vqe_runs + best_random_theta_vqe_runs)
    
