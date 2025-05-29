"""
Module with definitions of specific experiments.
"""

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

# it's in a notebook because it's easier to keep the math that way.
from ipynb.fs.full.ansatz import (
    params_MUB_1q,
    params_MUB_2q,
    gen_expressive_ansatz_1qubit,
    gen_expressive_ansatz_2qubits,
)

## Functions for landscape experiments ##


def run_and_record_landscape(ham: SparsePauliOp, n_mub_qubits: int, desc: str = "") -> TotalLandscapeResult:
    """Calculate a landscape, display and return the result.

    Args:
        ham (SparsePauliOp): hamiltonian to landscape.
        n_mub_qubits (int): size of qubit subsets to generate MUBs on.
        desc (str, optional): graph title. Defaults to "".

    Returns:
        TotalLandscapeResult: result of the landscape calculation.
    """
    n_qubits = ham.num_qubits
    assert n_mub_qubits <= n_qubits
    mub_subsets = generate_all_subsets(n_mub_qubits, n_qubits)
    print("attempting all MUB states over the operator", ham if desc == '' else desc)
    results = calculate_energy_landscape(ham, n_mub_qubits, mub_subsets)
    results.ground_energy = get_exact_ground(ham)
    print("Energy Landscape:")
    display_energy_landscape(results, graph_title="Energy Landscape" if desc == "" else f"Energy Landscape of {desc}")
    print("Energy Histogram:")
    display_energy_histogram(results, graph_title="Energy Histogram" if desc == "" else f"Energy Histogram of {desc}")
    return results


def run_and_record_landscape_list(hams: list[SparsePauliOp], n_mub_qubits: int, descs: list[str] = None) -> list[TotalLandscapeResult]:
    """
    Batch version of run_and_record_landscape.
    """
    if descs is None:
        descs = [""] * len(hams)
    return [run_and_record_landscape(ham, n_mub_qubits, desc) for ham,desc  in zip(hams,descs)]


def run_and_record_landscape_shifted(ham: SparsePauliOp, n_mub_qubits: int, ansatz: QuantumCircuit, desc: str = "") -> TotalLandscapeResult:
    """Calculate a landscape *using shifted MUBs*, display and return the result.

    Args:
        ham (SparsePauliOp): hamiltonian to landscape.
        n_mub_qubits (int): size of qubit subsets to generate MUBs on.
        desc (str, optional): graph title. Defaults to "".

    Returns:
        TotalLandscapeResult: result of the landscape calculation.
    """
    n_qubits = ham.num_qubits
    assert n_mub_qubits <= n_qubits
    mub_subsets = generate_all_subsets(n_mub_qubits, n_qubits)
    zeroset_anastz = ansatz.assign_parameters([0.0] * ansatz.num_parameters)
    #print(f"attempting all MUB states over the operator {ham if desc == "" else desc}")
    results = calculate_energy_landscape(ham, n_mub_qubits, mub_subsets, appended_ansatz=zeroset_anastz)
    results.ground_energy = get_exact_ground(ham)
    print("Energy Landscape:")
    display_energy_landscape(results, graph_title="Energy Landscape" if desc == "" else f"Energy Landscape of {desc}")
    print("Energy Histogram:")
    display_energy_histogram(results, graph_title="Energy Histogram" if desc == "" else f"Energy Histogram of {desc}")
    return results


def run_and_record_landscape_shifted_list(hams: list[SparsePauliOp], n_mub_qubits: int, ansatz: QuantumCircuit, descs: list[str] = None) -> list[TotalLandscapeResult]:
    """
    Batch version of run_and_record_landscape_shifted.
    """
    if descs is None:
        descs = [""] * len(hams)
    return [run_and_record_landscape_shifted(ham, n_mub_qubits, ansatz, desc) for ham,desc in zip(hams,descs)]


## Parameter definitions for VQE experiments ##
# Yes, I know, this is clearly a misuse of get_standard_params.
# You are welcome to improve this if you want to.

def get_expressive_1q_params(n_qubits: int, ground_energy: float, record_progress: bool = True) -> Parameters:
    """
    VQE hyperparameters for experiments on 1 qubit with an expressive ansatz.
    """
    std_params = get_standard_params(n_qubits)
    std_params.ground_energy = ground_energy
    std_params.report_period = 10
    std_params.success_bound = 1e-3
    std_params.tol = 1e-1
    std_params.num_of_starting_points = 6
    std_params.max_iter = 50
    std_params.record_progress = record_progress
    std_params.tol=1e-1
    return std_params


def get_expressive_2q_params(n_qubits: int, ground_energy: float, record_progress: bool = True) -> Parameters:
    """
    VQE hyperparameters for experiments on 2 qubits with an expressive ansatz.
    """
    std_params = get_standard_params(n_qubits)
    std_params.ground_energy = ground_energy
    std_params.report_period = 20
    std_params.success_bound = 5e-3
    std_params.max_iter = 100
    std_params.record_progress = record_progress
    return std_params


def get_shifted_params(n_qubits: int, ground_energy: float, record_progress: bool = True) -> Parameters:
    """
    VQE hyperparameters for experiments on shifted MUBs.
    """
    std_params = get_standard_params(n_qubits)
    std_params.ground_energy = ground_energy
    std_params.report_period = 200
    std_params.success_bound = 5e-3
    std_params.record_progress = record_progress
    return std_params


## Functions for VQE Experiments from landscape points ##

def run_and_record_vqe_expressive(landscape: TotalLandscapeResult, record_progress: bool = True) -> list[MyVQEResult]:
    """Run a VQE experiment for 1 qubit using an expressive ansatz from all landscape points.

    Args:
        landscape (TotalLandscapeResult): landscape points.
        record_progress (bool, optional): whether to keep the list of theta vectors and cost evals. Defaults to True.

    Returns:
        list[MyVQEResult]: The VQE result from each separate landscape point.
    """
    points = flatten_results(landscape)
    vqe_results = []
    print(f"The operator {landscape.op} has the exact value {landscape.ground_energy}.")
    print(f"Now trying to reach the value from different MUB points.")
    for point in points:
        # run VQE from all starting points
        params = get_expressive_1q_params(landscape.n_qubits, get_exact_ground(landscape.op), record_progress)
        print(f"running from state of index {point.index} and value {point.value}")
        vqe_res = run_vqe_experiment(
            hamiltonian=landscape.op,
            ansatz=None,
            initial_thetas=None,
            prepened_state_circ=None,
            params=params,
        )
        vqe_res.desc = str(point.index)
        vqe_results.append(vqe_res)
    return vqe_results
####
def run_and_record_vqe_expressive_1q(landscape: TotalLandscapeResult, record_progress: bool = True) -> list[MyVQEResult]:
    """Run a VQE experiment for 1 qubit using an expressive ansatz from all landscape points.

    Args:
        landscape (TotalLandscapeResult): landscape points.
        record_progress (bool, optional): whether to keep the list of theta vectors and cost evals. Defaults to True.

    Returns:
        list[MyVQEResult]: The VQE result from each separate landscape point.
    """
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
    """Run a VQE experiment for 2 qubits using an expressive ansatz from all landscape points.

    Args:
        landscape (TotalLandscapeResult): landscape points.
        record_progress (bool, optional): whether to keep the list of theta vectors and cost evals. Defaults to True.

    Returns:
        list[MyVQEResult]: The VQE result from each separate landscape point.
    """
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
    """
    Batch version of run_and_record_vqe_expressive_2q.
    """
    return [run_and_record_vqe_expressive_2q(landscape, record_progress) for landscape in landscapes]


def run_and_record_vqe_shifted(
    landscape: TotalLandscapeResult,
    ansatz: QuantumCircuit,
    record_progress: bool = True,
) -> list[MyVQEResult]:
    """Run a VQE experiment using a non-expressive ansatz and shifted MUBs from all landscape points.

    Args:
        landscape (TotalLandscapeResult): landscape points.
        ansatz (QuantumCircuit): parametric, non-expressive ansatz to use. 
        record_progress (bool, optional): whether to keep the list of theta vectors and cost evals. Defaults to True.

    Returns:
        list[MyVQEResult]: The VQE result from each separate landscape point.
    """
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
    """
    Batch version of run_and_record_vqe_shifted.
    """
    return [run_and_record_vqe_shifted(landscape, ansatz) for landscape in landscapes]


## Functions for comparing VQE runs from different points ##


def run_and_display_vqe_best_and_worst_expressive_2q(
    landscape: TotalLandscapeResult, k: int = 3
) -> None:
    """Run VQE over 2 qubits with an expressive ansatz, \
        from the best `k` landscape points and the worst `k` landscape points, \
        and display the result.

    Args:
        landscape (TotalLandscapeResult): The energy landscape.
        k (int, optional): number of starting points to take from the best and worst (each). Defaults to 3.
    """
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
    """Run VQE with an non-expressive ansatz and shifted MUBs, \
        from the best `k` landscape points and the worst `k` landscape points, \
        and display the result.

    Args:
        landscape (TotalLandscapeResult): The energy landscape.
        ansatz (QuantumCircuit): the parametric, non-expressive ansatz.
        k (int, optional): number of starting points to take from the best and worst (each). Defaults to 3.
    """
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


def run_and_display_vqe_best_vs_random_expressive_2q(landscape: TotalLandscapeResult, k: int=3) -> None:
    """Pick a set of random theta vectors, find the $k$ that get the best expectation value in the expressive ansatz, \
        and compare their VQE to those the best $k$ landscape points.

    Args:
        landscape (TotalLandscapeResult): The energy landscape.
        k (int, optional): number of starting points to take from the best and worst (each). Defaults to 3.
    """
    best_points = find_k_best_points(landscape, k)
    params = get_expressive_2q_params(landscape.n_qubits, get_exact_ground(landscape.op), True)
    mub_initial_thetas = [params_MUB_2q(point.index.basis_state_idx, point.index.mub_idx) for point in best_points]
    ansatz = gen_expressive_ansatz_2qubits()
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
    """Pick a set of random theta vectors, find the $k$ that get the best expectation value with the ansatz, \
        and compare their VQE to those the best $k$ shifted-MUB landscape points.

    Args:
        landscape (TotalLandscapeResult): The energy landscape.
        ansatz: the parametric, non-expressive ansatz.
        k (int, optional): number of starting points to take from the best and worst (each). Defaults to 3.
    """
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
    

### MY MUB Utils - I.Ram 2025

def gen_expressive_ansatz_2qubits() -> QuantumCircuit:
    NUM_THETAS = 8
    thetas = [Parameter(f"theta{i}") for i in range(NUM_THETAS)]
    circ = QuantumCircuit(2)
    circ.ry(thetas[0], 0)
    circ.ry(thetas[1], 1)
    circ.rz(thetas[2], 0)
    circ.rz(thetas[3], 1)
    circ.cx(0, 1)
    circ.ry(thetas[4], 0)
    circ.ry(thetas[5], 1)
    circ.rz(thetas[6], 0)
    circ.rz(thetas[7], 1)
    return circ

def params_MUB_2q(state_idx, mub_idx):
    # state_idx chooses the state in the basis (MUB)
    # mub_idx chooses the basis (MUB) itself
    if mub_idx == 0:
        if state_idx == 0:
            return [0, 0, 0, 0, 0, 0, 0, 0]
        elif state_idx == 1:
            return [0, 0, 0, 0, np.pi, 0, 0, 0]
        elif state_idx == 2:
            return [0, 0, 0, 0, 0, np.pi, 0, 0]
        elif state_idx == 3:
            return [0, 0, 0, 0, np.pi, np.pi, 0, 0]
    if mub_idx == 1:
        if state_idx == 0:
            return [0, 0, 0, 0, np.pi / 2, np.pi / 2, 0, 0]
        elif state_idx == 1:
            return [0, -np.pi, 0, 0, np.pi / 2, np.pi / 2, 0, 0]
        elif state_idx == 2:
            return [np.pi, 0, 0, 0, np.pi / 2, np.pi / 2, 0, 0]
        elif state_idx == 3:
            return [np.pi, -np.pi, 0, 0, np.pi / 2, np.pi / 2, 0, 0]
    elif mub_idx == 2:
        if state_idx == 0:
            return [np.pi / 2, -np.pi / 2, -np.pi/2, np.pi/2, 0, 0, 0, 0]
        elif state_idx == 1:
            return [-np.pi / 2, np.pi / 2, -np.pi/2, np.pi/2, 0, 0, 0, 0]
        elif state_idx == 2:
            return [np.pi / 2, np.pi / 2, -np.pi/2, np.pi/2, 0, 0, 0, 0]
        elif state_idx == 3:
            return [-np.pi / 2, -np.pi / 2, -np.pi/2, np.pi/2, 0, 0, 0, 0]
    elif mub_idx == 3:
        if state_idx == 0:
            return [0, 0, np.pi/2, np.pi/2, np.pi/2, np.pi/2, -np.pi/2, -np.pi/2]
        elif state_idx == 1:
            return [0, 0, np.pi/2, np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]
        elif state_idx == 2:
            return [0, 0, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2]
        elif state_idx == 3:
            return [0, 0, -np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2]
    elif mub_idx == 4:
        if state_idx == 0:
            return [np.pi/2, -np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0, 0]
        elif state_idx == 1:
            return [-np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0, 0]
        elif state_idx == 2:
            return [np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0, 0]
        elif state_idx == 3:
            return [np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0, 0]

#####

from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import Parameter

def get_mub_ansatz(num_qubits, ansatz_template = None, MUB_size = 2, MUB_mask = None):
    ansatz_combined, _ = get_mub_ansatz_and_thetas(num_qubits, ansatz_template, MUB_size, MUB_mask)
    return ansatz_combined

def get_mub_ansatz_and_thetas(num_qubits, ansatz_template = None, MUB_size = 2, MUB_mask = None, state_idx=0, mub_idx=0):
    #TODO - utilize MUB_mask to decide which are the MUB subset-qubits
    assert num_qubits >= MUB_size

    if ansatz_template == None:
        ansatz_template = EfficientSU2

    ansatz1 = ansatz_template(num_qubits - MUB_size)

    assert MUB_size == 2
    ansatz2 = gen_expressive_ansatz_2qubits()

    # Create a new circuit with the total number of qubits
    ansatz_combined = QuantumCircuit(num_qubits)

    if MUB_mask == None:
        # Append the ansatz circuits
        ansatz_combined.append(ansatz1, range(num_qubits - MUB_size))  # Append ansatz1 to the first N qubits
        ansatz_combined.append(ansatz2, range(num_qubits - MUB_size, num_qubits)) #Append ansatz2 to the rest of the qubits
    else: #I know who to put MUB on:
        the_rest_qubits = [index for index in range(num_qubits) if index not in MUB_mask]
        ansatz_combined.append(ansatz1, the_rest_qubits)
        ansatz_combined.append(ansatz2, MUB_mask)

    
    ### Decide Thetas
    initial_values_ansatz1 = np.random.rand(len(ansatz1.parameters))  # Random initial values for ansatz1 TODO- can change to not be random
    initial_values_ansatz2 = params_MUB_2q(state_idx, mub_idx)  # Random initial values for ansatz2

    initial_thetas = {}

    # Populate the dictionary with parameters from ansatz1
    for param, value in zip(ansatz1.parameters, initial_values_ansatz1):
        initial_thetas[param] = value

    # Populate the dictionary with parameters from ansatz2
    for param, value in zip(ansatz2.parameters, initial_values_ansatz2):
        initial_thetas[param] = value
    
    ###

    return ansatz_combined, initial_thetas

from vqe import run_VQE_simple

def run_VQE_MUB(H, min_eigenvalue, energy_values, theta_path, state_idx=0, mub_idx=0, MUB_mask = None):
    mub_ansatz, initial_thetas = get_mub_ansatz_and_thetas(H.num_qubits, state_idx=state_idx, mub_idx=mub_idx, MUB_mask = MUB_mask)
    vqe_result = run_VQE_simple(H, energy_values, theta_path, initial_thetas=initial_thetas ,min_eigenvalue=min_eigenvalue, ansatz=mub_ansatz, maxiter=1000, seed=42)
    return vqe_result

import itertools

def run_VQE_MUB_on_subset(H, min_eigenvalue, mub_state_mask_list = None, MAX_ITER=100):
    if mub_state_mask_list == None:
        mub_state_mask_list = list(itertools.product(range(5), range(4), generate_all_subsets(2, H.num_qubits)))
    
    n = 0
    n_correct = 0
    for mub_idx, state_idx, MUB_mask in mub_state_mask_list:
        print(f"ITERATION {n} === MUB VQE STATE (mub_idx={mub_idx}, state_idx={state_idx}, MUB_mask={MUB_mask})")
        result = run_VQE_MUB(H, min_eigenvalue, energy_values=[], theta_path=[], state_idx=state_idx, mub_idx=mub_idx, MUB_mask=MUB_mask)
        if abs(result.optimal_value - min_eigenvalue) < 3:
            n_correct += 1
            print("FOUND GLOBAL MINIMUM")
        n += 1

        if n > MAX_ITER:
            break
    
    print(f"===== SUCCESS RATE FOR GLOBAL MINIMUM {n_correct}/{n}={n_correct / n * 100}%")
    return n_correct, n

def run_VQE_MUB_for_all_mubs_on_one_pair_2q(H, min_eigenvalue, MAX_ITER=100, mub_and_state_idx_list = None, MUB_mask = None):
    if mub_and_state_idx_list == None:
        mub_and_state_idx_list = list(itertools.product(range(5), range(4)))
    
    mub_list = [(a[0], a[1], b) for a, b in list(itertools.product(mub_and_state_idx_list, [MUB_mask]))]
    return run_VQE_MUB_on_subset(H, min_eigenvalue, mub_list, MAX_ITER)

def run_VQE_MUB_for_all_choose_2q(H, min_eigenvalue, MAX_ITER=100, mub_and_state_idx_list = None):
    if mub_and_state_idx_list == None:
        mub_and_state_idx_list = list(itertools.product(range(5), range(4)))
    
    mub_list = [(a[0], a[1], b) for a, b in list(itertools.product(mub_and_state_idx_list, generate_all_subsets(2, H.num_qubits)))]
    return run_VQE_MUB_on_subset(H, min_eigenvalue, mub_list, MAX_ITER)

import random

def run_VQE_MUB_random(H, min_eigenvalue, MAX_ITER=100):
    elements = set()  # Use a set to avoid repetitions

    while len(elements) < MAX_ITER:
        a = random.randint(0, 4)  # a is between 0 and 4
        b = random.randint(0, 3)  # b is between 0 and 3
        c = random.randint(0, H.num_qubits - 1)  # Get two unique numbers between 1 and H.num_qubits
        d = random.randint(0, c - 1)
        
        elements.add((a, b, (d, c)))  # Add the tuple to the set
    
    mub_list = list(elements)
    #print(mub_list)
    return run_VQE_MUB_on_subset(H, min_eigenvalue, mub_list, MAX_ITER)

def run_VQE_stats(H, min_eigenvalue, N = 10, maxiter = 1000):
    n_correct = 0
    for n in range(N):
        seed = 42 + n
        print(f"ITERATION {n} - seed = {seed}")
        result = run_VQE_simple(H, min_eigenvalue=min_eigenvalue, energy_values=[], theta_path=[], seed=seed, maxiter=1000)

        if abs(result.optimal_value - min_eigenvalue) < 3:
            n_correct += 1
            print("FOUND GLOBAL MINIMUM")
    
    print(f"===== SUCCESS RATE FOR GLOBAL MINIMUM {n_correct}/{N}={n_correct / N * 100}%")
    return n_correct, N