"""
Module for VQE experiments, hyper-parameters and results.
"""

from dataclasses import dataclass
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.primitives import Estimator

THROW_ON_SUCCESS = False

class Parameters:
    """
    Class containing the hyper-parameters of a VQE experiment.
    """
    # If someone wants to, they can turn this into a dataclass.
    def __init__(self, n_qubits: int, n_layers: int, optimizer: str,
                    tol: float, success_bound: float, max_iter: int,
                    report_period: int, report_thetas: bool,
                    num_of_starting_points: int,
                    exact_result: float | None = None,
                    record_progress: bool = False):
        self.n_qubits = n_qubits            # number of qubist in the operator
        self.n_layers = n_layers            # number of ansatz layers
        self.optimizer = optimizer          # name of optimizer
        self.tol = tol                      # optimizer "tolerance" parameter. Semantic meaning differs between optimizers.
        self.success_bound = success_bound  # distance from exact result that we consider as success.
        self.max_iter = max_iter            # iteration bound
        self.report_period = report_period  # how many iterations before a callback print.
        self.report_thetas = report_thetas  # whether to show the paramneter vector at the callback print
        self.ground_energy = exact_result    # the exact result that we compare ourselves to.
        self.num_of_starting_points = num_of_starting_points  # What amount of the best initial points is tested
        self.record_progress = record_progress # whether to return a list of all thetas from the VQE experiment
        
@dataclass
class MyVQEResult:
    """
    Dataclass containing the results from a VQE experiment.
    """
    n_evals: int                        # Number of cost evaluations
    final_cost: np.float64              # Final expectation value
    erminated_on_success_bound: bool    # Whether the optimization ended because of my manual success bound
    costs_list_included: bool           # Whether the function evalulation list exists
    costs_list: list[np.float64]        # The list of function evalulations
    thetas_list_included: bool          # Whether the theta vectors are included in the result
    thetas_list: list[np.array]         # The list of theta vectors from each function evaluation
    desc: str = ""                      # Description string for the presentation.


def get_standard_params(n_qubits: int) -> Parameters:
    """Get the usual VQE experiment hyperparameters, depending on the # of qubits.

    Args:
        n_qubits (int): number of qubits in the VQE run. Affects the tol, # of iterations, etc.

    Returns:
        Parameters: The hyperparameter object.
    """
    # These values were found manually, because the VQE was stuck after reaching the goal.
    if n_qubits == 2:  
        tol = 1e-2
    elif n_qubits == 3:
        tol = 1e-2
    elif n_qubits == 4:
        tol = 5e-2
    elif n_qubits == 8:
        tol = 1e-2
    else:
        tol = 1e-6
    return Parameters(n_qubits=n_qubits, n_layers=n_qubits, optimizer='COBYLA', tol=tol,
                        success_bound=1e-3, max_iter=500*n_qubits, report_period=10000, report_thetas=False,
                        num_of_starting_points=5)



def run_vqe_experiment(hamiltonian: SparsePauliOp,
                       ansatz: QuantumCircuit,
                       initial_thetas: list[np.float64] | None,
                       prepened_state_circ: QuantumCircuit | None,
                       params: Parameters,
                       desc: str = ""
                       ) -> MyVQEResult:
    """Run a VQE expreiment.

    Args:
        hamiltonian (SparsePauliOp): Hamiltonian that we wish to solve.
        ansatz (QuantumCircuit): *parametric* ansatz for the VQE.
        initial_thetas (list[np.float64] | None): Initial vector for the ansatz. \
            If defined as None, the first parameter vector is the zero vector.
        prepened_state_circ (QuantumCircuit | None): Circuit to be added to the regular ansatz. \
            This is useful when trying to run from a specific (but shifted) MUB state. \
            Contact Ittay/Tal/Dekel for an explanation on shifted MUBs.
        params (Parameters): VQE hyperparameters.
        desc (str, optional): string description to add to the result. Defaults to "".

    Raises:
        Exception: When the optimizer hyper-parameter is not supported.
        BoundHitException: When the VQE comes close enough to the pre-calculated exact result. \
            Should always be handled in-function.

    Returns:
        MyVQEResult: Structure containing the result.
    """
    # preparing the VQE components
    estimator_obj = Estimator()  # Internal qiskit structure
    optimizer_obj = None
    if params.optimizer == 'COBYLA':
        optimizer_obj = COBYLA(
            maxiter = params.max_iter,
            tol = params.tol
        )
    elif params.optimizer == 'BFGS':
        optimizer_obj = L_BFGS_B(
            maxiter = params.max_iter,
            tol = params.tol
        )
    else:
        raise Exception('This optimizer is not supported in this experiment!')
    if prepened_state_circ is not None:
        ansatz = prepened_state_circ.compose(ansatz, range(ansatz.num_qubits), inplace=False)

    thetas_list = []
    cost_list = []
    # Manual Success Bound
    # Sometimes, when we run VQE and know what the exact result is going to be,
    # we want to stop the optimizer from running on and on when it reached the end but doesn't know it.
    # `scipy` (the package behind qiskit that handles optimization) lets us handle a manual callback function after each cost eval.
    # However, there is no graceful way to stop the execution using this callback.
    # The only option is to throw an exception and catch it outside of the optimizer.
    # This works on *some* OSes, but usually breaks with Macs and with Jupyter notbeooks.
    # For the VQE-MUB paper, we did not use this feature, but I'm leaving it in for future use.
    class BoundHitException(StopIteration):
        def __init__(self, n_evals, final_cost):
            self.n_evals = n_evals
            self.final_cost = final_cost

    def callback_fun(eval_count: int, theta: np.ndarray, cost: float, metadata: dict) -> None:
        # This function records the parameter vector and cost (if the hyperparams say to do so),
        # prints the result every once in a while and raises a BoundHitException if the option is enabled.
        if params.record_progress:
            thetas_list.append(theta)
            cost_list.append(cost)
        if params.ground_energy is None:
            return
        if (eval_count % params.report_period == 0):
            print(f"{eval_count}: {cost}")
            if (params.report_thetas):
                print(f"thetas: {theta}")
        if THROW_ON_SUCCESS and (cost < params.ground_energy + params.success_bound):
            raise BoundHitException(eval_count, cost)

    # default is a zeroed parameter vector.
    if initial_thetas is None:
        initial_thetas = [0.0]*ansatz.num_parameters
    vqe_obj = VQE(estimator=estimator_obj, ansatz=ansatz, optimizer=optimizer_obj, callback=callback_fun, initial_point = initial_thetas)

    try:
        res = vqe_obj.compute_minimum_eigenvalue(operator=hamiltonian)
        return MyVQEResult(n_evals=res.cost_function_evals, final_cost=res.optimal_value,
                           erminated_on_success_bound=False,
                           costs_list_included=params.record_progress, costs_list=cost_list,
                           thetas_list_included=params.record_progress, thetas_list=thetas_list, desc=desc)
    
    # If the run was terminated from reaching the success bound, extract the results from inside the exception.
    except BoundHitException as e:
        return MyVQEResult(n_evals=e.n_evals, final_cost=e.final_cost,
                           erminated_on_success_bound=True,
                           costs_list_included=params.record_progress, costs_list=cost_list,
                           thetas_list_included=params.record_progress, thetas_list=thetas_list, desc=desc)


def sample_single_vqe_value(hamiltonian: SparsePauliOp,
                       ansatz: QuantumCircuit,
                       initial_thetas: list[np.float64] | None,
                       prepened_state_circ: QuantumCircuit | None,
                       params: Parameters) -> float:
    """This is a "cheat" function, used to sample the expectation value of an ansatz \
        over some Hamiltonian with a specific parameter vector. \
        For the VQE-MUB paper, this is used to check which random theta vector has the \
        best starting option.
        This can, and should, be optimized.

    Args:
        the same as run_vqe_experiment.

    Raises:
        same.

    Returns:
        float: the expectation value with that vector.
    """
    # preparing the VQE components
    estimator_obj = Estimator()  # Internal qiskit structure
    optimizer_obj = None
    if params.optimizer == 'COBYLA':
        optimizer_obj = COBYLA(maxiter = 1)
    elif params.optimizer == 'BFGS':
        optimizer_obj = L_BFGS_B(maxiter = 1)
    else:
        raise Exception('This optimizer is not supported in this experiment!')
    if prepened_state_circ is not None:
        ansatz = prepened_state_circ.compose(ansatz, range(ansatz.num_qubits), inplace=False)

    val = 0
    def callback_fun(eval_count: int, theta: np.ndarray, cost: float, metadata: dict) -> None:
        val = cost

    if initial_thetas is None:
        initial_thetas = [0.0]*ansatz.num_parameters
    vqe_obj = VQE(estimator=estimator_obj, ansatz=ansatz, optimizer=optimizer_obj, callback=callback_fun, initial_point = initial_thetas)


    vqe_obj.compute_minimum_eigenvalue(operator=hamiltonian)
    return val


########################## 2025 I. Ram - simple VQE running

from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Estimator
import random

def store_energy(energy_values, theta_path, eval_count: int, theta: np.ndarray, cost: float, metadata: dict):
    energy_values.append(cost)
    theta_path.append(theta)

def plot_energies(energy_values, actual_min_eigenvalue=None):
  #Plot
  plt.plot(energy_values, marker='.')
  if actual_min_eigenvalue != None:
    plt.plot([0, len(energy_values)],[actual_min_eigenvalue]*2, "r--")
  plt.title('VQE Optimization Energy per Iteration')
  plt.xlabel('Iteration')
  plt.ylabel('Energy')
  plt.grid()
  plt.show()

def parameter_dict_to_list(parameter_dict, ansatz):
    return [parameter_dict[param] for param in ansatz.parameters]

def run_VQE(H_qub, ansatz = None, initial_thetas = None, callback_func=store_energy, plot_func=plot_energies, maxiter=500, seed=42):
  algorithm_globals.random_seed = seed #42
  random.seed(seed)
  np.random.seed(seed)

  if ansatz == None:
    print("No Ansatz Given, assuming EfficientSU2")
    ansatz = EfficientSU2(H_qub.num_qubits)
  
  if type(initial_thetas) == dict:
      initial_thetas = parameter_dict_to_list(initial_thetas, ansatz)

  # Set up the VQE instance with COBYLA optimizer
  optimizer = COBYLA(maxiter=maxiter)
  estimator_obj = Estimator()  # Internal qiskit structure
  vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=estimator_obj, callback=callback_func, initial_point=initial_thetas)

  # Run the VQE algorithm
  result = vqe.compute_minimum_eigenvalue(H_qub)
  print("Ground state energy:", result.eigenvalue.real)
  print("Optimal parameters:", result.optimal_point)
  #print("Number of iterations:", result.optimizer_evals)
  plot_func()
  return result #.eigenvalue.real
#######

#Visualize Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_path_2d(coordinates, axis1=None,axis2=None):
    coordinates = np.array(coordinates)
    n = len(coordinates[0])
    # Randomly select two axes (dimensions)
    axes = np.random.choice(range(n), size=2, replace=False)

    # Extract the selected axes
    x = coordinates[:, axes[0] if axis1 == None else axis1]
    y = coordinates[:, axes[1] if axis2 == None else axis2]

    # Create a plot
    plt.figure(figsize=(5, 3))
    plt.plot(x, y, marker='.', markersize=3, label='Path')

    colors = cm.viridis(np.linspace(0, 1, coordinates.shape[0]))

    # Plot the path with a color gradient
    for i in range(coordinates.shape[0] - 1):
        plt.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=2)


    # Add markers for all points (smaller markers for intermediate points)
    plt.scatter(x[1:-1], y[1:-1], c='black', s=10, zorder=1)  # Intermediate points

    # Highlight the first and last points with larger markers
    plt.scatter(x[0], y[0], c='red', s=50, label='Start Point', zorder=2)  # First point
    plt.scatter(x[-1], y[-1], c='green', s=50, label='End Point', zorder=2)  # Last point


    plt.title(f'Path in Random 2D Projection of the {n}-Dimensional Space')
    plt.xlabel(f'Dimension {axes[0] if axis1 == None else axis1}')
    plt.ylabel(f'Dimension {axes[1] if axis2 == None else axis2}')
    plt.grid()
    plt.show()

def visualize_path_1d(coordinates, axis=None):
    coordinates = np.array(coordinates)
    n = len(coordinates[0])
    # Randomly select two axes (dimensions)
    axes = np.random.choice(range(n), size=1, replace=False)[0]

    # Extract the selected axes
    y = coordinates[:, axes if axis == None else axis]
    x = range(len(y))

    # Create a plot
    plt.figure(figsize=(5, 3))
    plt.plot(x, y, marker='.', markersize=3, label='Path')

    plt.title(f'Path in Random 1D Projection of the {n}-Dimensional Space')
    plt.xlabel(f'Timestep')
    plt.ylabel(f'Dimension {axes if axis == None else axis}')
    plt.grid()
    plt.show()


def run_VQE_simple(H_qub, energy_values = [], theta_path = [], ansatz = None, min_eigenvalue=None, initial_thetas=None, maxiter: int = 500, seed: int = 42, verbose=True):
    plot_func = lambda: plot_energies(energy_values, min_eigenvalue) if verbose else lambda: None
    return run_VQE(H_qub, ansatz=ansatz, maxiter=maxiter, seed=seed, initial_thetas=initial_thetas, plot_func=plot_func, callback_func= lambda a,b,c,d: store_energy(energy_values, theta_path, a,b,c,d))