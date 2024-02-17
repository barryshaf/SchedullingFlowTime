from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.primitives import Estimator

import numpy as np

class Parameters:
    def __init__(self, n_qubits: int, n_layers: int, optimizer: str,
                    tol: float, success_bound: float, max_iter: int,
                    report_period: int, report_thetas: bool,
                    num_of_starting_points: int,
                    exact_result: float | None = None):
        self.n_qubits = n_qubits            # number of qubist in the operator
        self.n_layers = n_layers            # number of ansatz layers
        self.optimizer = optimizer          # name of optimizer
        self.tol = tol                      # optimizer "tolerance" parameter. Semantic meaning differs between optimizers.
        self.success_bound = success_bound  # distance from exact result that we consider as success.
        self.max_iter = max_iter            # iteration bound
        self.report_period = report_period  # how many iterations before a callback print.
        self.report_thetas = report_thetas  # whether to show the paramneter vector at the callback print
        self.exact_result = exact_result    # the exact result that we compare ourselves to.
        self.num_of_starting_points = num_of_starting_points  # What amount of the best initial points is tested



def get_standard_params(n_qubits: int) -> Parameters:
    return Parameters(n_qubits=n_qubits, n_layers=5, optimizer='COBYLA', tol=1e-6,
                        success_bound=1e-3, max_iter=1000, report_period=20, report_thetas=False,
                        num_of_starting_points=5)



def run_vqe_experiment(hamiltonian: SparsePauliOp, ansatz: QuantumCircuit, initial_state: QuantumCircuit, params: Parameters) -> tuple[int, float, bool]:
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
        raise Exception('Optimizer not supported in this experiment!~')
    ansatz_with_prepened_mub = initial_state.compose(ansatz, range(ansatz.num_qubits), inplace=False)

    # enforcing the success bound
    class BoundHitException(Exception):
        def __init__(self, n_evals, final_cost):
            self.n_evals = n_evals
            self.final_cost = final_cost

    def callback_fun(eval_count: int, theta: np.ndarray, cost: float, metadata: dict) -> None:
        if params.exact_result is None:
            return
        if (eval_count % params.report_period == 0):
            print(f"{eval_count}: {cost}")
            if (params.report_thetas):
                print(f"thetas: {theta}")
        if (cost < params.exact_result + params.success_bound):
            raise BoundHitException(eval_count, cost)
        
    try:
        vqe_obj = VQE(estimator=estimator_obj, ansatz=ansatz_with_prepened_mub, optimizer=optimizer_obj, callback=callback_fun, initial_point = [0.0]*ansatz.num_parameters)
        res = vqe_obj.compute_minimum_eigenvalue(operator=hamiltonian)
        return res.cost_function_evals, res.optimal_value, False
    except BoundHitException as e:
        return e.n_evals, e.final_cost, True