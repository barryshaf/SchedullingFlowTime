import numpy as np
from qiskit.circuit.classical.types import ordering
from qiskit.quantum_info import SparsePauliOp
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt


# Define identity and Z matrices as Pauli strings
Z = 'Z'
I = 'I'

# Function to generate SparsePauliOp
def kronecker_product(operators, coeff, N):
    pauli_string = ''.join(operators)
    return SparsePauliOp.from_list([(pauli_string, coeff)], num_qubits=N)

class J():
    def __init__(self, r_j, p_j): #r_j is the release time, p_j is the processing time
        self.r_j = r_j
        self.p_j = p_j
        #print(f"Job created: release_time={r_j}, processing_time={p_j}")

class scheduling_flowtime_problem():
    def __init__(self, M, J):
        self.M = M
        self.J = J
        self.T = np.sum([J.p_j for J in J])+np.max([J.r_j for J in J])-1
        self.num_qubits = M * len(J) * (self.T + 1) + len(J) * (self.T + 1)
        self.term = SparsePauliOp.from_list([], num_qubits=self.num_qubits)
        
        # Print initialization info
        #print(f"\n=== Scheduling Problem Initialized ===")
        #print(f"Number of machines (M): {self.M}")
        #print(f"Number of jobs: {len(self.J)}")
        #print(f"Total time horizon (T): {self.T}")
        #print(f"Total qubits needed: {self.num_qubits}")
        #print(f"Job details:")
        # for i, job in enumerate(self.J):
        #     print(f"  Job {i}: r_j={job.r_j}, p_j={job.p_j}")
        print("="*40)

    def print_solution(self, binary_solution):
        print("Quantum State (computational basis):", '|' + binary_solution + '>')
        print("\n📊 SOLUTION:")
        
        # Assignment variables X_{m,j,t}
        print("Assignment variables X_{m,j,t}:")
        for j in range(len(self.J)):
            for t in range(self.T+1):  # Only up to T-1, not T+1
                for m in range(self.M):
                    idx = self.get_assignment_index(m, j, t)
                    value = binary_solution[idx]
                    print(f"  X_{{{m},{j},{t}}} = {value}, idx={idx}")
        
        # Completion variables C_{j,t}
        print("Completion variables C_{j,t}:")
        for j in range(len(self.J)):
            for t in range(self.T+1):  # Completion variables exist from 0 to T
                idx = self.get_completion_index(j, t)
                value = binary_solution[idx]
                print(f"  C_{{{j},{t}}} = {value}, idx={idx}")
        print()

    def get_indicator_index(self, m, j, t):
        """Get the qubit index for machine m, job j, time t"""
        # index = t+self.T*j+self.T*self.J*m
        # return index

        return m + (self.M+1) * j + (self.M+1) * len(self.J) * t
    
    def get_assignment_index(self, m, j, t):
        """For x_{m,j,t} variables (assignment of job j to machine m at time t)"""
        index = self.get_indicator_index(m, j, t)
        return index

    def get_completion_index(self, j, t):
        """For c_{j,t} variables (completion time variables)"""
        index = self.get_indicator_index(self.M, j, t)  # Use M as fake machine index
        return index
    
    def get_hamiltonian(self, penalty_coeff,no_early_start=1,score=1,finish_time_2=1,finish_time_1=1,finish_all_jobs=10,no_overlap=10):
        self.term = SparsePauliOp.from_list([], num_qubits=self.num_qubits)
        #All conditions
        self.H_no_early_start(no_early_start*penalty_coeff)
        self.H_score(score/((self.T)**2*len(self.J)))
        self.H_finish_time_2(finish_time_2*penalty_coeff)
        self.H_finish_time_1_new(finish_time_1*penalty_coeff)
        self.H_fiinish_all_jobs(finish_all_jobs*penalty_coeff)
        self.H_no_overlap(no_overlap*penalty_coeff)

        return self.term
    def H_no_early_start(self, penalty_coeff):
        """
        Constraint: ∝ - ∑ Z^m_{j,t} for j∈[J], t<r_j, m∈[M]
        Penalizes jobs starting before their release time r_j
        """
        #print(f"\n--- Building H_no_early_start (penalty_coeff={penalty_coeff}) ---")
        terms_added = 0
        
        for j_idx, job in enumerate(self.J):
            r_j = job.r_j  # Release time of job j
            #print(f"Processing job {j_idx} with release time {r_j}")
            for t in range(min(r_j, self.T)):  # For all t < r_j
                for m in range(self.M):
                    operators = [I] * self.num_qubits
                    operators[self.get_assignment_index(m, j_idx, t)] = Z
                    # Negative coefficient to penalize early start
                    self.term += kronecker_product(operators, -1 * penalty_coeff, self.num_qubits)
                    terms_added += 1
        
        #print(f"H_no_early_start: Added {terms_added} terms")
        #print(f"Current total terms in Hamiltonian: {len(self.term)}")
    
    def H_score(self, penalty_coeff):
        """
        Objective function: = - ∑∑ t · Z^c_{j,t}
        Minimizes completion times by rewarding earlier completions
        Double sum over j (jobs) and t (time)
        """
        #print(f"\n--- Building H_score (penalty_coeff={penalty_coeff}) ---")
        terms_added = 0
        
        for j_idx, job in enumerate(self.J):
            for t in range(self.T+1):  # Sum from t=0 to t=T
                operators = [I] * self.num_qubits
                operators[self.get_completion_index(j_idx, t)] = Z
                # Negative coefficient: earlier completion times get higher reward
                self.term += kronecker_product(operators, -1 * (t+1) * penalty_coeff, self.num_qubits)
                #print(f"Score numnum {j_idx} at time {t}, mekadem={-1 * (t+1) * penalty_coeff}")
                terms_added += 1


                operators2 = [I] * self.num_qubits
                self.term += kronecker_product(operators2, (t+1) * penalty_coeff, self.num_qubits)
                terms_added += 1
        
        #print(f"H_score: Added {terms_added} terms")
        #print(f"Current total terms in Hamiltonian: {len(self.term)}")

    def H_finish_time_2(self, penalty_coeff):
        #print(f"\n--- Building H_finish_time_2 (penalty_coeff={penalty_coeff}) ---")
        terms_added = 0
        
        for j_idx, job in enumerate(self.J):
            ##print(f"Processing finish time constraint 2 for job {j_idx}")
            # First part: (1 - T/2) ∑[t] Z^c_{j,t}
            for t in range(self.T+1):
                operators = [I] * self.num_qubits
                operators[self.get_completion_index(j_idx, t)] = Z
                self.term += kronecker_product(operators, (1-(self.T+1)/2) * penalty_coeff, self.num_qubits)
                terms_added += 1

            # Second part: 1/4 ∑[t,t'] Z^c_{j,t} Z^c_{j,t'}
            for t in range(self.T+1):
                for t_prime in range(t+1, self.T+1):  # t_prime >= t to get unique pairs only
                    operators = [I] * self.num_qubits
                    operators[self.get_completion_index(j_idx, t)] = Z      # First Z
                    operators[self.get_completion_index(j_idx, t_prime)] = Z  # Second Z
                    self.term += kronecker_product(operators, 0.5 * penalty_coeff, self.num_qubits)
                    terms_added += 1
        
        #print(f"H_finish_time_2 old: Added {terms_added} terms")
        #print(f"Current total terms in Hamiltonian: {len(self.term)}")

    def H_finish_time_1(self, penalty_coeff):
        """
        Complex finish time constraint involving both completion and assignment variables:
        = (1/2) ∑[j∈[J], t] [(M²t²/2 - p_j² + p_j Mt) Z^c_{j,t} + (p_j - Mt/2) ∑[m,t'<t] Z^x_{m,j,t'} 
                              + (Mt/2 - p_j) Z^c_{j,t} ∑[m,t'<t] Z^x_{m,j,t'} + (1/4) ∑[m,m',t',t''] (1 - Z^c_{j,t} Z^x_{m,j,t'} Z^x_{m,j,t''})]
        """
        terms_added = 0
        for j_idx, job in enumerate(self.J):
            p_j = job.p_j  # Processing time of job j
            
            for t in range(self.T+1):  # Sum from t=0 to t=T
                # Term 1: (M²t²/2 - p_j² + p_j Mt) Z^c_{j,t}
                coeff_1 = (self.M**2 * (t+1)**2 / 2 - p_j**2 + p_j * self.M * (t+1)) * 0.5 * penalty_coeff
                operators = [I] * self.num_qubits
                operators[self.get_completion_index(j_idx, t)] = Z
                self.term += kronecker_product(operators, coeff_1, self.num_qubits)
                terms_added += 1
                ##print(f"Term1 j={j_idx}, t={t}")

                # Term 2: (p_j - Mt/2) ∑[m,t'<t] Z^x_{m,j,t'}
                coeff_2 = (p_j - self.M * (t+1) / 2) * 0.5 * penalty_coeff
                for m in range(self.M):
                    for t_prime in range(t+1):  # t' <= t 
                        operators = [I] * self.num_qubits
                        operators[self.get_assignment_index(m, j_idx, t_prime)] = Z
                        self.term += kronecker_product(operators, coeff_2, self.num_qubits)
                        terms_added += 1
                        ##print(f"Term2 j={j_idx}, t={t}, t'={t_prime} m={m}")


                # Term 3: (Mt/2 - p_j) Z^c_{j,t} ∑[m,t'<t] Z^x_{m,j,t'}
                coeff_3 = (self.M * (t+1) / 2 - p_j) * 0.5 * penalty_coeff
                for m in range(self.M):
                    for t_prime in range(t+1): #t<=t because it's better with qubits
                        operators = [I] * self.num_qubits
                        operators[self.get_completion_index(j_idx, t)] = Z
                        operators[self.get_assignment_index(m, j_idx, t_prime)] = Z
                        self.term += kronecker_product(operators, coeff_3, self.num_qubits)
                        terms_added += 1
                        ##print(f"Term3 j={j_idx}, t={t}, t'={t_prime} m={m}")
                # Term 4: (1/4) ∑[m,m',t',t''] (1 - Z^c_{j,t} Z^x_{m,j,t'} Z^x_{m,j,t''})
                for m in range(self.M):
                    for m_prime in range(self.M):
                        for t_prime in range(t+1):  # t' <= t
                            for t_double_prime in range(t+1):  # t'' >= t' and t'' < t
                                # Triple product term: -1/4 * Z^c_{j,t} Z^x_{m,j,t'} Z^x_{m',j,t''}
                                if (t_prime!=t_double_prime or m!=m_prime):
                                    operators = [I] * self.num_qubits
                                    operators[self.get_completion_index(j_idx, t)] = Z
                                    operators[self.get_assignment_index(m, j_idx, t_prime)] = Z
                                    operators[self.get_assignment_index(m_prime, j_idx, t_double_prime)] = Z
                                    self.term += kronecker_product(operators, -0.25 * 0.5 * penalty_coeff, self.num_qubits)
                                    terms_added += 1
                                    #Term5
                                    operators = [I] * self.num_qubits
                                    operators[self.get_assignment_index(m, j_idx, t_prime)] = Z
                                    operators[self.get_assignment_index(m_prime, j_idx, t_double_prime)] = Z
                                    self.term += kronecker_product(operators, 0.25 * 0.5 * penalty_coeff, self.num_qubits)
                                    terms_added += 1    
                                else:
                                    operators = [I] * self.num_qubits
                                    operators[self.get_completion_index(j_idx, t)] = Z
                                    self.term += kronecker_product(operators, -0.25 * 0.5 * penalty_coeff, self.num_qubits)
                                    terms_added += 1                            
        #print(f"H_finish_time_1: Added {terms_added} terms")
        #print(f"Current total terms in Hamiltonian: {len(self.term)}")

    def H_no_overlap(self, penalty_coeff):
        """
        No overlap constraint: ensures no two jobs are assigned to the same machine at the same time.
        H = (1/4) * [∑[m,t,j≠j'] Z^x_{m,j,t} Z^x_{m,j',t} - 2(J-1) ∑[m,t,j] Z^x_{m,j,t}]
        """
        terms_added = 0
        
        # First term: ∑[m,t,j≠j'] Z^x_{m,j,t} Z^x_{m,j',t}
        for m in range(self.M):
            for t in range(self.T + 1):
                for j_idx in range(len(self.J)):
                    for j_prime_idx in range(j_idx + 1, len(self.J)):  # j ≠ j'
                        operators = [I] * self.num_qubits
                        operators[self.get_assignment_index(m, j_idx, t)] = Z
                        operators[self.get_assignment_index(m, j_prime_idx, t)] = Z
                        self.term += kronecker_product(operators, 0.5 * penalty_coeff, self.num_qubits)
                        #print(f"first added j={j_idx} j'={j_prime_idx} t={t}")
                        terms_added += 1
        
        # Second term: -2(J-1) ∑[m,t,j] Z^x_{m,j,t}
        for m in range(self.M):
            for t in range(self.T + 1):
                for j_idx in range(len(self.J)):
                    operators = [I] * self.num_qubits
                    operators[self.get_assignment_index(m, j_idx, t)] = Z
                    self.term += kronecker_product(operators, -0.5 * (len(self.J) - 1) * penalty_coeff, self.num_qubits)
                    terms_added += 1
                    #print(f"second added j={j_idx} t={t}")
        #print(f"H_no_overlap: Added {terms_added} terms")
        #print(f"Current total terms in Hamiltonian: {len(self.term)}")



    def H_fiinish_all_jobs(self, penalty_coeff):
        terms_added = 0
        for j_idx, job in enumerate(self.J):
            p_j = job.p_j
            # First term: (p_j - MT/2) * sum_{m,t} Z^x_{m,j,t}
            coeff1 = p_j - (self.M * (self.T+1)) / 2
            for m in range(self.M):
                for t in range(self.T + 1):
                    operators = [I] * self.num_qubits
                    operators[self.get_assignment_index(m, j_idx, t)] = Z
                    self.term += kronecker_product(operators, coeff1 * penalty_coeff, self.num_qubits)
                    terms_added += 1

            # Second term: (1/4) * sum_{m,m',t,t'} Z^x_{m,j,t} Z^x_{m',j,t'}
            for m in range(self.M):
                for m_prime in range(self.M):
                    for t in range(self.T + 1):
                        for t_prime in range(self.T + 1):
                            operators = [I] * self.num_qubits
                            if (t!=t_prime or m!=m_prime):
                                operators[self.get_assignment_index(m, j_idx, t)] = Z
                                operators[self.get_assignment_index(m_prime, j_idx, t_prime)] = Z
                            self.term += kronecker_product(operators, 0.25 * penalty_coeff, self.num_qubits)
                            terms_added += 1
        #print(f"H_fiinish_all_jobs: Added {terms_added} terms")



    def H_finish_time_1_new(self, penalty_coeff):
        # Implements the constraint:
        # (1/4) * [ sum_{m,j,t,t'>t} Z^x_{m,j,t'} Z^c_{j,t} - M*(T-t) sum_{j,t} Z^c_{j,t} - T sum_{m,j,t'>t} Z^x_{m,j,t'} ]
        #print(f"\n--- Building H_finish_time_1_new (penalty_coeff={penalty_coeff}) ---")
        terms_added = 0
        M = self.M
        T = self.T

        # First term: sum_{m,j,t,t'>t} Z^x_{m,j,t'} Z^c_{j,t}
        for m in range(M):
            for j in range(len(self.J)):
                for t in range(T+1):
                    for t_prime in range(t+1, T+1):  # t' > t
                        operators = [I] * self.num_qubits
                        operators[self.get_assignment_index(m, j, t_prime)] = Z
                        operators[self.get_completion_index(j, t)] = Z
                        self.term += kronecker_product(operators, 0.25 * penalty_coeff, self.num_qubits)
                        terms_added += 1

        # Second term: -M*(T-t) sum_{j,t} Z^c_{j,t}
        for j in range(len(self.J)):
            for t in range(T+1):
                operators = [I] * self.num_qubits
                operators[self.get_completion_index(j, t)] = Z
                coeff = -0.25 * M * (T+1 - t-1) * penalty_coeff
                self.term += kronecker_product(operators, coeff, self.num_qubits)
                terms_added += 1

        # Third term: -T sum_{m,j,t'>t} Z^x_{m,j,t'}
        for m in range(M):
            for j in range(len(self.J)):
                for t in range(T+1):
                    for t_prime in range(t+1, T+1):  # t' > t
                        operators = [I] * self.num_qubits
                        operators[self.get_assignment_index(m, j, t_prime)] = Z
                        coeff = -0.25 * penalty_coeff
                        self.term += kronecker_product(operators, coeff, self.num_qubits)
                        terms_added += 1

        #print(f"H_finish_time_1_new: Added {terms_added} terms")
        #print(f"Current total terms in Hamiltonian: {len(self.term)}")

        terms_added = 0
        
def get_pauli_matrix(pauli):
    if pauli == 'I':
        return np.array([1,1])  # Identity matrix
    elif pauli == 'Z':
        return np.array([1,-1])  # Pauli Z matrix
    else:
        raise ValueError(f"Unknown Pauli operator: {pauli}")

def get_diagonal_kronker(sparse_pauli_op):
    # Get the Pauli strings and their coefficients
    pauli_strings = sparse_pauli_op.paulis
    coefficients = sparse_pauli_op.coeffs

    # Initialize the aggregate vector as the identity operator
    aggregate_vector = np.array((2**len(sparse_pauli_op.paulis[0]))*[0],dtype=np.complex128)  # Identity matrix for the initial Kronecker product
    
    # Calculate the aggregate Kronecker products
    for coeff, pauli_string in zip(coefficients, pauli_strings):
        if coeff == 0:
            continue

        # Start with the identity matrix for this specific Pauli string
        pauli_product = np.array([1])  # Identity matrix for the initial Kronecker product

        # Iterate over each character in the Pauli string
        for pauli in pauli_string[::-1]: #For some reason the pauli string is in the reverse direction than what it should be
            # Get the matrix representation of the current Pauli operator
            pauli_matrix = get_pauli_matrix(str(pauli))
            
            # Perform the Kronecker product with the current Pauli matrix
            pauli_product = np.kron(pauli_product, pauli_matrix)
        ##print(f"{coeff}*{pauli_string}->{pauli_product}")

        # Scale the resulting product by its coefficient
        scaled_pauli_product = coeff * pauli_product
        
        # Aggregate the result
        aggregate_vector += scaled_pauli_product
        ##print(aggregate_vector)
        
    return np.array(aggregate_vector).reshape((-1,1))

def get_diagonal_kronker_torch(sparse_pauli_op, device='auto'):
    """
    GPU/CPU-accelerated diagonal computation using PyTorch.
    
    Args:
        sparse_pauli_op: SparsePauliOp object
        device: 'auto', 'cpu', 'cuda', or 'mps' (for Apple Silicon)
    
    Returns:
        numpy array of diagonal elements
    """
    import torch
    
    # Auto-detect device if requested
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon
        else:
            device = 'cpu'
    
    #print(f"🔧 Using device: {device}")
    
    pauli_strings = sparse_pauli_op.paulis
    coefficients = sparse_pauli_op.coeffs
    num_qubits = len(sparse_pauli_op.paulis[0])
    
    #print(f"📊 Computing diagonal for {num_qubits} qubits ({2**num_qubits} states)")
    
    # Move to selected device
    diagonal = torch.zeros(2**num_qubits, dtype=torch.complex128, device=device)
    
    # Create all binary states efficiently
    states = torch.arange(2**num_qubits, device=device)
    binary_states = torch.stack([(states >> i) & 1 for i in range(num_qubits)], dim=1)
    
    #print(f"🔄 Processing {len(pauli_strings)} Pauli strings...")
    
    # Process each Pauli string
    for i, (coeff, pauli_string) in enumerate(zip(coefficients, pauli_strings)):
        if coeff == 0:
            continue
        
        # Initialize eigenvalues for this Pauli string
        eigenvalues = torch.ones(2**num_qubits, dtype=torch.complex128, device=device)
        
        # Compute eigenvalues for all states at once
        for qubit_idx, pauli in enumerate(pauli_string):
            if pauli == 'Z':
                bit_values = binary_states[:, qubit_idx]
                eigenvalues *= torch.where(bit_values == 0, 1, -1)
        
        # Add contribution to diagonal
        diagonal += coeff * eigenvalues
        
        # Progress indicator for large problems
        #if i % 100 == 0 and i > 0:
            #print(f"   Processed {i}/{len(pauli_strings)} Pauli strings")
    
    #print(f"✅ Diagonal computation complete")
    
    # Move back to CPU and convert to numpy
    return diagonal.cpu().numpy()