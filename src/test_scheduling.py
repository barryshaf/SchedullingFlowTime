from src.scheduling_flowtime import J, scheduling_flowtime_problem
from src.scheduling_flowtime import get_diagonal_kronker
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

def is_matrix_diagonal(matrix, tolerance=1e-10):
    """Check if a matrix is diagonal within a given tolerance"""
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if all off-diagonal elements are close to zero
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and abs(matrix[i, j]) > tolerance:
                return False
    return True

def binary_to_kronker(binary_string):
    n = len(binary_string)
    # Convert binary string to decimal
    decimal_index = int(binary_string, 2)
    # Create a one-hot encoded vector of size 2^n
    one_hot_vector = [0] * (2 ** n)
    one_hot_vector[decimal_index] = 1
    return np.array(one_hot_vector).reshape((2 ** n, 1))

def test_basic_functionality():
    """Test basic functionality with a simple example"""
    print("🚀 Testing Scheduling Flowtime Problem")
    print("="*50)
    
    # Create some sample jobs
    print("📋 Creating sample jobs...")
    jobs = [
        J(r_j=1, p_j=1),  
        J(r_j=0, p_j=2), 
    ]
    # Create scheduling problem with 2 machines
    print("\n🏭 Creating scheduling problem...")
    M = 2  # Number of machines
    problem = scheduling_flowtime_problem(M, jobs)
    
    print(f"\n📊 Problem Statistics:")
    print(f"   - Total processing time: {sum(job.p_j for job in jobs)}")
    print(f"   - Latest release time: {max(job.r_j for job in jobs)}")
    print(f"   - Time horizon (T): {problem.T}")
    print(f"   - Total qubits: {problem.num_qubits}")
    
    return problem
def kronker_vector_to_binary(vector):
    # Find the index of the non-zero element
    index = np.argmax(vector)
    
    # Convert the index to binary representation
    n = int(np.log2(len(vector)))  # Calculate the number of qubits
    binary_representation = format(index, f'0{n}b')  # Format index as binary with leading zeros
    
    # Create the quantum state string
    quantum_state = ''.join(binary_representation)
    
    return quantum_state


if __name__ == "__main__":
    import time
    start_time = time.time()
    try:
        problem = test_basic_functionality()
        H = problem.get_hamiltonian(penalty_coeff=10.0)

        # Get diagonal elements  
        diagonal = get_diagonal_kronker(H)
        
        #Find minimal eigenvalue
        min_eigenvalue_index = np.argmin(diagonal)  # Find INDEX of minimum value
        min_eigenvalue = diagonal[min_eigenvalue_index]  # Get the actual minimum value
        
        # Convert the index to binary representation (the optimal quantum state)
        binary_solution = format(min_eigenvalue_index, f'0{problem.num_qubits}b')

        print("Minimum Eigenvalue:", min_eigenvalue.real)
        print(f"Optimal state index: {min_eigenvalue_index}")
        print(f"Binary representation: {binary_solution}")
        problem.print_solution(binary_solution)

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n⏱️ Total execution time: {elapsed_time:.4f} seconds")