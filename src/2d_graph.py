from src.scheduling_flowtime import J, scheduling_flowtime_problem
from src.scheduling_flowtime import get_diagonal_kronker
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def test_basic_functionality():
    """Test basic functionality with a simple example"""
    print("🚀 Testing Scheduling Flowtime Problem")
    print("="*50)
    
    # Create some sample jobs
    print("📋 Creating sample jobs...")
    jobs = [
        J(r_j=0, p_j=2),  
        J(r_j=1, p_j=1), 
    ]
    # Create scheduling problem with 2 machines
    print("\n🏭 Creating scheduling problem...")
    M = 1  # Number of machines
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
    try:
        # Define the ranges for the two coefficients
        finish_jobs_range = np.linspace(0, 10, 50)
        finish_time_2_range = np.linspace(0, 10, 50)

        # Prepare a 2D array to store the minimal eigenvalue indices
        min_eig_indices = np.zeros((len(finish_jobs_range), len(finish_time_2_range)), dtype=int)
        min_eig_values = np.zeros((len(finish_jobs_range), len(finish_time_2_range)), dtype=float)

        # Run the tests
        problem = test_basic_functionality()

        # For storing which indices are "OK" or "NOT OK"
        index_status = {}

        # First pass: compute all minimal eigenvalue indices and values
        for i, a1 in enumerate(finish_jobs_range):
            for j, a2 in enumerate(finish_time_2_range):
                H = problem.get_hamiltonian(
                    penalty_coeff=10.0,
                    finish_all_jobs=a1,
                    finish_time_2=a2
                )
                diagonal = get_diagonal_kronker(H)
                min_eigenvalue_index = int(np.argmin(diagonal))  # Find INDEX of minimum value
                min_eig_indices[i, j] = min_eigenvalue_index
                print(f"j={j},i={i},min_eigenvalue_index={min_eigenvalue_index}")

        # Find all unique indices
        unique_indices = set(min_eig_indices.flatten())
        print("\n🔎 Printing solutions for each unique minimal eigenvalue index found:")

        # For each unique index, print the solution and ask for user input
        for idx in unique_indices:
            binary_solution = format(idx, f'0{problem.num_qubits}b')
            print(f"\n--- Solution for index {idx} ---")
            problem.print_solution(binary_solution)
            user_input = input(f"Is this solution OK? (y/n): ").strip().lower()
            # Handle special indices with hardcoded status, otherwise ask user
            if idx == 33728:
                index_status[idx] = "OK"
            elif idx == 39936 or idx == 53248:
                index_status[idx] = "NOT OK"
            else:
                if user_input == 'y':
                    print(f"  ✅ Computational basis: |{binary_solution}>")
                    index_status[idx] = "OK"
                else:
                    print(f"  🔴 (red dot) Solution flagged as NOT OK")
                    index_status[idx] = "NOT OK"

        # Prepare a status matrix for plotting
        status_matrix = np.empty_like(min_eig_indices, dtype=object)
        for i in range(len(finish_jobs_range)):
            for j in range(len(finish_time_2_range)):
                idx = min_eig_indices[i, j]
                status_matrix[i, j] = index_status.get(idx, "UNKNOWN")

        # Plotting the 2D level set (contour) of minimal eigenvalue index
        import matplotlib.pyplot as plt

        X, Y = np.meshgrid(finish_jobs_range, finish_time_2_range, indexing='ij')

        plt.figure(figsize=(12, 8))
        # Draw contour lines (קווי גובה) for the minimal eigenvalue index
        # Use a discrete set of levels for the unique indices
        levels = sorted(unique_indices)
        contour_lines = plt.contour(X, Y, min_eig_indices, levels=levels, colors='black', linewidths=1.2)
        plt.clabel(contour_lines, fmt='%d', fontsize=10, colors='black')

        # Get all OK indices
        ok_indices = [idx for idx, status in index_status.items() if status == "OK"]
        not_ok_indices = [idx for idx, status in index_status.items() if status == "NOT OK"]
        
        # Define different markers and colors for OK indices
        ok_markers = ['o']
        ok_colors = ['green', 'limegreen', 'lime']
        
        # Plot OK solutions with different markers for each index
        for i, idx in enumerate(ok_indices):
            marker = ok_markers[i % len(ok_markers)]
            color = ok_colors[i % len(ok_colors)]
            ok_mask = (min_eig_indices == idx)
            plt.scatter(
                X[ok_mask], Y[ok_mask],
                marker=marker, color=color, s=50, 
                label=f"OK Solution (Index {idx})"
            )

        # Plot NOT OK solutions with red X
        for idx in not_ok_indices:
            not_ok_mask = (min_eig_indices == idx)
            plt.scatter(
                X[not_ok_mask], Y[not_ok_mask],
                marker='x', color='red', s=50, 
                label=f"NOT OK Solution (Index {idx})"
            )

        plt.xlabel('finish_jobs', fontsize=14)
        plt.ylabel('finish_time_2', fontsize=14)
        plt.title('Contour Lines of Min Eigenvalue Index\nvs finish_jobs and finish_time_2', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # Save to CSV

        os.makedirs("csvs", exist_ok=True)
        csv_filename = os.path.join("csvs", "2d_levelset_min_eig_status_new_fin2.csv")
        with open(csv_filename, mode='w', newline='') as csvfile:
            fieldnames = ["finish_jobs", "finish_time_2", "min_eig_index", "min_eig_value", "status"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, a1 in enumerate(finish_jobs_range):
                for j, a2 in enumerate(finish_time_2_range):
                    idx = min_eig_indices[i, j]
                    value = min_eig_values[i, j]
                    status = index_status.get(idx, "UNKNOWN")
                    writer.writerow({
                        "finish_jobs": float(a1),
                        "finish_time_2": float(a2),
                        "min_eig_index": int(idx),
                        "min_eig_value": float(value),
                        "status": status
                    })
        print(f"Saved results to {csv_filename}")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()    