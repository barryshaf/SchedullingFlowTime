from src.scheduling_flowtime import J, scheduling_flowtime_problem
from src.scheduling_flowtime import get_diagonal_kronker
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def test_all_coefficients():
    """Test all coefficients systematically to find valid solutions"""
    print("🚀 Testing All Scheduling Coefficients")
    print("="*60)
    
    # Ensure output directories exist
    fig_dir = "good graphs"
    csv_dir = "csvs"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    # Create sample jobs
    print("📋 Creating sample jobs...")
    jobs = [
        J(r_j=0, p_j=2),  
        J(r_j=1, p_j=1), 
    ]
    
    # Create scheduling problem with 1 machine
    print("\n🏭 Creating scheduling problem...")
    M = 1  # Number of machines
    problem = scheduling_flowtime_problem(M, jobs)
    
    print(f"\n📊 Problem Statistics:")
    print(f"   - Total processing time: {sum(job.p_j for job in jobs)}")
    print(f"   - Latest release time: {max(job.r_j for job in jobs)}")
    print(f"   - Time horizon (T): {problem.T}")
    print(f"   - Total qubits: {problem.num_qubits}")
    
    # Define coefficient ranges for each parameter
    coeff_ranges = {
        'no_early_start': np.linspace(0, 15, 150),
        'score': np.linspace(0, 15, 150),
        'finish_time_2': np.linspace(0, 15, 150),
        'finish_time_1': np.linspace(0, 15, 150),
        'finish_all_jobs': np.linspace(0, 15, 150),
        'no_overlap': np.linspace(0, 15, 150)
    }
    
    # Store results for each coefficient type
    all_results = {}
    
    for coeff_name, coeff_values in coeff_ranges.items():
        print(f"\n🔍 Testing {coeff_name} coefficient...")
        print("-" * 40)
        
        min_eig_indices = []
        min_eig_values = []
        
        for coeff_val in coeff_values:
            # Create default parameters
            params = {
                'penalty_coeff': 10.0,
                'no_early_start': 1,
                'score': 1,
                'finish_time_2': 1,
                'finish_time_1': 1,
                'finish_all_jobs': 10,
                'no_overlap': 10
            }
            
            # Update the specific coefficient being tested
            params[coeff_name] = coeff_val
            
            # Get Hamiltonian
            H = problem.get_hamiltonian(**params)
            diagonal = get_diagonal_kronker(H)
            
            # Find minimum eigenvalue
            min_eigenvalue_index = np.argmin(diagonal)
            min_eigenvalue = diagonal[min_eigenvalue_index]
            
            min_eig_indices.append(min_eigenvalue_index)
            min_eig_values.append(min_eigenvalue.real)
            
            if len(min_eig_indices) % 30 == 0:  # Progress indicator
                print(f"  Progress: {len(min_eig_indices)}/{len(coeff_values)}")
        
        all_results[coeff_name] = {
            'coeff_values': coeff_values,
            'min_eig_indices': min_eig_indices,
            'min_eig_values': min_eig_values
        }
        
        print(f"  ✅ Completed {coeff_name}")
    
    # Now analyze solutions for each coefficient type
    print("\n🔎 Analyzing solutions for each coefficient type...")
    
    for coeff_name, results in all_results.items():
        print(f"\n--- Analyzing {coeff_name} ---")
        
        # Find unique minimal eigenvalue indices
        unique_indices = set(results['min_eig_indices'])
        print(f"  Found {len(unique_indices)} unique minimal eigenvalue indices")
        
        # For each unique index, ask user if solution is OK
        index_status = {}
        
        for idx in unique_indices:
            binary_solution = format(idx, f'0{problem.num_qubits}b')
            print(f"\n  --- Solution for index {idx} ---")
            problem.print_solution(binary_solution)
            
            user_input = input(f"  Is this solution OK? (y/n): ").strip().lower()
            
            if user_input == 'y':
                print(f"    ✅ Computational basis: |{binary_solution}>")
                index_status[idx] = "OK"
            else:
                print(f"    🔴 Solution flagged as NOT OK")
                index_status[idx] = "NOT OK"
        
        # Create CSV for this coefficient type
        csv_filename = os.path.join(csv_dir, f"{coeff_name}_coeff_min_eig_status.csv")
        csv_rows = []
        
        for coeff, idx in zip(results['coeff_values'], results['min_eig_indices']):
            status = index_status.get(idx, "UNKNOWN")
            csv_rows.append({
                f"{coeff_name}_coeff": coeff, 
                "min_eig_index": idx, 
                "status": status
            })
        
        # Save to CSV
        with open(csv_filename, mode='w', newline='') as csvfile:
            fieldnames = [f"{coeff_name}_coeff", "min_eig_index", "status"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        
        print(f"  💾 Saved results to {csv_filename}")
        
        # Create plot for this coefficient type
        plt.figure(figsize=(12, 8))
        
        # Plot the main line
        plt.plot(results['coeff_values'], results['min_eig_indices'], 'b-', alpha=0.7, label='Min eigenvalue index')
        
        # Add horizontal lines for each unique index with color coding
        for idx in unique_indices:
            status = index_status.get(idx, "UNKNOWN")
            color = 'green' if status == "OK" else 'red'
            label = f'{"OK" if status == "OK" else "NOT OK"} idx={idx}'
            plt.axhline(y=idx, color=color, alpha=0.6, linewidth=0.5, label=label)
        
        plt.xlabel(f'{coeff_name} coefficient')
        plt.ylabel('Minimal eigenvalue index')
        plt.title(f'Minimal eigenvalue index vs {coeff_name} coefficient')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(fig_dir, f"{coeff_name}_coeff_min_eig_status_graph.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"  📊 Saved plot to {plot_filename}")
    
    # Create summary CSV with all results
    print("\n📋 Creating summary CSV...")
    summary_csv_filename = os.path.join(csv_dir, "all_coeffs_summary.csv")
    
    # Find the maximum length among all coefficient arrays
    max_length = max(len(results['coeff_values']) for results in all_results.values())
    
    summary_rows = []
    for i in range(max_length):
        row = {}
        for coeff_name, results in all_results.items():
            if i < len(results['coeff_values']):
                row[f"{coeff_name}_coeff"] = results['coeff_values'][i]
                row[f"{coeff_name}_min_eig_index"] = results['min_eig_indices'][i]
                row[f"{coeff_name}_min_eig_value"] = results['min_eig_values'][i]
            else:
                row[f"{coeff_name}_coeff"] = ""
                row[f"{coeff_name}_min_eig_index"] = ""
                row[f"{coeff_name}_min_eig_value"] = ""
        
        summary_rows.append(row)
    
    # Save summary CSV
    with open(summary_csv_filename, mode='w', newline='') as csvfile:
        fieldnames = []
        for coeff_name in all_results.keys():
            fieldnames.extend([f"{coeff_name}_coeff", f"{coeff_name}_min_eig_index", f"{coeff_name}_min_eig_value"])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    
    print(f"💾 Saved summary to {summary_csv_filename}")
    print("\n🎉 All coefficient testing completed!")

if __name__ == "__main__":
    try:
        test_all_coefficients()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
