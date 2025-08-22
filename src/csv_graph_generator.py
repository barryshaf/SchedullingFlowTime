import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def load_csv_data(csv_filename):
    """Load data from CSV file using standard library"""
    print(f"📊 Loading data from {csv_filename}...")
    
    data = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                'finish_jobs': float(row['finish_jobs']),
                'finish_time_1': float(row['finish_time_1']),
                'min_eig_index': int(row['min_eig_index']),
                'min_eig_value': float(row['min_eig_value']),
                'status': row['status']
            })
    
    print(f"   - Loaded {len(data)} data points")
    print(f"   - Columns: finish_jobs, finish_time_1, min_eig_index, min_eig_value, status")
    return data

def reshape_data_for_plotting(data):
    """Reshape the data for 2D plotting"""
    # Get unique values for x and y axes
    finish_jobs_values = sorted(list(set(row['finish_jobs'] for row in data)))
    finish_time_1_values = sorted(list(set(row['finish_time_1'] for row in data)))
    
    print(f"   - finish_jobs range: {min(finish_jobs_values):.2f} to {max(finish_jobs_values):.2f}")
    print(f"   - finish_time_1 range: {min(finish_time_1_values):.2f} to {max(finish_time_1_values):.2f}")
    
    # Create 2D arrays for plotting
    X, Y = np.meshgrid(finish_jobs_values, finish_time_1_values, indexing='ij')
    
    # Reshape the data into 2D arrays
    min_eig_indices = np.zeros((len(finish_jobs_values), len(finish_time_1_values)), dtype=int)
    min_eig_values = np.zeros((len(finish_jobs_values), len(finish_time_1_values)), dtype=float)
    status_matrix = np.empty((len(finish_jobs_values), len(finish_time_1_values)), dtype=object)
    
    # Fill the arrays
    for i, fj in enumerate(finish_jobs_values):
        for j, ft in enumerate(finish_time_1_values):
            # Find the row that matches these coordinates
            for row in data:
                if row['finish_jobs'] == fj and row['finish_time_1'] == ft:
                    min_eig_indices[i, j] = row['min_eig_index']
                    min_eig_values[i, j] = row['min_eig_value']
                    status_matrix[i, j] = row['status']
                    break
    
    return X, Y, min_eig_indices, min_eig_values, status_matrix

def create_contour_plot(X, Y, min_eig_indices, status_matrix, output_filename=None):
    """Create the contour plot with status markers"""
    print("🎨 Creating contour plot...")
    
    # Find all unique indices for contour levels
    unique_indices = sorted(set(min_eig_indices.flatten()))
    print(f"   - Found {len(unique_indices)} unique eigenvalue indices")
    
    plt.figure(figsize=(12, 8))
    
    # Draw contour lines for the minimal eigenvalue index
    contour_lines = plt.contour(X, Y, min_eig_indices, levels=unique_indices, 
                               colors='black', linewidths=1.2)
    plt.clabel(contour_lines, fmt='%d', fontsize=10, colors='black')
    
    # Overlay OK/NOT OK points
    for status, marker, color in [("OK", "o", "green"), ("NOT OK", "x", "red")]:
        status_mask = (status_matrix == status)
        if status_mask.any():
            plt.scatter(
                X[status_mask], Y[status_mask],
                marker=marker, color=color, s=30, label=f"{status} Solution"
            )
    
    plt.xlabel('finish_jobs', fontsize=14)
    plt.ylabel('finish_time_1', fontsize=14)
    plt.title('Contour Lines of Min Eigenvalue Index\nvs finish_jobs and finish_time_1', fontsize=16)
    plt.legend()
    plt.tight_layout()
    
    if output_filename:
        output_dir = "good graphs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # If output_filename is just a filename, prepend the directory
        if not os.path.isabs(output_filename):
            output_filename = os.path.join(output_dir, output_filename)
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"   - Saved plot to {output_filename}")
    
    plt.show()

def analyze_data(data):
    """Analyze the data and print statistics"""
    print("\n📈 Data Analysis:")
    print("="*50)
    
    # Count statuses
    status_counts = {}
    unique_indices = set()
    
    for row in data:
        status = row['status']
        status_counts[status] = status_counts.get(status, 0) + 1
        unique_indices.add(row['min_eig_index'])
    
    print("Status distribution:")
    for status, count in status_counts.items():
        print(f"   - {status}: {count} points")
    
    # Count unique eigenvalue indices
    unique_indices = sorted(unique_indices)
    print(f"\nUnique eigenvalue indices: {len(unique_indices)}")
    print(f"Index range: {min(unique_indices)} to {max(unique_indices)}")
    
    # Show some sample solutions
    print("\nSample solutions by status:")
    for status in ['OK', 'NOT OK']:
        status_indices = set()
        for row in data:
            if row['status'] == status:
                status_indices.add(row['min_eig_index'])
        if status_indices:
            sample_indices = sorted(list(status_indices))[:3]
            print(f"   - {status}: {sample_indices}")

def main():
    """Main function to load CSV and create graph"""
    print("🚀 CSV Graph Generator")
    print("="*50)
    
    # CSV filename to use
    csv_filename = "csvs/2d_levelset_min_eig_status_new.csv"
    
    try:
        # Load the data
        data = load_csv_data(csv_filename)
        
        # Analyze the data
        analyze_data(data)
        
        # Reshape data for plotting
        X, Y, min_eig_indices, min_eig_values, status_matrix = reshape_data_for_plotting(data)
        
        # Create the plot
        create_contour_plot(X, Y, min_eig_indices, status_matrix, 
                           output_filename="csv_generated_graph.png")
        
        print("\n✅ Graph generation completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_filename}")
        print("Please make sure the CSV file exists in the current directory.")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 