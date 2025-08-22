import csv
import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory for saving graphs
output_dir = "good graphs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all CSV files in the current directory
csv_files = [f for f in os.listdir('csva') if f.endswith('.csv') and not "2d" in f]

if not csv_files:
    print("No CSV files found in the current directory.")
    exit()

# Store all data for the big subplot figure
all_figures = []
all_titles = []
all_coeffs = []
all_indices = []
all_ok_coeffs = []
all_ok_indices = []
all_notok_coeffs = []
all_notok_indices = []
all_unique_solutions = []
all_coeff_meanings = []

for csv_filename in csv_files:
    print(f"\nProcessing file: {csv_filename}")

    # Ask user for the meaning of the 'coeff' column
    coeff_meaning = input(f"Please enter the meaning of the 'coeff' column in '{csv_filename}': ").strip()
    if not coeff_meaning:
        coeff_meaning = "Coefficient"

    data = []
    with open(csv_filename, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # Try to find the correct column names
        # Always expect: coeff, min_eig_index, status
        for row in reader:
            try:
                coeff_value = float(row["coeff"])
            except KeyError:
                print(f"Could not find 'coeff' column in {csv_filename}. Skipping file.")
                data = []
                break
            try:
                min_eig_index = int(row["min_eig_index"])
            except KeyError:
                print(f"Could not find 'min_eig_index' column in {csv_filename}. Skipping file.")
                data = []
                break
            try:
                status = str(row["status"])
            except KeyError:
                print(f"Could not find 'status' column in {csv_filename}. Skipping file.")
                data = []
                break
            data.append({
                "coeff": coeff_value,
                "min_eig_index": min_eig_index,
                "status": status
            })

    if not data:
        print(f"No data found in {csv_filename}.")
        continue

    # Separate data by status for better plotting
    ok_coeff = [d["coeff"] for d in data if d["status"] == "OK"]
    ok_index = [d["min_eig_index"] for d in data if d["status"] == "OK"]
    notok_coeff = [d["coeff"] for d in data if d["status"] != "OK"]
    notok_index = [d["min_eig_index"] for d in data if d["status"] != "OK"]

    # For background lines: unique solutions
    unique_solutions = {}
    for d in data:
        key = (d["min_eig_index"], d["status"])
        if key not in unique_solutions:
            unique_solutions[key] = []
        unique_solutions[key].append(d["coeff"])

    # Store for subplot
    all_coeffs.append([d["coeff"] for d in data])
    all_indices.append([d["min_eig_index"] for d in data])
    all_ok_coeffs.append(ok_coeff)
    all_ok_indices.append(ok_index)
    all_notok_coeffs.append(notok_coeff)
    all_notok_indices.append(notok_index)
    all_unique_solutions.append(unique_solutions)
    all_coeff_meanings.append(coeff_meaning)
    all_titles.append(f"{os.path.splitext(csv_filename)[0]}")

    # Plot and save individual figure
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot the main line
    all_coeff = [d["coeff"] for d in data]
    all_index = [d["min_eig_index"] for d in data]
    plt.plot(all_coeff, all_index, color='royalblue', linewidth=2, label='Min Eigenvalue Index')

    # Scatter OK and NOT OK points
    plt.scatter(ok_coeff, ok_index, color='green', s=20, marker='o', label='OK Solution')
    plt.scatter(notok_coeff, notok_index, color='red', s=20, marker='x', label='NOT OK Solution')

    # Add horizontal lines for unique solutions
    for (idx, status), coeffs in unique_solutions.items():
        color = 'green' if status == 'OK' else 'red'
        label = f'idx={idx} ({status})'
        plt.axhline(y=idx, xmin=0, xmax=1, color=color, linestyle='--', linewidth=1, alpha=0.3, label=label)

    # Remove duplicate labels in legend
    handles, labels_ = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12, loc='best', frameon=True)

    plt.xlabel(coeff_meaning, fontsize=16, fontweight='bold', color='#333366', labelpad=15)
    plt.ylabel('Minimum Eigenvalue Index', fontsize=16, fontweight='bold', color='#333366', labelpad=15)
    plt.title(f'Min Eigenvalue Index vs {coeff_meaning}', fontsize=16, weight='bold')
    plt.tight_layout()

    # Add grid, minor ticks, and background
    plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
    plt.minorticks_on()

    # Save the figure
    save_path = os.path.join(output_dir, f"{os.path.splitext(csv_filename)[0]}_graph.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved graph to {save_path}")
    plt.close()

# Now, create a big figure with all the small graphs as subplots
num_files = len(all_coeffs)
if num_files > 0:
    # Choose grid size
    ncols = min(3, num_files)
    nrows = (num_files + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows))
    plt.style.use('seaborn-v0_8-darkgrid')
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.reshape(axes, (nrows, ncols))

    for idx in range(num_files):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col] if nrows > 1 else axes[col] if ncols > 1 else axes[0, 0]
        coeffs = all_coeffs[idx]
        indices = all_indices[idx]
        ok_coeff = all_ok_coeffs[idx]
        ok_index = all_ok_indices[idx]
        notok_coeff = all_notok_coeffs[idx]
        notok_index = all_notok_indices[idx]
        unique_solutions = all_unique_solutions[idx]
        coeff_meaning = all_coeff_meanings[idx]
        title = all_titles[idx]

        # Main line
        ax.plot(coeffs, indices, color='royalblue', linewidth=2, label='Min Eigenvalue Index')
        # OK/NOT OK
        ax.scatter(ok_coeff, ok_index, color='green', s=20, marker='o', label='OK Solution')
        ax.scatter(notok_coeff, notok_index, color='red', s=20, marker='x', label='NOT OK Solution')
        # Horizontal lines
        for (sol_idx, status), sol_coeffs in unique_solutions.items():
            color = 'green' if status == 'OK' else 'red'
            label = f'idx={sol_idx} ({status})'
            ax.axhline(y=sol_idx, xmin=0, xmax=1, color=color, linestyle='--', linewidth=1, alpha=0.3, label=label)
        # Remove duplicate labels in legend
        handles, labels_ = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best', frameon=True)
        ax.set_xlabel(coeff_meaning, fontsize=12, fontweight='bold', color='#333366', labelpad=10)
        ax.set_ylabel('Min Eigenvalue Index', fontsize=12, fontweight='bold', color='#333366', labelpad=10)
        ax.set_title(title, fontsize=13, weight='bold')
        ax.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
        ax.minorticks_on()

    # Hide unused subplots
    for idx in range(num_files, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    big_save_path = os.path.join(output_dir, "all_graphs_subplot.png")
    plt.savefig(big_save_path, bbox_inches='tight')
    print(f"Saved combined subplot graph to {big_save_path}")
    plt.show()
