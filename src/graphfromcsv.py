import csv
import matplotlib.pyplot as plt
import numpy as np
import os

csv_filename = "csvs/2d_levelset_min_eig_status_new_fin2.csv"
# Read the CSV and extract the data
finish_jobs = []
finish_time_2 = []
min_eig_index = []
min_eig_value = []
status = []

with open(csv_filename, mode='r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        finish_jobs.append(float(row["finish_jobs"]))
        finish_time_2.append(float(row["finish_time_2"]))
        min_eig_index.append(int(row["min_eig_index"]))
        min_eig_value.append(float(row["min_eig_value"]))
        status.append(row["status"])

# Convert to numpy arrays and reshape to 2D grids
# Infer grid shape
unique_finish_jobs = sorted(list(set(finish_jobs)))
unique_finish_time_2 = sorted(list(set(finish_time_2)))
n_a1 = len(unique_finish_jobs)
n_a2 = len(unique_finish_time_2)

finish_jobs_grid = np.array(finish_jobs).reshape((n_a1, n_a2))
finish_time_2_grid = np.array(finish_time_2).reshape((n_a1, n_a2))
min_eig_index_grid = np.array(min_eig_index).reshape((n_a1, n_a2))
min_eig_value_grid = np.array(min_eig_value).reshape((n_a1, n_a2))
status_grid = np.array(status).reshape((n_a1, n_a2))

X, Y = finish_jobs_grid, finish_time_2_grid

# Find all unique indices and their status
unique_indices = set(min_eig_index)
index_status = {}
for idx, stat in zip(min_eig_index, status):
    index_status[idx] = stat  # last occurrence wins, but should be consistent

ok_indices = [idx for idx, stat in index_status.items() if stat == "OK"]
not_ok_indices = [idx for idx, stat in index_status.items() if stat == "NOT OK"]

# Plotting
plt.figure(figsize=(12, 8))
levels = sorted(set(min_eig_index))
contour_lines = plt.contour(X, Y, min_eig_index_grid, levels=levels, colors='black', linewidths=1.2)
#plt.clabel(contour_lines, fmt='%d', fontsize=10, colors='black')

# Define different markers and colors for OK indices
ok_markers = ['o']
ok_colors = ['green','limegreen','lime'] 

# Plot OK solutions with different markers for each index
for i, idx in enumerate(ok_indices):
    marker = ok_markers[i % len(ok_markers)]
    color = ok_colors[i % len(ok_colors)]
    ok_mask = (min_eig_index_grid == idx)
    plt.scatter(
        X[ok_mask], Y[ok_mask],
        marker=marker, color=color, s=50, 
        label=f"OK Solution (Index {idx})"
    )

# Plot NOT OK solutions with red X
for idx in not_ok_indices:
    not_ok_mask = (min_eig_index_grid == idx)
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
