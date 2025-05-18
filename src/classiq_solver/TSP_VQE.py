# Full Classiq-style Hamiltonian for TSP using Qmod in Python
import classiq
classiq.authenticate()
from classiq.core.model import QuantumModel
from classiq.core.qmod import QuantumVariable, Z, I
from itertools import product

# Problem parameters
N = 4
num_qubits = (N - 1) ** 2
A = 10 * 6  # For example, max(W.values()) * len(E), adjust as needed

# Define edges and weights as in your example
E = {(0, 1), (0, 3), (0, 2), (1, 2), (1, 3), (2, 3)}
W = {
    (0, 1): 1,
    (0, 3): 10,
    (1, 2): 9,
    (2, 3): 5,
    (0, 2): 4,
    (1, 3): 3,
}

# Define quantum variables indexed by vertex and time
qubits = [[QuantumVariable(f"q_{v}_{j}") for j in range(N - 1)] for v in range(N - 1)]

# Identity operator
identity = I

# --- Hamiltonian terms ---

# H_vertices_TSP
H_vertices = A * (N - 1) * (1 - (N - 1) + ((N - 1) ** 2) / 4) * identity
for v in range(N - 1):
    for j in range(N - 1):
        H_vertices += (A * (2 - (N - 1)) / 2) * Z(qubits[v][j])

for v in range(N - 1):
    for j in range(N - 1):
        for i in range(N - 1):
            if j != i:
                H_vertices += (A / 4) * Z(qubits[v][j]) * Z(qubits[v][i])

# H_time_TSP
H_time = A * (N - 1) * (1 - (N - 1) + ((N - 1) ** 2) / 4) * identity
for j in range(N - 1):
    for v in range(N - 1):
        H_time += (A * (2 - (N - 1)) / 2) * Z(qubits[v][j])

for j in range(N - 1):
    for v in range(N - 1):
        for u in range(N - 1):
            if v != u:
                H_time += (A / 4) * Z(qubits[v][j]) * Z(qubits[u][j])

# H_pen_TSP
H_pen = 0
for u, v in product(range(N - 1), repeat=2):
    if u != v and ((u + 1, v + 1) not in E and (v + 1, u + 1) not in E):
        for j in range(N - 2):
            H_pen += (A / 4) * (
                Z(qubits[u][j]) * Z(qubits[v][(j + 1) % (N - 1)])
                - Z(qubits[u][j])
                - Z(qubits[v][(j + 1) % (N - 1)])
                + identity
            )

# H_Pen_depot_TSP
H_pen_depot = 0
for v in range(N - 1):
    if (0, v + 1) not in E and (v + 1, 0) not in E:
        H_pen_depot += (A / 2) * (
            -Z(qubits[v][0])
            - Z(qubits[v][N - 2])
            + 2 * identity
        )

# H_weight_TSP
H_weight = 0
for (u, v) in E:
    if u != 0 and v != 0:
        for j in range(N - 2):
            Wuv = W[(u, v)]
            H_weight += (Wuv / 4) * (
                Z(qubits[u - 1][j]) * Z(qubits[v - 1][(j + 1) % (N - 1)])
                - Z(qubits[u - 1][j])
                - Z(qubits[v - 1][(j + 1) % (N - 1)])
                + identity
            )
            H_weight += (Wuv / 4) * (
                Z(qubits[v - 1][j]) * Z(qubits[u - 1][(j + 1) % (N - 1)])
                - Z(qubits[v - 1][j])
                - Z(qubits[u - 1][(j + 1) % (N - 1)])
                + identity
            )

# H_Weight_depot_TSP
H_weight_depot = 0
for (u, v) in E:
    if u == 0:
        W0v = W[(0, v)]
        H_weight_depot += (W0v / 2) * (
            -Z(qubits[v - 1][0])
            - Z(qubits[v - 1][N - 2])
            + 2 * identity
        )

# Sum all terms for total Hamiltonian
H_total = H_vertices + H_time + H_pen + H_pen_depot + H_weight + H_weight_depot

# --- Define the quantum model ---
model = QuantumModel(name="TSP_Hamiltonian")
model.add_operator(H_total)

# Compile the model (optional, depending on SDK usage)
model.compile()

# For demonstration: print model info
print(f"Model name: {model.name}")
print(f"Number of qubits: {num_qubits}")
print("Hamiltonian constructed as a sum of algebraic operators.")

# If you want, export or continue with synthesis etc.