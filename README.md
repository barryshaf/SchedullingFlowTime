# Flowtime Scheduling – Final Project

**Authors:** Barry Shafran & Nimrod Gutt  
**Institution:** Faculty of Physics, Technion – Israel Institute of Technology  
**Date:** August 22, 2025

---

## Abstract
This project explores the **Flowtime Job Shop Scheduling Problem (JSP)** in the context of the XPRIZE Quantum Computing competition. We define the flowtime JSP with preemption, review classical baselines (e.g., **Shortest Job First (SJF)** and **Shortest Processing Remaining Time (SPRT)**), and formulate the problem as a **Quadratic Unconstrained Binary Optimization (QUBO)** Hamiltonian that is then mapped to an **Ising** model for quantum-oriented optimization. We evaluate small instances (up to ~24 qubits) and analyze phase-diagram behavior as Hamiltonian coefficients vary, highlighting how constraint weights influence feasibility and solution quality.

---

## Problem (Brief)
- **Jobs:** Each job *j* has processing time \(p_j\) and release time \(r_j\).  
- **Machines:** Assign jobs to machines with possible **preemption**.  
- **Objective (Flowtime):** Minimize \(\sum_j (C_j - r_j)\), where \(C_j\) is completion time of job *j*.  
- **Key constraints:** No processing before release time, machine exclusivity, exactly one completion time per job, and no work after completion.

---

## Method
### QUBO / Hamiltonian
- Encode scheduling decisions in binary variables.
- Write objective and constraints as quadratic forms; square any potentially negative term to ensure non-negativity.
- Total Hamiltonian
We combine the objective and all constraint penalties into a single Hamiltonian:

$$
H_{\text{total}} \=\ \lambda_{0}\, H_{\text{obj}} \+\ \sum_{k=1}^{K} \lambda_{k}\, H_{k}^{\text{(constraint)}},
$$

where each $(\lambda_{k} > 0\)$ controls the strength of the \(k\)-th constraint penalty relative to the objective term $\(H_{\text{obj}}\).h \(\lambda_{k} > 0\)$ controls the strength of the \(k\)-th constraint penalty relative to the objective term $\(H_{\text{obj}}\)$.

### Ising Mapping
- Map binary variables to spin variables via the standard \(\{0,1\}\leftrightarrow\{-1,+1\}\) transformation.
- Implement using **Pauli \(Z\)** and identity **\(I\)** operators; the resulting Hamiltonian is **diagonal**, so the ground state can be identified by directly inspecting diagonal entries (small instances).

---

## Results
We solved one-machine cases with up to three jobs efficiently by sweeping release times. Scaling the number of machines/jobs quickly became infeasible under available resources.

**Runtime vs. Qubits**

| Qubits | Time (s)  |
|:------:|----------:|
|   4    |   0.0298  |
|  12    |   0.045   |
|  16    |   0.258   |
|  18    |   1.299   |
|  24    | 100.132   |
|  27    | 1349.796  |
|  34    | MemoryError |

The empirical curve indicates **exponential growth** in runtime with qubit count, consistent with expectations for brute-force eigenstate search on diagonal Hamiltonians.

---

## Figures
> Add the corresponding images (or notebooks) to the repository and link them here.

- **Figure 1.** Preemptive vs. non-preemptive schedules for two jobs on one machine; both yield the same total flowtime but distribute waiting differently.  
- **Figure 3.** Runtime as a function of qubit count (shows exponential growth).  
- **Figures 4–6.** Phase diagrams illustrating solution sensitivity to Hamiltonian coefficient choices (constraint weights).

---

## Repository
Code and materials: **https://github.com/barryshaf/SchedullingFlowTime**

---

## Reference
Bansal, N. (2003). *Algorithms for Flow Time Scheduling* (Publication No. 3121052) [Doctoral dissertation, Carnegie Mellon University]. ProQuest Dissertations & Theses Global.

---

## How to Cite
If you use this project, please cite the repository and the dissertation above.

