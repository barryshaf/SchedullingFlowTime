# Quantum-exhaustive-search
Methods to do quantum exhaustive search.
This repository also includes a lot of infrastructural "utility" code that can be used for other VQE/MUB related papers and projects.

# File Description
## Code Utility Files
-   `hamiltonians.py`: generation of different Hamiltonians (as qiskit `SparsePauliOp`).
-   `vqe.py`: performing VQE on any required Hamiltonian.
-   `landscape.py`: calculation of the energy landscape of a Hamiltonian.
-   `graphing.py`: generation of displayable graphs. Energy landscape, energy value histograms, and graphs.
-   `maxcut.py`: utilities for maxcut problems.
-   `mub_state_gen.py`: generation of states of MUBs.
-   `ansatz.ipynb`: Contains the code to generate all of the experiment asnatzes (expressive 1-2 qubits, and hardware-efficient for >2 qubits).
It is a notebook because it includes the formulas for the parameters of the 2-qubit expressive ansatz. 
-   `experiment_utils.py`: Definitions of specific experiments.


## Experiment Notebooks
### `collective_experiments.ipynb`
-   Aggregation of experiments on all types of input Hamiltonians.
-   only includes the experiments we wish to put in the paper.
-   should act as the most updated version.

## Documentation files
-   `thoughts.md`: internal record of reserach directions.
-   `thoughts_on_graphing_method.md`: ideas on how to best present the results of energy landscapes and VQE runs.
-   `exp_requirements.md`: the list of required experiments for the VQE-MUB paper.