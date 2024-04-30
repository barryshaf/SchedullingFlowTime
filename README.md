# Quantum-exhaustive-search
Test methods to do quantum exhaustive search

# File Description
## Code Utility Files
-   `hamiltonians.py`: generation of different Hamiltonians (as qiskit `SparsePauliOp`).
-   `vqe.py`: performing VQE on any required Hamiltonian.
-   `landscape.py`: calculation of the energy landscape of a Hamiltonian.
-   `graphing.py`: generation of displayable graphs. Energy landscape, energy value histograms, and graphs.
-   `maxcut.py`: utilities for maxcut problems.
-   `mub_state_gen.py`: generation of states of MUBs.
-   `ansatz.ipynb`: Contains the code to generate all of the experiment asnatzes (expressive 1-2 qubits, and hardware-efficient for >2 qubits).
Currently a notebook because it includes the formulas for the parameters of the 2-qubit expressive ansatz. 

## Experiment Notebooks
### `transverse_experiments.ipynb`
-   experiments for VQE on transverse-ising Hamiltonians.
-   Experiments included:
    -   runs on 3 qubits with full MUBs, with landscapes and histograms. (no VQE)
    -   runs on 4-6 qubits qith 3-qubit partial MUBs, with landscapes and histograms. (no VQE)
    -   runs on 3 qubits with full MUBs, with landscapes and histograms **and VQE** from 10 best points.
    -   runs on 4-6 qubits qith 3-qubit partial MUBs, with landscapes and histograms **and VQE** from 10 best points.
    -   Uses the prepended zero-set andsatz method.
-   The bottom two are the ones to be included.
-   Interpretation of the results appears at the bottom of the notebook.
### `molecular_experiments.ipynb`
-   VQE experiments on molecular Hamiltonians.
-   3 pre-generated Hamiltonian entries:
    -   $H_2$ with 0.75 Angstrom. 2 qubits.
    -   $HeH$ with 1 Angstrom. 2 qubits.
    -   $LiH$ with 1.5 Angstrom. 4 qubits.
    -   More Hamiltonians can potentially be generated using Dekel's code, but the quskit-nature dependencies are especially tricky.
-   Experiments included:
    -   runs on 2 qubits with 2-qubit full MUBs for the $H_2$ and $HeH$. Landscaping, histogram and VQE from 10 best points.
    -   Experiment on LiH with 3-qubit partial MUBs. Landscaping, histogram and VQE from 10 best points.
    -   Uses the prepended zero-set andsatz method.
-   Interpretation still lacking.
### `maxcut_experiments.ipynb`
-   VQE experiments on MAXCUT reduction Hamiltonians.
-   num nodes === num qubits.
-   Experiments included:
    -   a triangle graph with full MUBs. Landscaping, histogram and VQE from 10 best points.
    -   From 4 to 8 qubits:
        -   a random graph with a set number of edges.
        -   the number of edges was taken from Dekel's code.
        -   generation of graphs (showing the correct result still lacking)
        -   landscapng, histogram, and VQE from 10 best points.
        -   Uses the prepended zero-set andsatz method.
-   Interpretation still lacking.
### `MUB_molecular_energy.ipynb`
-   Dekel's unmodified code for running VQE experiments on molecular Hamiltonians.
-   Standalone (does not use the utility files).
-   Contains code to generate molecular hamiltonians.
-   Contains code to perform a measurement of an expectation value of a state against an operator. In the other notebooks, we use the `Statevector.expectation_value(<op>)` method.
-   Defines standalone graph methods:
    -   some graphs compare to the subspace FCI energy.
    -   other graphs compare to the HF energy and to the exact energy.
-   Runs experiments on $H_2$, $HeH$ and $LiH$.
-   Shows landscapes (no histograms) for all 3 molecules.
-   For $H_2$ and $HeH$, it uses the expressive-ansatz VQE method and performs VQE from each possible state, plotting the resulting point. (I don't plot in the other notebooks)
### `collective_experiments.ipynb`
-   Aggregation of experiments on all types of input Hamiltonians.
-   **does not** include every one of the experiments in the above notebooks.
-   only includes the experiments we wish to put in the paper.
-   should act as the most updated version.
-   still almost empty.

## Documentation files
-   `thoughts.md`: internal record of reserach directions.
