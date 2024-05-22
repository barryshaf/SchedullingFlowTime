# Required experiments

regarding the calcuations and experiments, we will use the experimetns in this repo (from Ittay).
We will use Dekel's illustrations to explain MUB generation and expressive vs. non-expressive asnatzes.

1-qubit will be used as an illustration.
Choose (two) *clear* illustrations that have different paths and lengths.
for eac one, put the full pic and a close-up on the end of the convergence.

4 figs.

**NOTE:** try and color the sphere.



start from landscaping then go to VQE.

landscaping:
start from Full MUB for 2-3 qubits.
Give examples from 2-qubit molecules, 3-qubit transverse.
put a 3-qubit MAXCUT example, but **clarify** that it is obvious that we will reach the optimal point with the comp. basis.
then go to Partial MUB for 4 and 9 qubits.
We'll put examples from all 3 hamiltonian groups for 4 and 8 qubits.
make sure that for the 9-qubit example, the cut has a 4-5 partition, so the partial MUBs don't get
an auto-win with the comp. basis.

8*2 = 16 figs.

**NOTE:** make sure that each subset gets a constant color between different bases.

VQE:
interesting graphs:
-   vqe from the best $m$ MUB landscape points out of $n$ sampled.
-   vqe from the best $m$ and the *worst* $m$ MUB landscape points out of $n$ sampled.
-   vqe from the best $m$ of $n$ MUB landscape points, vs. the best $m$ of $n$ random theta vectors.


use the exact same Hamiltonians.
Pick a single Hamiltonian from each size group, and present all three graphs for it.
pick the most interesting one. :)
  


# Ohter notes
we want to mention that we *could* choose other optimizer but wanted COBYLA because of so and so (no many small perturbations).

when explaining how we have the MUB statesw for 2 and 3 qubits (and not for more), we can use the Lawence paper.