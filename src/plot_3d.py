# === ADD: visualization helpers for your schedule layout ===
from src.scheduling_flowtime import J, scheduling_flowtime_problem
from src.scheduling_flowtime import get_diagonal_kronker
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

def extract_ising_from_sparse_pauli_op(op: SparsePauliOp):
    """
    Extract Ising fields/couplings from a SparsePauliOp.
    Returns:
        h: dict {i: h_i} for single-Z terms
        J: dict {(i,j): J_ij} for ZZ terms with i<j
        const: scalar constant
    """
    h, J = {}, {}
    const = 0.0
    for coeff, pstr in zip(op.coeffs, op.paulis):
        s = str(pstr)  # e.g. 'IIZZI...'
        z_pos = [k for k, ch in enumerate(s) if ch == 'Z']
        c = float(coeff.real)
        if len(z_pos) == 0:
            const += c
        elif len(z_pos) == 1:
            i = z_pos[0]
            h[i] = h.get(i, 0.0) + c
        elif len(z_pos) == 2:
            i, j = sorted(z_pos)
            J[(i, j)] = J.get((i, j), 0.0) + c
        else:
            # higher-order ZZZ... terms not shown in the 3D graph
            continue
    return h, J, const


def index_to_coords(idx, M, num_jobs, T):
    """
    Inverse of your get_indicator_index:
    return (t, j, m) with m in {0..M-1} for X_{m,j,t} and m==M for C_{j,t}
    """
    m = idx % (M + 1)
    j = (idx // (M + 1)) % num_jobs
    t = idx // ((M + 1) * num_jobs)
    return t, j, m


from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plot

def plot_schedule_spins_3d(op: SparsePauliOp, M, num_jobs, T, bitstring):
    """
    3D lattice of spins with up/down arrows.
    Colors:
        Magenta (↑) = spin +1
        Cyan (↓)    = spin -1
    Gray edges connect nearest neighbors in (t, j, m).
    """
    assert bitstring is not None, "bitstring is required for spin arrows"
    N = op.num_qubits
    assert len(bitstring) == N
    assert N == (M * num_jobs * (T + 1) + num_jobs * (T + 1)), \
        "num_qubits doesn't match (M,J,T) layout."

    # Get Ising couplings
    _, J, _ = extract_ising_from_sparse_pauli_op(op)

    # Positions
    pos = [index_to_coords(i, M, num_jobs, T) for i in range(N)]

    # Spins from bitstring
    def bit_to_spin(b): return 1 if b == '0' else -1
    spins = [bit_to_spin(b) for b in bitstring]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw spins as arrows
    for (t, j, m), s in zip(pos, spins):
        color = 'magenta' if s == 1 else 'cyan'
        ax.quiver(t, j, m, 0, 0, s, length=0.5, normalize=True, color=color)

    # Draw nearest-neighbor couplings as gray edges
    for (i, j_idx), _ in J.items():
        (t1, j1, m1) = pos[i]
        (t2, j2, m2) = pos[j_idx]
        if abs(t1 - t2) + abs(j1 - j2) + abs(m1 - m2) == 1:
            ax.plot([t1, t2], [j1, j2], [m1, m2], color='gray', linewidth=0.5)

    # Axis settings
    ax.set_xlabel('time t')
    ax.set_ylabel('job j')
    ax.set_zlabel('layer m (M = completion)')
    ax.set_xlim(0, T)
    ax.set_ylim(0, num_jobs - 1)
    ax.set_zlim(0, M)
    ax.set_xticks(range(T + 1))
    ax.set_yticks(range(num_jobs))
    ax.set_zticks(range(M + 1))
    #ax.set_title('Spin configuration on scheduling lattice')

    plt.tight_layout()
    plt.show()





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

if __name__ == "__main__":
    try:
        problem = test_basic_functionality()
        H = problem.get_hamiltonian(penalty_coeff=10.0)
        diagonal = get_diagonal_kronker(H)
        min_eigenvalue_index = np.argmin(diagonal)  # Find INDEX of minimum value
        binary_solution = format(min_eigenvalue_index, f'0{problem.num_qubits}b')
        plot_schedule_spins_3d(H, problem.M, len(problem.J), problem.T, binary_solution)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()    