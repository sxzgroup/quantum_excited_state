import time
import tensorcircuit as tc
import numpy as np
import optax
import jax.numpy as jnp
import scipy
import scipy.sparse
from jax.experimental import sparse  # Use the modern COO class
import openfermion
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator


K = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("cotengra")

# --- Parameters ---
x_dimension = 2  # Number of rows
y_dimension = 3  # Number of columns
t = 1.0  # Hopping strength
U = 4.0  # On-site interaction strength
chemical_potential = (
    U / 2
)  # Optional: Set chemical potential (often U/2 for half-filling studies)

n_sites = x_dimension * y_dimension
n_qubits = 2 * n_sites  # Each site has spin up and spin down


def get_site_index(row, col):
    """Maps a 2D lattice coordinate (row, col) to a 1D site index."""
    if not (0 <= row < x_dimension and 0 <= col < y_dimension):
        raise ValueError("Coordinates out of bounds")
    return row * y_dimension + col


def get_spin_orbital_index(row, col, spin):
    """Maps a lattice site and spin to a single fermionic mode index.
    Spin 0: up, Spin 1: down.
    Qubit ordering: site0_up, site0_down, site1_up, site1_down, ...
    """
    site_index = get_site_index(row, col)
    if spin not in [0, 1]:
        raise ValueError("Spin must be 0 (up) or 1 (down)")
    return 2 * site_index + spin


# --- Build the Fermionic Hamiltonian ---
hamiltonian_fermionic = FermionOperator()

# 1. Hopping terms (-t * sum_{<i,j>,sigma} (a†_{i,sigma} a_{j,sigma} + H.c.))
# print("\nAdding hopping terms...")
for r in range(x_dimension):
    for c in range(y_dimension):
        site_i = get_site_index(r, c)

        # Hopping right (horizontal)
        if c + 1 < y_dimension:
            site_j = get_site_index(r, c + 1)
            for spin in [0, 1]:  # 0 for up, 1 for down
                idx_i_sigma = get_spin_orbital_index(r, c, spin)
                idx_j_sigma = get_spin_orbital_index(r, c + 1, spin)
                # -t * a†_{i,sigma} a_{j,sigma}
                hamiltonian_fermionic += FermionOperator(
                    ((idx_i_sigma, 1), (idx_j_sigma, 0)), -t
                )
                # -t * a†_{j,sigma} a_{i,sigma} (Hermitian conjugate)
                hamiltonian_fermionic += FermionOperator(
                    ((idx_j_sigma, 1), (idx_i_sigma, 0)), -t
                )
                # print(f"  Hopping: ({r},{c})<->({r},{c+1}), spin={spin}, indices=({idx_i_sigma}, {idx_j_sigma})")

        # Hopping down (vertical)
        if r + 1 < x_dimension:
            site_j = get_site_index(r + 1, c)
            for spin in [0, 1]:  # 0 for up, 1 for down
                idx_i_sigma = get_spin_orbital_index(r, c, spin)
                idx_j_sigma = get_spin_orbital_index(r + 1, c, spin)
                # -t * a†_{i,sigma} a_{j,sigma}
                hamiltonian_fermionic += FermionOperator(
                    ((idx_i_sigma, 1), (idx_j_sigma, 0)), -t
                )
                # -t * a†_{j,sigma} a_{i,sigma} (Hermitian conjugate)
                hamiltonian_fermionic += FermionOperator(
                    ((idx_j_sigma, 1), (idx_i_sigma, 0)), -t
                )
                # print(f"  Hopping: ({r},{c})<->({r+1},{c}), spin={spin}, indices=({idx_i_sigma}, {idx_j_sigma})")


# 2. On-site interaction terms (U * sum_i n_{i,up} n_{i,down})
#    n_{i,sigma} = a†_{i,sigma} a_{i,sigma}
# print("\nAdding on-site interaction terms...")
for r in range(x_dimension):
    for c in range(y_dimension):
        idx_up = get_spin_orbital_index(r, c, 0)  # Spin up index
        idx_down = get_spin_orbital_index(r, c, 1)  # Spin down index

        # U * a†_{up} a_{up} a†_{down} a_{down}
        # Note OpenFermion standard ordering (normal ordering often preferred for efficiency,
        # but this direct translation n_up * n_down is also correct)
        # n_up * n_down = (a†_up a_up) * (a†_down a_down)
        # OpenFermion requires descending indices in creation operators first, then annihilation
        # So we represent it as a†_up a†_down a_down a_up (after normal ordering, check commutation)
        # Let's use the direct number operator definition:
        term = FermionOperator(
            ((idx_up, 1), (idx_up, 0), (idx_down, 1), (idx_down, 0)), U
        )
        hamiltonian_fermionic += term
        # print(f"  Interaction: ({r},{c}), indices=({idx_up}, {idx_down})")

# 3. Optional: Chemical potential terms (-mu * sum_{i, sigma} n_{i, sigma})
# print("\nAdding chemical potential terms...")  # Uncomment if using
for r in range(x_dimension):
    for c in range(y_dimension):
        for spin in [0, 1]:
            idx_sigma = get_spin_orbital_index(r, c, spin)
            # -mu * a†_{sigma} a_{sigma}
            hamiltonian_fermionic += FermionOperator(
                ((idx_sigma, 1), (idx_sigma, 0)), -chemical_potential
            )
            # print(f"  Chem Pot: ({r},{c}), spin={spin}, index={idx_sigma}")


hamiltonian_qubit = jordan_wigner(hamiltonian_fermionic)
hamiltonian_sparse = get_sparse_operator(hamiltonian_qubit, n_qubits=n_qubits)

scipy_coo = hamiltonian_sparse.tocoo()
data_np = scipy_coo.data
row_np = scipy_coo.row
col_np = scipy_coo.col
shape_tuple = scipy_coo.shape
data_jax = jnp.asarray(data_np)
row_jax = jnp.asarray(row_np)
col_jax = jnp.asarray(col_np)
indices_jax = jnp.stack([row_jax, col_jax], axis=1)
jax_coo = sparse.BCOO((data_jax, indices_jax), shape=shape_tuple)
hamiltonian = jax_coo


def psi(param):
    param = K.real(param)
    c = tc.Circuit(n)
    # for i in range(0, n, 2):
    # c.h(i)
    # c.cx(i, i+1)
    # c.x(i+1)

    for i in range(d):
        for j in range(n):
            c.ry(j, theta=param[i, j, 3])
            c.rz(j, theta=param[i, j, 4])
            c.rx(j, theta=param[i, j, 5])
        for j in range(n - 1):
            c.exp1(j, j + 1, theta=param[i, j, 0], unitary=tc.gates._zz_matrix)
            c.exp1(j, j + 1, theta=param[i, j, 1], unitary=tc.gates._xx_matrix)
            c.exp1(j, j + 1, theta=param[i, j, 2], unitary=tc.gates._yy_matrix)

    return c.state()


k = 16
n = 12
d = 5
eps = 1e-15

psi_vmap = K.vmap(psi)


def get_h_s(params):
    states = psi_vmap(params)
    S = K.zeros([k, k])
    H = K.zeros([k, k])
    for i in range(k):
        for j in range(i, k):
            # print((mpss[i].adjoint()@mpss[j]).eval())
            s = K.tensordot(K.conj(states[i]), states[j], axes=1)
            S = S.at[i, j].set(s)
            if i != j:
                S = S.at[j, i].set(K.conj(s))
            h = K.sparse_dense_matmul(hamiltonian, states[j])
            h = K.tensordot(K.conj(states[i]), h, axes=1)
            H = H.at[i, j].set(h)
            if i != j:
                H = H.at[j, i].set(K.conj(h))
    return H, S


def excited_loss(params):
    H, S = get_h_s(params)

    return K.real(K.trace(K.inv(S + eps * K.eye(k)) @ H))


vgf = K.jit(K.value_and_grad(excited_loss))

loss_jit = K.jit(excited_loss)


@K.jit
def step(params, opt_state):
    value, grad = vgf(params)
    updates, opt_state = solver.update(
        grad,
        opt_state,
        params,
        value=value,
        grad=grad,
        value_fn=loss_jit,
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, value


if __name__ == "__main__":
    params = K.implicit_randn(stddev=0.01, shape=[k, d, n, 6])

    hists = []
    paramss = []
    for _ in range(10):
        params = K.implicit_randn(stddev=0.01, shape=[k, d, n, 6])

        solver = optax.lbfgs()
        opt_state = solver.init(params)
        # vgf = K.jit(K.value_and_grad(loss))
        hist = []
        for i in range(2000):
            params, opt_state, value = step(params, opt_state)
            hist.append(value)
            if i % 50 == 0:
                print(value)
        hists.append(hist)
        paramss.append(params)

    np.save(f"vqe_n{n}_d{d}_k{k}.npy", paramss)
    np.save(f"vqe_hist_n{n}_d{d}_k{k}.npy", hists)
