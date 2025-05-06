import tensorcircuit as tc
import numpy as np
import quimb
import scipy
from scipy.linalg import cholesky

from mps_common import psiihhpsijn, psiihpsijn_v, psiipsijn_v, title, L, k, chi

tc.set_dtype("complex128")
K = tc.set_backend("jax")

qb_mpo = quimb.tensor.tensor_builder.MPO_ham_heis(
    L, j=(1.0, 1, 1), bz=-0.0, S=1 / 2, cyclic=True
)
mpo = tc.quantum.quimb2qop(qb_mpo)
hnodes = K.stack([qb_mpo.tensors[i].data for i in range(L)])


if __name__ == "__main__":
    x = np.load(title + f"_L{L}k{k}chi{chi}.npy")
    u = K.reshape(x, [k, L, chi, chi, 2])
    hs = hnodes
    S = K.zeros([k, k])
    H = K.zeros([k, k])
    ux = []
    uy = []
    for i in range(k):
        for j in range(i, k):
            ux.append(K.conj(u[i]))
            uy.append(u[j])
    ux = K.stack(ux)
    uy = K.stack(uy)
    hij = psiihpsijn_v(ux, uy, hs, L)
    sij = psiipsijn_v(ux, uy, L)
    m = 0
    for i in range(k):
        for j in range(i, k):
            S = S.at[i, j].set(sij[m])
            if i != j:
                S = S.at[j, i].set(K.conj(sij[m]))
            H = H.at[i, j].set(hij[m])
            if i != j:
                H = H.at[j, i].set(K.conj(hij[m]))
            m += 1
    l = cholesky(S)

    es, v = K.eigh(K.inv(K.adjoint(l)) @ (H) @ K.inv(l))
    print("Variational results:", es[:k])
    if L <= 24:
        g = tc.templates.graphs.Line1D(L, pbc=True)
        h = tc.quantum.heisenberg_hamiltonian(
            g, hxx=1 / 4, hyy=1 / 4, hzz=1 / 4, hz=0.0 / 2, sparse=True
        )
        es0, _ = scipy.sparse.linalg.eigsh(K.numpy(h), k=3 * k, which="SA")
        es0 = np.sort(es0)  # necessary
        print(K.sum(es0[:k]), es0[:k])
        print("relative energy error:", 1 - es[:k] / es0[:k])

    def get_hsquare(u, hs, N):
        hsquare = np.zeros([k, k])
        for i in range(k):
            for j in range(i, k):
                if i == j:
                    # print(psiihhpsijn(u[i], u[j], hs, N))
                    hsquare[i, j] = K.real(psiihhpsijn(u[i], u[j], hs, N))

                else:
                    ele = K.real(psiihhpsijn(u[i], u[j], hs, N))
                    # print(ele)
                    hsquare[i, j] = ele
                    hsquare[j, i] = np.conj(ele)
        return hsquare

    def variance_estimate(hsquare, hexp, trans, N):
        # u: k, chi, chi, 2
        # trans: k, k

        var = []
        rel_var = []
        for i in range(k):
            estsquare = 0
            estexp = 0
            for ii in range(k):
                for iii in range(k):
                    estsquare += (
                        K.adjoint(trans[i, ii]) * trans[i, iii] * hsquare[ii, iii]
                    )
                    estexp += K.adjoint(trans[i, ii]) * trans[i, iii] * hexp[ii, iii]
            var.append(estsquare - estexp**2)
            rel_var.append((estsquare - estexp**2) / estexp**2)
        return K.real(K.stack(var)), K.real(K.stack(rel_var))

    trans = K.adjoint(v) @ K.inv(K.adjoint(l))
    H2 = get_hsquare(u, hs, L)
    print("Variance", variance_estimate(H2, H, trans, L))
