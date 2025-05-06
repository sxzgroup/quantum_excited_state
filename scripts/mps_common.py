from functools import partial
import time
import tensorcircuit as tc

tc.set_dtype("complex128")
K = tc.set_backend("jax")
tc.set_contractor("cotengra")

L = 16
k = 16
chi = 24
eps = 1e-15
title = "mps_1_1_1_0"


def psiihpsijn(A, B, Hs, N):
    # A: L552 H 5522
    Ans = [tc.gates.Gate(K.copy(A[i])) for i in range(N)]
    Bns = [tc.gates.Gate(K.copy(K.conj(B[i]))) for i in range(N)]
    Hns = [tc.gates.Gate(K.copy(Hs[i])) for i in range(N)]
    for i in range(N):
        Ans[i][1] ^ Ans[(i + 1) % N][0]
        Bns[i][1] ^ Bns[(i + 1) % N][0]
        Hns[i][1] ^ Hns[(i + 1) % N][0]
        Ans[i][2] ^ Hns[i][2]
        Bns[i][2] ^ Hns[i][3]
    r = tc.contractor(Ans + Bns + Hns).tensor
    return r


def psiipsijn(A, B, N):
    Ans = [tc.gates.Gate(K.copy(A[i])) for i in range(N)]
    Bns = [tc.gates.Gate(K.copy(K.conj(B[i]))) for i in range(N)]
    for i in range(N):
        Ans[i][1] ^ Ans[(i + 1) % N][0]
        Bns[i][1] ^ Bns[(i + 1) % N][0]
        Ans[i][2] ^ Bns[i][2]
    r = tc.contractor(Ans + Bns).tensor
    return r


# psiihpsijn_jit = K.jit(psiihpsijn, static_argnums=3)
# psiipsijn_jit = K.jit(psiipsijn, static_argnums=2)
psiihpsijn_v = K.jit(K.vmap(psiihpsijn, vectorized_argnums=(0, 1)), static_argnums=3)
psiipsijn_v = K.jit(K.vmap(psiipsijn, vectorized_argnums=(0, 1)), static_argnums=2)


def loss(u, hs, L):
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
    obj = K.real(K.trace(K.inv(S + eps * K.eye(k)) @ H))
    return obj


vgf = K.jit(K.value_and_grad(loss), static_argnums=2)
loss_jit = K.jit(loss, static_argnums=2)


@partial(K.jit, static_argnums=(3,))
def psiihhpsijn(A, B, Hs, N):
    Ans = [tc.gates.Gate(K.copy(A[i])) for i in range(N)]
    Bns = [tc.gates.Gate(K.copy(K.conj(B[i]))) for i in range(N)]
    Hns = [tc.gates.Gate(K.copy(Hs[i])) for i in range(N)]
    Hns2 = [tc.gates.Gate(K.copy(Hs[i])) for i in range(N)]

    for i in range(N):
        Ans[i][1] ^ Ans[(i + 1) % N][0]
        Bns[i][1] ^ Bns[(i + 1) % N][0]
        Hns[i][1] ^ Hns[(i + 1) % N][0]
        Hns2[i][1] ^ Hns2[(i + 1) % N][0]

        Ans[i][2] ^ Hns[i][2]
        Hns[i][3] ^ Hns2[i][2]
        Bns[i][2] ^ Hns2[i][3]
    r = tc.contractor(Ans + Bns + Hns + Hns2).tensor
    return r
