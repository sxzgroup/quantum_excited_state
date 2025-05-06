import tensorcircuit as tc
import numpy as np
import tensornetwork as tn
import jax

K = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("cotengra")

D = 42301 / 17.78068 / 2
a = 2.144
xe = 0.9527
k = 16
eps = 1e-16
chi = 64
n_digits = 16
domainl = 0
domainr = 10


def morse(x):
    return D * (1 - K.exp(-a * (x - xe))) ** 2


points = np.arange(domainl, domainr, (-domainl + domainr) / 2**n_digits)
vs = K.convert_to_tensor(morse(points))
title = f"morse_d{n_digits}_k{k}_chi{chi}"


def psi(params):
    n_digits = params.shape[0]
    chi = params.shape[1]
    ## generate an MPO with bond dimension chi and tensor parameters params[i] for i-th tensor in tensornetwork API
    # left right upper
    mpo_nodes = []
    for i in range(n_digits):
        tensor_param = params[i]
        # Create a tensornetwork Node from the parameter tensor
        site_node = tn.Node(tensor_param, name=f"site_{i}_tensor")
        if i > 0:
            tn.connect(mpo_nodes[i - 1][1], site_node[0])
        mpo_nodes.append(site_node)
    lb = np.zeros([chi])
    lb[0] = 1.0
    left_boundary = tn.Node(lb)
    left_boundary[0] ^ mpo_nodes[0][0]
    mpo_nodes.append(left_boundary)
    right_boundary = tn.Node(lb)
    right_boundary[0] ^ mpo_nodes[-2][1]
    mpo_nodes.append(right_boundary)
    t = tc.contractor(
        mpo_nodes, output_edge_order=[n[2] for n in mpo_nodes[:-2]]
    ).tensor
    return tc.backend.reshape(t, [-1])


def dd(f, x, delta):
    return (f(x + delta) - 2 * f(x) + f(x - delta)) / delta**2


def dd_vector(f, delta):
    return ((jax.numpy.roll(f, -1) - 2 * f + jax.numpy.roll(f, 1)) / delta**2)[1:-1]


def dd4_vector(f, delta):
    # (-f(x + 2Δ) + 16f(x + Δ) - 30f(x) + 16f(x - Δ) - f(x - 2Δ)) / (12Δ²)
    return (
        (
            -jax.numpy.roll(f, -2)
            + 16 * jax.numpy.roll(f, -1)
            - 30 * f
            + 16 * jax.numpy.roll(f, 1)
            - jax.numpy.roll(f, 2)
        )
        / delta**2
        / 12
    )[2:-2]


def get_h_s(params):
    states = [psi(param) for param in params]
    S = K.zeros([k, k])
    H = K.zeros([k, k])
    for i in range(k):
        for j in range(i, k):
            # print((mpss[i].adjoint()@mpss[j]).eval())
            s = K.tensordot(states[i], states[j], axes=1)
            S = S.at[i, j].set(s)
            if i != j:
                S = S.at[j, i].set(s)
            h = K.real(
                K.tensordot(vs, states[i] * states[j], axes=1)
                - 0.5
                * K.tensordot(
                    states[i][1:-1],
                    dd_vector(states[j], (domainr - domainl) / 2**n_digits),
                    axes=1,
                )
            )
            h += K.real(
                K.tensordot(vs, states[j] * states[i], axes=1)
                - 0.5
                * K.tensordot(
                    states[j][1:-1],
                    dd_vector(states[i], (domainr - domainl) / 2**n_digits),
                    axes=1,
                )
            )
            h = h / 2
            H = H.at[i, j].set(h)
            if i != j:
                H = H.at[j, i].set(h)
    return H, S


def excited_loss(params):
    H, S = get_h_s(params)

    return K.real(K.trace(K.inv(S + eps * K.eye(k)) @ H))


loss_jit = K.jit(excited_loss)
vgf = K.jit(K.value_and_grad(excited_loss))
