import tensorcircuit as tc
import numpy as np
from scipy.linalg import cholesky
from morse_common import title, D, a, k, get_h_s

tc.set_dtype("complex128")
K = tc.set_backend("jax")


def get_baseline(D, a, n):
    hv = a * np.sqrt(D * 2)
    return (n + 0.5) * hv - (hv * (n + 0.5)) ** 2 / 4 / D


if __name__ == "__main__":
    es0 = np.array([get_baseline(D, a, i) for i in range(k)])
    x = np.load(title + f".npy")
    H, S = get_h_s(x)
    l = cholesky(S)
    es, v = K.eigh(K.inv(K.adjoint(l)) @ (H) @ K.inv(l))
    print("Variational results:", es[:k])
    print("relative energy error:", es[:k] / es0[:k] - 1)
