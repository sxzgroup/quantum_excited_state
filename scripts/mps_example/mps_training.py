import time
from functools import partial

import numpy as np
import optax
import quimb
import tensorcircuit as tc

from mps_common import vgf, loss_jit, L, k, chi, title

tc.set_dtype("complex128")
K = tc.set_backend("jax")


qb_mpo = quimb.tensor.tensor_builder.MPO_ham_heis(
    L, j=(1.0, 1.0, 1.0), bz=0.0, S=1 / 2, cyclic=True
)
mpo = tc.quantum.quimb2qop(qb_mpo)
hnodes = K.stack([qb_mpo.tensors[i].data for i in range(L)])

tc.set_contractor("cotengra")


@K.jit
def step(params, opt_state):
    value, grad = vgf(params, hnodes, L)

    updates, opt_state = solver.update(
        grad,
        opt_state,
        params,
        value=value,
        grad=grad,
        value_fn=partial(loss_jit, hs=hnodes, L=L),
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, value


if __name__ == "__main__":
    hist = []
    u = K.cast(K.implicit_randn(shape=[k, L, chi, chi, 2]), tc.rdtypestr)
    # lbfgs only support real values
    solver = optax.lbfgs()
    params = u
    opt_state = solver.init(params)
    times = []
    for i in range(200000):
        time0 = time.time()
        params, opt_state, value = step(params, opt_state)
        print("Objective function: ", value)
        time1 = time.time()
        print("time:", time1 - time0)
        hist.append(value)

        if i % 500 == 0:
            print("writing...")
            np.save(title + f"_L{L}k{k}chi{chi}.npy", params)
            np.save(title + f"_hist_L{L}k{k}chi{chi}.npy", np.array(hist))
