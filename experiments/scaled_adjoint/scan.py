"""The frozen scan's first-order VJP with exponent-separated recurrence."""
import math

import numpy as np

from experiments.scaled_adjoint.scaled import Gradient, MAX_EXPONENT, align, pair_add, require


def scaled_vjp(jacobian, upstream):
    jacobian, upstream = np.asarray(jacobian, np.float64), np.asarray(upstream, np.float64)
    require(jacobian.ndim == 3 and jacobian.shape[0] == 5 and min(jacobian.shape[1:]) > 0,
            'invalid local Jacobian geometry')
    _, batch, cells = jacobian.shape
    require(upstream.shape == (batch, cells + 1), 'invalid upstream geometry')
    require(np.isfinite(jacobian).all() and np.isfinite(upstream).all(), 'nonfinite local derivative/upstream')
    jm, je = np.frexp(jacobian)
    um, ue = np.frexp(upstream)
    je, ue = je.astype(np.int64), ue.astype(np.int64)
    adj_m, adj_e = um[:, -1].copy(), ue[:, -1].copy()
    partial_m, partial_e = np.empty((4, batch, cells)), np.empty((4, batch, cells), dtype=np.int64)
    lost = 0
    for t in range(cells - 1, -1, -1):
        # frexp(jac) first prevents even a subnormal local derivative from being
        # multiplied outside the mantissa's representable range.
        values, shifts = np.frexp(adj_m[None] * jm[1:, :, t])
        partial_m[:, :, t] = values
        partial_e[:, :, t] = np.where(values != 0, adj_e[None] + je[1:, :, t] + shifts, 0)
        product_m, product_shift = np.frexp(adj_m * jm[0, :, t])
        product_e = np.where(product_m != 0, adj_e + je[0, :, t] + product_shift, 0)
        require((np.abs(product_e) <= MAX_EXPONENT).all(), 'binary exponent budget exceeded')
        adj_m, adj_e, missing = pair_add(product_m, product_e, um[:, t], ue[:, t])
        lost += missing
    require((np.abs(partial_e) <= MAX_EXPONENT).all(), 'binary exponent budget exceeded')
    candidates = np.concatenate((partial_e[partial_m != 0], adj_e[adj_m != 0]))
    common = int(candidates.max()) if candidates.size else 0
    partial, missing = align(partial_m, np.minimum(partial_e - common, 0))
    lost += missing
    initial, missing = align(adj_m, np.minimum(adj_e - common, 0))
    lost += missing
    return Gradient.create(dict(periods=partial[0], observations=np.stack((partial[1], partial[2]), axis=-1),
                                initial=initial, gain=np.asarray(math.fsum(partial[3].reshape(-1)))),
                           common, lost)
