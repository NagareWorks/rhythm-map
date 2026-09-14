"""Deterministic groupwise projection, without a model or optimizer update."""
import math

import numpy as np

from experiments.gradient_audit.audit import cosine
from experiments.scaled_adjoint.scaled import Gradient, accumulate, require

GROUPS = ('period_head', 'shared_cnn', 'phase_parameters')


def model_shapes():
    shapes = {'projection.weight': (32, 512, 1), 'projection.bias': (32,),
              'output.weight': (3, 32, 1), 'output.bias': (3,),
              'origin.weight': (1, 32, 1), 'origin.bias': (1,), 'gain_logit': ()}
    for i in range(6):
        shapes.update({f'blocks.{i}.depthwise.weight': (32, 1, 5),
                       f'blocks.{i}.depthwise.bias': (32,),
                       f'blocks.{i}.pointwise.weight': (32, 32, 1),
                       f'blocks.{i}.pointwise.bias': (32,)})
    return shapes


def partition(gradient):
    require({n: x.shape for n, x in gradient.blocks.items()} == model_shapes(),
            'routing requires exact PhaseSyncReadout parameter geometry')
    parts = {g: {n: np.zeros_like(x) for n, x in gradient.blocks.items()} for g in GROUPS}
    for name, value in gradient.blocks.items():
        if name in ('output.weight', 'output.bias'):
            parts['period_head'][name][0] = value[0]
            parts['phase_parameters'][name][1:] = value[1:]
        elif name.startswith(('projection.', 'blocks.')):
            parts['shared_cnn'][name][...] = value
        else:
            parts['phase_parameters'][name][...] = value
    return {g: Gradient.create(p, gradient.exponent, gradient.alignment_underflows)
            for g, p in parts.items()}


def _dot(a, b):
    require(a.blocks.keys() == b.blocks.keys() and
            all(a.blocks[n].shape == b.blocks[n].shape for n in a.blocks), 'gradient geometry differs')
    return math.fsum(float(np.sum(a.blocks[n] * b.blocks[n])) for n in a.blocks)


def _count_projection_log2(direction, count):
    """log2(dot(direction,count) / ||count||^2), without full-scale vectors."""
    dot, square = _dot(direction, count), _dot(count, count)
    if not square:
        return None
    require(dot > 0, 'routed direction does not descend count')
    return direction.exponent - count.exponent + math.log2(dot / square)


def project(phase, count):
    """One group; return complete count+projected-phase, retained phase, receipt."""
    require(phase.alignment_underflows == count.alignment_underflows == 0, 'input alignment loss')
    dot, square = _dot(phase, count), _dot(count, count)
    active = bool(square and dot < 0)
    retained = (Gradient.create({n: x - (dot / square) * count.blocks[n]
                                 for n, x in phase.blocks.items()}, phase.exponent)
                if active else phase)
    total = accumulate([count, retained])
    require(total.alignment_underflows == retained.alignment_underflows == 0, 'output alignment loss')
    projection = _count_projection_log2(total, count)
    require(projection is None or projection >= math.log2(1 - 1e-10),
            'routing lost protected count component')
    receipt = dict(conflict=active, count_direction_defined=bool(square),
                   phase_count_cosine=cosine(phase, count),
                   routed_count_cosine=cosine(total, count),
                   retained_phase_norm_log2=retained.norm_log2(),
                   original_phase_norm_log2=phase.norm_log2(),
                   count_projection_ratio_log2=projection)
    return total, retained, receipt


def route(phase, count):
    """All groups, after complete batch accumulation; no public strategy switch."""
    p, c = partition(phase), partition(count)
    routed, receipts = [], {}
    for name in GROUPS:
        value, _, receipts[name] = project(p[name], c[name])
        routed.append(value)
    total = accumulate(routed)
    require(total.alignment_underflows == 0, 'group recombination lost alignment')
    # Recheck the ACTUAL recombined direction, not just intermediate groups.
    for name, value in partition(total).items():
        ratio = _count_projection_log2(value, c[name])
        require(ratio is None or ratio >= math.log2(1 - 1e-10), 'recombination lost protection')
    return total, receipts


def route_batch(records):
    """Records are (positive weight, phase gradient, count gradient)."""
    require(bool(records) and all(math.isfinite(w) and w > 0 for w, _, _ in records),
            'positive nonempty batch weights required')
    phase = accumulate([p.multiply(w) for w, p, _ in records])
    count = accumulate([c.multiply(w) for w, _, c in records])
    return route(phase, count)


def displacement_check(count, before, after):
    """Inspect actual rounded parameter displacement; NEVER fixes/applies a step.

    A nonpositive first-order derivative is necessary, not a finite-loss gate.
    Undefined count gradients and a zero displacement remain distinct.
    """
    require(before.keys() == after.keys() == count.blocks.keys(), 'displacement keys differ')
    delta = {}
    for name, x in before.items():
        x, y = np.asarray(x), np.asarray(after[name])
        require(x.shape == y.shape == count.blocks[name].shape and
                np.isfinite(x).all() and np.isfinite(y).all(), 'invalid displacement geometry or values')
        delta[name] = y.astype(np.float64) - x.astype(np.float64)
    step = Gradient.create(delta)
    alignment = cosine(step, count)
    return dict(count_direction_defined=count.norm_log2() is not None,
                moved=step.norm_log2() is not None, displacement_count_cosine=alignment,
                first_order_nonincrease=None if count.norm_log2() is None else
                (True if step.norm_log2() is None else alignment <= 0))
