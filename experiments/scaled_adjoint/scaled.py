"""Float64 mantissas plus a shared binary exponent for gradient transport.

This extends range, not precision. Cancellation and underflow of very small
aligned components remain observable limitations, not a license to call a
clipped direction exact in arbitrary ill-conditioned floating-point problems.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

MAX_EXPONENT = 1 << 52


def require(condition, message):
    if not condition:
        raise ValueError(message)


def align(value, shift):
    """Only downscale; shifts below float64 range produce counted zeros."""
    shift = np.asarray(shift, dtype=np.int64)
    require((shift <= 0).all(), 'alignment must not increase magnitude')
    with np.errstate(under='ignore'):
        result = np.ldexp(value, np.maximum(shift, -1075))
    return result, int(np.count_nonzero((value != 0) & (result == 0)))


def pair_add(am, ae, bm, be):
    """Add normalized per-scalar mantissa/exponent arrays without overflow."""
    am, bm = np.asarray(am, np.float64), np.asarray(bm, np.float64)
    ae, be = np.asarray(ae, np.int64), np.asarray(be, np.int64)
    common = np.maximum(np.where(am != 0, ae, -MAX_EXPONENT),
                        np.where(bm != 0, be, -MAX_EXPONENT))
    common = np.where((am == 0) & (bm == 0), 0, common)
    a, lost_a = align(am, np.minimum(ae - common, 0))
    b, lost_b = align(bm, np.minimum(be - common, 0))
    mantissa, shift = np.frexp(a + b)
    exponent = np.where(mantissa != 0, common + shift, 0)
    require((np.abs(exponent) <= MAX_EXPONENT).all(), 'binary exponent budget exceeded')
    return mantissa, exponent, lost_a + lost_b


@dataclass(frozen=True)
class Gradient:
    blocks: dict
    exponent: int
    alignment_underflows: int = 0

    def __post_init__(self):
        require(type(self.exponent) is int and abs(self.exponent) <= MAX_EXPONENT, 'invalid binary exponent')
        require(type(self.alignment_underflows) is int and self.alignment_underflows >= 0,
                'invalid alignment loss count')
        require(bool(self.blocks), 'empty gradient')
        values = {name: np.asarray(value, dtype=np.float64).copy() for name, value in self.blocks.items()}
        require(all(isinstance(name, str) and name and x.size and np.isfinite(x).all()
                    for name, x in values.items()), 'invalid gradient block')
        peak = max(float(np.abs(x).max()) for x in values.values())
        require((peak == 0 and self.exponent == 0) or .5 <= peak < 1., 'unnormalized gradient')
        for value in values.values():
            value.setflags(write=False)
        object.__setattr__(self, 'blocks', MappingProxyType(values))

    @classmethod
    def create(cls, blocks, exponent=0, alignment_underflows=0):
        require(type(exponent) is int and abs(exponent) <= MAX_EXPONENT, 'invalid binary exponent')
        require(bool(blocks), 'empty gradient')
        values = {name: np.asarray(value, dtype=np.float64).copy() for name, value in blocks.items()}
        require(all(x.size and np.isfinite(x).all() for x in values.values()), 'nonfinite or empty mantissa')
        peak = max(float(np.abs(x).max()) for x in values.values())
        if peak == 0:
            return cls(values, 0, alignment_underflows)
        _, shift = math.frexp(peak)
        require(abs(exponent + shift) <= MAX_EXPONENT, 'binary exponent budget exceeded')
        lost = alignment_underflows
        for name, value in values.items():
            with np.errstate(under='ignore'):
                values[name] = np.ldexp(value, -shift)
            lost += int(np.count_nonzero((value != 0) & (values[name] == 0)))
        return cls(values, exponent + shift, lost)

    def multiply(self, weight):
        require(math.isfinite(weight), 'nonfinite loss weight')
        if weight == 0:
            return Gradient.create({name: np.zeros_like(x) for name, x in self.blocks.items()},
                                   alignment_underflows=self.alignment_underflows)
        mantissa, exponent = math.frexp(weight)
        # [1, 2) avoids needlessly halving the smallest subnormal for weight=1.
        mantissa, exponent = mantissa * 2., exponent - 1
        with np.errstate(under='ignore'):
            values = {name: x * mantissa for name, x in self.blocks.items()}
        lost = sum(int(np.count_nonzero((x != 0) & (values[name] == 0))) for name, x in self.blocks.items())
        return Gradient.create(values, self.exponent + exponent, self.alignment_underflows + lost)

    def norm_log2(self):
        squared = math.fsum(float(np.sum(x * x)) for x in self.blocks.values())
        return None if squared == 0 else self.exponent + .5 * math.log2(squared)

    def clipped(self, max_norm=1., epsilon=1e-6):
        """Algebraic global L2 clipping: g * min(1, cap/(||g|| + eps)).

        Never materialize an overflowing full gradient/norm. Returning finite
        clipped numbers does not remove the large derivative or cure conditioning.
        """
        require(math.isfinite(max_norm) and max_norm > 0 and math.isfinite(epsilon) and epsilon >= 0,
                'invalid global clipping contract')
        norm = math.sqrt(math.fsum(float(np.sum(x * x)) for x in self.blocks.values()))
        if norm == 0:
            return {name: x.copy() for name, x in self.blocks.items()}
        log_norm = self.exponent + math.log2(norm)
        nm, ne = math.frexp(norm)
        cm, ce = math.frexp(max_norm)
        larger = self.exponent + ne > ce or (self.exponent + ne == ce and nm > cm)
        if larger:
            # Here eps / 2**exponent could overflow for an unusually tiny cap;
            # use the ratio eps / ||g|| in log space instead.
            log_ratio = -math.inf if epsilon == 0 else math.log2(epsilon) - log_norm
            if log_ratio > 0:
                inverse = 2. ** -log_ratio
                factor = (max_norm * inverse) / (1. + inverse)
            else:
                factor = max_norm / (1. + 2. ** log_ratio)
            result = {name: (x / norm) * factor for name, x in self.blocks.items()}
        else:
            # ||g|| <= finite cap, so materializing each component is now safe.
            require(self.exponent <= 1024, 'inconsistent unclipped exponent')
            with np.errstate(under='ignore'):
                raw = {name: np.ldexp(x, max(self.exponent, -1075)) for name, x in self.blocks.items()}
            actual_norm = math.ldexp(norm, max(self.exponent, -1075))
            if epsilon <= max_norm - actual_norm:
                coefficient = 1.
            else:
                scale = max(actual_norm, epsilon)
                coefficient = (max_norm / scale) / (actual_norm / scale + epsilon / scale)
            result = {name: x * coefficient for name, x in raw.items()}
        require(all(np.isfinite(x).all() for x in result.values()), 'nonfinite clipped gradient')
        return result


def accumulate(gradients):
    """Combine all weighted recordings BEFORE global clipping, not per record."""
    require(bool(gradients), 'no accumulated gradients')
    first = gradients[0]
    require(all(g.blocks.keys() == first.blocks.keys() and
                all(g.blocks[k].shape == first.blocks[k].shape for k in first.blocks) for g in gradients),
            'gradient parameter geometry differs')
    nonzero = [g for g in gradients if g.norm_log2() is not None]
    if not nonzero:
        return Gradient.create(first.blocks, alignment_underflows=sum(g.alignment_underflows for g in gradients))
    common = max(g.exponent for g in nonzero)
    result, lost = {}, sum(g.alignment_underflows for g in gradients)
    for name, template in first.blocks.items():
        aligned = []
        for gradient in nonzero:
            value, missing = align(gradient.blocks[name], gradient.exponent - common)
            aligned.append(value.reshape(-1))
            lost += missing
        # fsum retains ordinary mixed-sign cancellation better than an arbitrary
        # order of +=; it still cannot recover previously underflowed components.
        result[name] = np.array([math.fsum(column) for column in zip(*aligned)]).reshape(template.shape)
    return Gradient.create(result, common, lost)
