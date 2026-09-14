"""Nominal source-time maps, independent of any time-stretch implementation."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Timeline:
    """Rates are source seconds per output second, one per source interval.

    A requested map is not evidence that rendered audio obeys it. Never rescale
    this map to hide a renderer's duration error. Boundaries are right-owned.
    """

    source_edges: tuple
    rates: tuple

    def __post_init__(self):
        edges, rates = np.asarray(self.source_edges), np.asarray(self.rates)
        if (edges.ndim != 1 or rates.ndim != 1 or len(edges) != len(rates) + 1
                or not len(rates) or not np.isfinite(edges).all()
                or not np.isfinite(rates).all() or edges[0] != 0
                or np.any(np.diff(edges) <= 0) or np.any(rates <= 0)):
            raise ValueError('invalid time map')
        object.__setattr__(self, 'source_edges', tuple(float(v) for v in edges))
        object.__setattr__(self, 'rates', tuple(float(v) for v in rates))

    @property
    def output_edges(self):
        return np.r_[0., np.cumsum(np.diff(self.source_edges) / self.rates)]

    @staticmethod
    def _map(values, domain, image):
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < domain[0]) or np.any(values > domain[-1]):
            raise ValueError('time outside map')
        return np.interp(values, domain, image)

    def to_output(self, source_times):
        return self._map(source_times, self.source_edges, self.output_edges)

    def to_source(self, output_times):
        return self._map(output_times, self.output_edges, self.source_edges)

    def phase_and_rate(self, beats, output_times):
        """Compose the original beat-phase function with the source-time map.

        Do not linearly interpolate *mapped* beats across a rate boundary: it
        would smear a known speed step over the bracketing beat interval.
        Rates are instantaneous beats/s, not finite-difference cell averages.
        Unbracketed points and the final output endpoint have no label.
        """
        beats = np.asarray(beats, dtype=np.float64)
        if (beats.ndim != 1 or len(beats) < 2 or not np.isfinite(beats).all()
                or np.any(beats < 0) or np.any(np.diff(beats) <= 0)):
            raise ValueError('invalid beats')
        t = np.atleast_1d(np.asarray(output_times, dtype=np.float64))
        source = self.to_source(t)
        i = np.searchsorted(beats, source, side='right') - 1
        valid = (i >= 0) & (i < len(beats) - 1) & (t < self.output_edges[-1])
        phase, rate = np.zeros(t.shape), np.zeros(t.shape)
        j = np.minimum(np.searchsorted(self.output_edges, t, side='right') - 1, len(self.rates) - 1)
        period = np.diff(beats)[i[valid]]
        phase[valid] = i[valid] + (source[valid] - beats[i[valid]]) / period
        rate[valid] = np.asarray(self.rates)[j[valid]] / period
        return phase, rate, valid
