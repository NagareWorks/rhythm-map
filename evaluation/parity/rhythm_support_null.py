"""Small algebraic null checks, not an audio detector or a calibrated test."""
import argparse
import json
from pathlib import Path

import numpy as np

from rhythm_presence_audit import build_report as presence_report, require


ABS_TOL = 1e-10  # Numerical witness budget, never a rhythm-confidence cutoff.


def vector(values):
    array = np.asarray(values)
    require(array.ndim == 1 and 2 <= array.size <= 2048 and array.dtype.kind in 'fiu',
            'expected a bounded real vector')
    array = array.astype(np.float64)
    require(np.isfinite(array).all() and np.max(np.abs(array)) <= 1e6, 'invalid vector values')
    return array


def phase_surrogate(values, phase_offsets):
    values = vector(values)
    offsets = vector(phase_offsets)
    spectrum = np.fft.rfft(values)
    require(offsets.shape == spectrum.shape, 'phase shape mismatch')
    require(offsets[0] == 0 and (values.size % 2 or offsets[-1] == 0),
            'DC and even-length Nyquist must remain real and unchanged')
    return np.fft.irfft(spectrum * np.exp(1j * offsets), n=values.size)


def circular_autocorrelation(values):
    """Direct time-domain calculation, independent of the Fourier construction."""
    values = vector(values)
    return np.array([np.dot(values, np.roll(values, lag)) for lag in range(values.size)])


def circular_feature_lags(features):
    features = np.asarray(features)
    require(features.ndim == 2 and 2 <= features.shape[0] <= 2048
            and 1 <= features.shape[1] <= 32 and features.dtype.kind in 'fiu',
            'expected bounded real feature matrix')
    features = features.astype(np.float64)
    require(np.isfinite(features).all() and np.max(np.abs(features)) <= 1e6,
            'invalid feature values')
    return np.array([np.mean(np.sum(features * np.roll(features, lag, axis=0), axis=1))
                     for lag in range(len(features))])


def rank_tail(observed, surrogates):
    """Arithmetic only: caller must justify exchangeability; not a detector API."""
    require(type(observed) in (int, float) and np.isfinite(observed), 'invalid statistic')
    require(type(surrogates) is list and 1 <= len(surrogates) <= 9999
            and all(type(x) in (int, float) and np.isfinite(x) for x in surrogates),
            'invalid surrogate statistics')
    return (1 + sum(x >= observed for x in surrogates)) / (len(surrogates) + 1)


def build_report():
    # Authored abstract arrays, not PCM, model features or newly labeled music.
    x = ((np.arange(64) ** 2) % 23 - 11).astype(np.float64) / 16
    offsets = ((np.arange(33) * 7) % 13).astype(np.float64) * np.pi / 13
    offsets[0] = offsets[-1] = 0
    surrogate = phase_surrogate(x, offsets)
    power_error = float(np.max(np.abs(abs(np.fft.rfft(x)) ** 2
                                     - abs(np.fft.rfft(surrogate)) ** 2)))
    lag_error = float(np.max(np.abs(circular_autocorrelation(x)
                                   - circular_autocorrelation(surrogate))))
    require(power_error <= ABS_TOL and lag_error <= ABS_TOL, 'Fourier invariant violated')
    require(float(np.max(np.abs(x - surrogate))) > 0.1, 'identity is not a useful witness')

    features = ((np.arange(64).reshape(16, 4) ** 2) % 19 - 9).astype(np.float64)
    shift_error = float(np.max(np.abs(circular_feature_lags(features)
                                     - circular_feature_lags(np.roll(features, 7, axis=0)))))
    require(shift_error <= ABS_TOL, 'circular feature invariant violated')

    # Uniform random rotations of this block vector form a stationary dependent
    # process on the finite circle. The alternating permutation is outside its
    # support: stationarity does not imply exchangeability under arbitrary order.
    block = np.array([1] * 4 + [-1] * 4)
    alternating = block[[0, 4, 1, 5, 2, 6, 3, 7]]
    orbit = {tuple(np.roll(block, lag)) for lag in range(len(block))}
    require(tuple(alternating) not in orbit, 'permutation witness lost its support difference')

    prior = presence_report()
    return dict(
        schema='rhythm-map.rhythm-support-null.v1', complete=True,
        decision='close_preserved_second_order_and_naive_permutation_shortcuts',
        numerical_abs_budget=ABS_TOL, reported_decimal_places=12,
        phase=dict(length=len(x), max_power_error=round(power_error, 12),
                   max_direct_circular_lag_error=round(lag_error, 12),
                   changed_sequence=True, same_second_order_statistics=True),
        shift=dict(frames=16, channels=4, offset=7, max_lag_error=round(shift_error, 12),
                   same_circular_feature_lags=True),
        permutation=dict(original=block.tolist(), permuted=alternating.tolist(),
                         stationary_rotation_orbit_size=len(orbit),
                         original_orbit_probability=1 / len(orbit), permuted_orbit_probability=0,
                         original_lag_one=float(circular_autocorrelation(block)[1] / len(block)),
                         permuted_lag_one=float(circular_autocorrelation(alternating)[1] / len(block)),
                         stationarity_does_not_imply_exchangeability=True),
        nonstationary_reference=dict(
            construction='X_t = a_t * independent_symmetric_random_sign_t',
            amplitude=[1] * 8 + [4] + [1] * 7,
            mean=[0] * 16, variance=[1] * 8 + [16] + [1] * 7,
            distinct_time_covariances=0, repeated_pulse_imposed=False,
            conclusion='variance_nonstationarity_is_not_a_rhythm_label', listener_label=None),
        rank_arithmetic=dict(draw_count=99, minimum_tail=rank_tail(1.0, [0.0] * 99),
                             all_tied_tail=rank_tail(1.0, [1.0] * 99),
                             full_extraction_forward_sets=100,
                             actual_test_performed=False, rhythm_probability=None),
        valid_no_support_reference_established=False,
        producer_gate=prior['producer_gate'],
        new_audio=False, new_inference=False, feature_access=False, holdout_access=False,
        training=False, training_necessity_proven=False, production_change=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths = parser.add_mutually_exclusive_group(required=True)
    paths.add_argument('--output', type=Path)
    paths.add_argument('--check', type=Path)
    args = parser.parse_args()
    report = build_report()
    if args.check:
        require(json.loads(args.check.read_text(encoding='utf-8')) == report, 'retained report differs')
    else:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
