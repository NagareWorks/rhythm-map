"""Conditional clock discrimination on the old seven pairs, never admission."""
from bisect import bisect_left
from fractions import Fraction

import numpy as np

import musicfm_temporal as temporal
import prehead_recurrence_audit as prior

require, oracle = temporal.require, prior.oracle


def pair_result(hidden, owner, data, pair):
    """Reuse physical windows/geometry; no period, phase or boundary fitting."""
    require(isinstance(hidden, np.ndarray) and hidden.dtype == np.float32 and
            hidden.ndim == 2 and hidden.shape[1] == temporal.WIDTH and np.isfinite(hidden).all(),
            'invalid complete features')
    require(isinstance(owner, np.ndarray) and owner.dtype == np.int32 and
            owner.shape == (len(hidden),) and np.all(owner >= 0) and
            np.all(np.diff(owner.astype(np.int64)) >= 0), 'invalid complete ownership')
    times = list(map(oracle.coordinate, data['beats']))
    segment = pair['pre_segment_index']
    pre = Fraction(3000) / Fraction(str(data['segments'][segment]['bpm']))
    post = Fraction(3000) / Fraction(str(data['segments'][segment + 1]['bpm']))
    windows, geometry, coordinates = {}, {}, {}
    for role in oracle.replay.ROLES:
        start = pair[role + '_start_frame']
        prepared = oracle.prepare(data, pair, role, {s: [] for s in oracle.replay.SOURCES})
        require(prepared[0]['status'] == 'eligible', 'frozen geometry unavailable')
        geometry[role] = [c['geometry'] for c in prepared[2]]
        before = bisect_left(times, start)
        require(before >= 4, 'missing fixed prefix anchors')
        windows[role], coordinates[role] = [], []
        for i in range(4):
            anchor = times[before - 4 + i] + (3 - i) * pre
            coord = temporal.window_from_50hz(start, anchor, pre, post, start + prepared[3])
            left, right = coord['token_start'], coord['token_stop']
            require(0 <= left < right <= len(hidden), 'paired feature coverage changed')
            windows[role].append(temporal.evaluate(hidden[left:right], owner[left:right],
                coord['anchor'], coord['pre'], coord['post'], coord['boundary']))
            coordinates[role].append({k: str(v) if isinstance(v, Fraction) else v for k, v in coord.items()})
    compatible = all(g['compatible'] for rows in geometry.values() for g in rows)
    phases = [temporal.verdict(windows['constant'][i], windows['change'][i]) for i in range(4)]
    status = ('geometry_incompatible' if not compatible else 'pair_rejected'
              if any(p['status'] != 'eligible' for p in phases) else 'eligible')
    if status != 'eligible':
        # Do not retain an orphan favorable side when the pair cannot be judged.
        for rows in windows.values():
            for row in rows:
                row['clocks'] = None
    return dict(coordinates=coordinates, geometry=geometry, geometry_compatible=compatible,
                windows=windows, verdict=dict(status=status,
                    shape_supported=status == 'eligible' and all(p['shape_supported'] for p in phases),
                    density_resolved=status == 'eligible' and all(p['density_resolved'] for p in phases)))


def summarize(pairs, expected_ids):
    require([p['id'] for p in pairs] == list(expected_ids) and len(set(expected_ids)) == 7,
            'incomplete, reordered or replaced paired population')
    return dict(selected_pairs=7,
                eligible_pairs=sum(p['verdict']['status'] == 'eligible' for p in pairs),
                both_native_shapes=sum(p['verdict']['shape_supported'] for p in pairs),
                all_density_rivals=sum(p['verdict']['density_resolved'] for p in pairs),
                product_admission=False, automatic_accuracy_measured=False,
                independent_acceptance=False, no_rhythm_failure_resolved=False)
