"""Retrospective PCM-availability composition; never a new music/tempo decoder."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

import musicfm_fidelity as fidelity
import musicfm_loader as loader
import musicfm_temporal as temporal
import musicfm_temporal_probe as previous

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK_SHA = '49f49861aef1b7c22268418cdb74cd3113d4baa2c68db1a7538a75960d45d824'


def protocol():
    raw = (HERE / 'musicfm-availability-lock-v1.json').read_bytes()
    loader.require(loader.digest(raw) == LOCK_SHA, 'availability protocol changed')
    plan = loader.strict_json(raw)
    for name, sha in plan['helpers'].items():
        loader.require(loader.digest((HERE / name).read_bytes()) == sha, 'previous helper changed: ' + name)
    for entry in plan['native_sources'].values():
        loader.require(loader.digest((ROOT / entry['path']).read_bytes()) == entry['sha256'], 'native source changed')
    previous.protocol()
    raw = (HERE / 'musicfm-temporal-v1.json').read_bytes()
    loader.require(loader.digest(raw) == plan['previous_report_sha256'], 'previous failure report changed')
    return plan, loader.strict_json(raw)


def controls():
    output = {name: temporal.authored_pcm(name) for name in temporal.CASES}
    clean = output['constant-clean']
    output['quiet-gain'] = clean * np.float32(2 ** -16)
    for name, start, stop in (('short-rest', 138000, 150000), ('long-rest', 96000, 192000),
                               ('prefix-rest', 0, 72000), ('suffix-rest', 216000, 288240)):
        output[name] = clean.copy()
        output[name][start:stop] = 0
    t = np.arange(288240, dtype=np.float64) / 24000
    output['steady-tone'] = (0.125 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    output['dc'] = np.full(288240, 0.125, np.float32)
    output['noise'] = np.random.Generator(np.random.PCG64(142)).uniform(-0.125, 0.125, 288240).astype(np.float32)
    output['below-floor'] = clean * np.float32(2 ** -60)
    return output


def validate_native(packet, pcm_sha, samples, plan):
    loader.require(packet['schema'] == 'rhythm-map.native-activity.v1' and packet['samples'] == samples and
                   packet['sample_rate'] == 24000 and packet['pcm_sha256'] == pcm_sha, 'native PCM identity differs')
    loader.require(packet['source_sha256'] == {k: v['sha256'] for k, v in plan['native_sources'].items()},
                   'binary embedded source identity differs')
    loader.require(packet['method'] == 'one-sentinel-membership-query-per-native-center-not-beat-evidence' and
                   packet['pretrained_inference'] is False and packet['production_change'] is False,
                   'unexpected native method')
    for key in ('silence_threshold_db', 'minimum_silence_s'):
        loader.require(packet[key] == plan['native_contract'][key], 'shipping activity default changed')
    activity = packet['activity']
    loader.require(type(activity) is list and len(activity) == (samples + 1199) // 1200,
                   'incomplete native activity envelope')
    for i, point in enumerate(activity):
        loader.require(set(point) == {'time_s', 'rms', 'relative_db'} and all(
            type(v) in (int, float) and math.isfinite(v) for v in point.values()) and
            abs(point['time_s'] - i / 20) <= 1e-12 and point['rms'] >= 0 and -120 <= point['relative_db'] <= 0,
            'invalid native activity cell')
    mask = packet['native_token_retained']
    loader.require(type(mask) is list and len(mask) == temporal.tokens(samples) and
                   all(type(v) is bool or v is None for v in mask), 'invalid native availability mask')
    return mask


def availability(mask, start, post):
    """One shared veto, no per-clock favorable subset or changes to recurrence."""
    loader.require(type(mask) is list and all(type(v) is bool or v is None for v in mask), 'invalid availability mask')
    loader.require(type(start) is int and start >= 0 and start + 100 <= len(mask), 'uncovered complete window')
    targets, queries, _ = temporal.queries(0, Fraction(25, 2), post, 50)
    required = set(targets)
    for rows in queries.values():
        for t in targets:
            for coordinate in rows[t]:
                required.update((math.floor(coordinate), math.ceil(coordinate)))
    values = [mask[start + i] for i in sorted(required)]
    unknown = sum(v is None for v in values)
    inactive = sum(v is False for v in values)
    status = 'unknown_activity' if not values or unknown else 'low_activity' if inactive else 'available_for_periodic_test'
    return dict(status=status, required_native_centers=len(values), unknown_centers=unknown,
                low_activity_centers=inactive, target_frames=len(targets), score_support_modified=False)


def score_if_available(gate, callback):
    loader.require(gate['status'] in ('unknown_activity', 'low_activity', 'available_for_periodic_test'),
                   'invalid availability state')
    if gate['status'] != 'available_for_periodic_test':
        return dict(status=gate['status'], clocks=None, feature_access=False)
    row = callback()
    return dict(row, feature_access=True)


def cached_features(directory, name, plan, baseline):
    cache = plan['cache']
    arrays = []
    for kind, size, sha, shape, dtype in (
            ('features', cache['feature_file_bytes'], cache['features'][name], (301, 1024), np.dtype('float32')),
            ('owners', cache['owner_file_bytes'], cache['owner_file_sha256'], (301,), np.dtype('int32'))):
        # Verify bounded full file bytes on the same handle BEFORE parsing NPY.
        with loader.verified_file(directory / (name + '.' + kind + '.npy'), size, sha) as stream:
            value = np.load(stream, allow_pickle=False, max_header_size=4096)
        loader.require(type(value) is np.ndarray and value.dtype == dtype and value.shape == shape and
                       np.isfinite(value).all(), 'invalid frozen feature archive')
        key = 'feature_sha256' if kind == 'features' else 'owner_sha256'
        loader.require(loader.digest(value.tobytes()) == baseline[key], 'frozen tensor identity differs')
        arrays.append(value)
    expected_owner = np.full(301, -1, np.int32)
    for chunk in temporal.chunks(288240):
        expected_owner[chunk['own_start']:chunk['own_stop']] = chunk['id']
    loader.require(np.array_equal(arrays[1], expected_owner), 'frozen context ownership differs')
    return arrays


def run_native(executable, output, name, pcm, plan):
    loader.require(pcm.dtype == np.float32 and pcm.shape == (288240,) and np.isfinite(pcm).all(), 'invalid authored PCM')
    raw = pcm.astype('<f4', copy=False).tobytes()
    sha = loader.digest(raw)
    path, destination = output / (name + '.f32le'), output / (name + '.activity.json')
    with path.open('xb') as stream:
        stream.write(raw)
    subprocess.run([str(executable), '--pcm', str(path), '--pcm-sha256', sha, '--output', str(destination)],
                   check=True, timeout=plan['native_timeout_seconds_per_case'], capture_output=True)
    packet = loader.strict_json(destination.read_bytes())
    return packet, validate_native(packet, sha, len(pcm), plan), sha


def audit(executable, features, output):
    plan, old = protocol()
    loader.require(np.__version__ == plan['numpy_version'], 'NumPy runtime changed')
    pcm_cases = controls()
    ids = plan['original_cases'] + [c['id'] for c in plan['new_activity_controls']]
    loader.require(list(pcm_cases) == ids, 'authored population differs')
    output.mkdir()
    report = dict(schema='rhythm-map.musicfm-availability.v1', complete=False, decision='not_completed',
                  protocol_sha256=LOCK_SHA, source_sha256=loader.digest(Path(__file__).read_bytes()),
                  native_sources=plan['native_sources'], native_binary_sha256=loader.digest(executable.read_bytes()),
                  previous_report_sha256=plan['previous_report_sha256'], cases=[],
                  new_pretrained_inference=False, music_access=False, holdout_access=False, training=False,
                  production_change=False, new_user_parameters=False, independent_acceptance=False)
    started = time.perf_counter()
    try:
        # All new PCM activity controls are checked before replaying known features.
        for name, pcm in pcm_cases.items():
            packet, mask, sha = run_native(executable, output, name, pcm, plan)
            original = name in plan['original_cases']
            if original:
                baseline = next(c for c in old['cases'] if c['id'] == name)
                loader.require(sha == baseline['pcm_sha256'], 'old PCM recipe no longer exact')
                start, expected = 100, 'low_activity' if name == 'silence' else 'available_for_periodic_test'
            else:
                control = next(c for c in plan['new_activity_controls'] if c['id'] == name)
                start, expected = control['window_start'], control['expected']
            gates = {speed: availability(mask, start, period) for speed, period in
                     (('fast', Fraction(25, 4)), ('slow', Fraction(20)))}
            row = dict(id=name, pcm_sha256=sha, samples=len(pcm), activity_points=len(packet['activity']),
                       native_tokens=len(mask), retained_tokens=sum(v is True for v in mask),
                       low_activity_tokens=sum(v is False for v in mask), unknown_tokens=sum(v is None for v in mask),
                       activity_min_relative_db=min(p['relative_db'] for p in packet['activity']),
                       activity_max_relative_db=max(p['relative_db'] for p in packet['activity']),
                       window_start=start, availability=gates, expected_state=expected,
                       activity_contract_pass=all(g['status'] == expected for g in gates.values()),
                       feature_access=False, periodicity_evaluated=False)
            report['cases'].append(row)
            loader.require(row['activity_contract_pass'], 'authored activity contract failed: ' + name)
            print(json.dumps(dict(case=name, activity_contract_pass=True)), flush=True)
        scores = {}
        for row in report['cases'][:6]:
            name = row['id']
            loaded = []

            def evaluate(period):
                if not loaded:
                    baseline = next(c for c in old['cases'] if c['id'] == name)
                    loaded.extend(cached_features(features, name, plan, baseline))
                return temporal.evaluate(loaded[0][100:200], loaded[1][100:200], 0, Fraction(25, 2), period, 50)

            scores[name] = {}
            for speed, period in (('fast', Fraction(25, 4)), ('slow', Fraction(20))):
                score = score_if_available(row['availability'][speed], lambda: evaluate(period))
                if score['feature_access']:
                    loader.require({k: v for k, v in score.items() if k != 'feature_access'} ==
                                   old['discrimination']['scores'][name][speed], 'retained score changed')
                scores[name][speed] = score
            row.update(feature_access=bool(loaded), periodicity_evaluated=bool(loaded),
                       retained_scores_exact=bool(loaded), scores=scores[name])
        pairs = [dict(constant=name, change='step-' + speed,
                      **temporal.verdict(scores[name][speed], scores['step-' + speed][speed]))
                 for name in temporal.CASES[:3] for speed in ('fast', 'slow')]
        loader.require(len(pairs) == 6 and all(p['shape_supported'] and p['density_resolved'] for p in pairs),
                       'original paired verdicts changed')
        silence = scores['silence']
        loader.require(all(r['status'] == 'low_activity' and r['clocks'] is None and not r['feature_access']
                           for r in silence.values()), 'known silence was not rejected before features')
        report.update(complete=True, decision='known_counterexample_composition_pass_not_independent_acceptance',
                      paired_denominator=6, pairs=pairs, activity_case_denominator=len(ids),
                      feature_cases_replayed=5, known_silence_rejected_before_feature_access=True,
                      previous_failure_preserved=True)
    except Exception as error:
        report.update(failure_type=type(error).__name__, decision='stopped_without_composition_acceptance')
        raise
    finally:
        report['elapsed_s'] = time.perf_counter() - started
        with (output / 'report.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-executable', required=True, type=Path)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = fidelity.private_output(args.output)
    audit(args.native_executable.resolve(), args.features.resolve(), output)


if __name__ == '__main__':
    main()
