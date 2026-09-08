"""Frozen seven-pair pre-head recurrence replay; no holdout or model fitting.

All original PCM and dense-head identities must match before feature inference.
Native trace files and feature arrays stay in a new private directory.
"""
import argparse
from bisect import bisect_left
from fractions import Fraction
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import prehead_capture as capture
import prehead_capture_audit as fidelity
import prehead_recurrence as recurrence

oracle = recurrence.oracle
require = capture.require
HERE, ROOT = fidelity.HERE, fidelity.ROOT
LOCK_PATH = HERE / 'prehead-recurrence-lock-v1.json'
PINS = {
    'prehead_capture.py': 'b11224ca2cc99934e30e67144a83ad890e38b3c0bedbba38d4b09d082db2c233',
    'prehead_capture_audit.py': '7f48559c4098796a2b61ddab3916d1c61bae43455d46f04ff1551517265a1724',
    'prehead-capture-v1.json': 'f2102630d9e5b36d14cea43ea8a0121df2dfabf5c2ccfd57b7df0b8cdaf4d90a',
    'oracle_period_paired_audit.py': '2086a72728279375158cad5f73178f9e093afd4541d7ce2c36034d6b3ff22732',
    'oracle-period-paired-lock-v1.json': 'c92cbb9fdc2659e00dfa4c0e460d934a1e9967336ea16615cd509d4e7173c635',
}
TRACE_SOURCES = {
    'trace_exporter_sha256': 'crates/rhythm-map-eval/examples/beat_this_trace.rs',
    'trace_support_sha256': 'crates/rhythm-map-eval/examples/support/mod.rs',
    'adapter_source_sha256': 'crates/rhythm-map-beat-this/src/lib.rs',
    'audio_preprocessing_sha256': 'crates/rhythm-map-beat-this/src/audio.rs',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_plan():
    require(all(sha(HERE / p) == h for p, h in PINS.items()), 'frozen helper changed')
    require(all(sha(HERE / p) == h for p, h in oracle.PINS.items()), 'frozen oracle dependency changed')
    lock = json.loads(LOCK_PATH.read_bytes())
    require(lock['schema_version'] == 1 and lock['budgets'] == dict(window_frames=200,
            width=512, numerical_tie=recurrence.MARGIN_EPS, zero_norm=recurrence.NORM_EPS),
            'recurrence numerical contract changed')
    require(all(lock[k] is False for k in ('real_features_opened_for_discrimination',
            'training', 'holdout_access', 'production_changed', 'new_user_parameters')),
            'recurrence preregistration scope changed')
    helpers, packet_report = oracle.replay.verified_helpers()
    metadata, cases, plan = oracle.replay.verified_plan()
    require(len(cases) == 40 and len(plan) == 7, 'paired population changed')
    return lock, helpers, packet_report, metadata, cases, plan


def validate_trace(trace, case, record, old, source_hashes):
    import compare_reference
    compare_reference.validate_trace(trace, oracle.dense.MODEL)
    require(all(trace.get(k) == v for k, v in source_hashes.items()), 'trace implementation changed')
    require(trace['suite_id'] == 'artbeat-v1' and trace['suite_sha256'] == oracle.dense.SUITES['artbeat'][1]
            and trace['case_id'] == case['id'] and trace['audio_sha256'] == case['audio_sha256'] and
            trace['observation_contract'] == old['observation_contract'] and
            trace['prefix_seconds'] == 60 and trace['sample_rate'] == case['sample_rate'] == 22050 and
            trace['decoded_sample_count'] == len(trace['mono_samples']) == case['sample_count'],
            'trace is not the same complete recording')
    require(capture.array_hash(np.asarray(trace['mono_samples'], np.float32)) == case['pcm_sha256'],
            'frozen complete PCM changed')
    # The old dense exporter uses load(), while the trace exporter uses
    # load_with_model_identity(). Verify their exact known provenance spelling;
    # do not ignore arbitrary source drift or relax any event/numerical check.
    require(old['observations']['source'] == dict(backend='beat-this-rten', frame_rate_hz=50.,
            model='beat_this.onnx', version=None) and trace['observations']['source'] ==
            dict(backend='beat-this-rten', frame_rate_hz=50., model='beat-this-full-rten',
                 version=oracle.dense.MODEL), 'unexpected exporter provenance')
    require(trace['mel_shape'] == [1, record['frame_count'], 128] and
            {k: v for k, v in trace['observations'].items() if k != 'source'} ==
            {k: v for k, v in old['observations'].items() if k != 'source'},
            'frame or observation identity changed')
    for key in ('beat_logits', 'downbeat_logits'):
        capture.difference(np.asarray(trace[key], np.float32), np.asarray(old[key], np.float32), exact=True)


def evaluate_pair(hidden, owner, data, pair):
    windows, geometry = {}, {}
    times = list(map(oracle.coordinate, data['beats']))
    pre = Fraction(3000) / Fraction(str(data['segments'][pair['pre_segment_index']]['bpm']))
    post = Fraction(3000) / Fraction(str(data['segments'][pair['pre_segment_index'] + 1]['bpm']))
    for role in ('constant', 'change'):
        start = pair[role + '_start_frame']
        end = start + pair['window_frames']
        require(0 <= start < end <= len(hidden), 'paired feature coverage changed')
        prepared = oracle.prepare(data, pair, role, {s: [] for s in oracle.replay.SOURCES})
        require(prepared[0]['status'] == 'eligible', 'frozen clock geometry unavailable')
        geometry[role] = [c['geometry'] for c in prepared[2]]
        before = bisect_left(times, start)
        windows[role] = [recurrence.evaluate(hidden[start:end], owner[start:end],
            times[before - 4 + i] + (3 - i) * pre - start, pre, post, prepared[3]) for i in range(4)]
    verdict = recurrence.verdict(windows)
    compatible = all(c['compatible'] for rows in geometry.values() for c in rows)
    # Even an otherwise eligible contrast is not interpretable after clock drift.
    if not compatible:
        verdict = dict(status='geometry_incompatible', shape_phases=0, density_phases=0,
                       shape_supported=False, density_resolved=False)
    if verdict['status'] == 'pair_rejected':
        for rows in windows.values():
            for row in rows:
                row['clocks'] = None  # No orphan winning side after a rejection.
    return dict(geometry=geometry, geometry_compatible=compatible, windows=windows, verdict=verdict)


def load_model(upstream, checkpoint_path, lock, reference):
    require(subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD']).decode().strip() ==
            reference['reference_revision'] and not subprocess.check_output(
                ['git', '-C', str(upstream), 'status', '--porcelain']), 'upstream not clean and pinned')
    require(checkpoint_path.stat().st_size == reference['checkpoint']['size_bytes'], 'checkpoint size changed')
    data = checkpoint_path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == reference['checkpoint']['sha256'], 'checkpoint hash changed')
    require({k: importlib.metadata.version(k) for k in lock['versions']} == lock['versions'], 'runtime changed')
    require(not any(k == 'beat_this' or k.startswith('beat_this.') for k in sys.modules), 'upstream already imported')
    sys.path.insert(0, str(upstream))
    import torch
    from beat_this import inference
    from beat_this.model.beat_tracker import BeatThis
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.utils import replace_state_dict_key
    require(Path(inference.__file__).resolve().is_relative_to(upstream), 'wrong upstream package')
    torch.set_num_threads(lock['threads'])
    torch.set_num_interop_threads(lock['interop_threads'])
    checkpoint = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    hparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in inspect.signature(BeatThis).parameters}
    model = BeatThis(**hparams).eval()
    model.load_state_dict(replace_state_dict_key(checkpoint['state_dict'], 'model.', ''))
    require(type(model.task_heads).__name__ == 'SumHead' and
            type(model.transformer_blocks.norm).__name__ == 'RMSNorm' and
            tuple(model.task_heads.beat_downbeat_lin.weight.shape) == (2, 512), 'wrong architecture')
    return model, inference, Postprocessor(type='minimal', fps=50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trace-executable', 'audio-dir', 'model-dir', 'upstream', 'checkpoint',
                 'artbeat-evidence', 'dense-captures', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    lock, helpers, packet_report, metadata, cases, plan = verified_plan()
    capture_lock, reference, _ = fidelity.verified_plan()
    output = fidelity.private_directory(args.output_dir)
    evidence, _ = oracle.dense.read_json(args.artbeat_evidence, oracle.dense.INPUTS['artbeat'][1])
    prior = next(c for c in packet_report['cohorts'] if c['cohort'] == 'artbeat')
    summary, _ = oracle.dense.read_json(args.dense_captures / 'summary.json', prior['capture_summary_sha256'])
    records = oracle.dense.validate_summary(summary, evidence['cases'], 15)
    require([c['id'] for c in evidence['cases']] == [c['id'] for c in metadata[0]['cases']], 'ordered identity changed')
    lookup = {c['id']: (c, r) for c, r in zip(evidence['cases'], records)}
    by_id = {c['identity']['id']: c for c in cases}
    source_hashes = {k: sha(ROOT / p) for k, p in TRACE_SOURCES.items()}
    dense_sources = {k: sha(ROOT / p) for k, p in oracle.dense.SOURCES.items()}
    payloads = {}
    for selected in plan:
        case, record = lookup[selected['id']]
        require(record['capture_sha256'] == by_id[case['id']]['identity']['capture_sha256'] and
                case['truth_sha256'] == by_id[case['id']]['identity']['truth_sha256'], 'selected identity changed')
        old, _ = oracle.dense.read_json(args.dense_captures / (case['id'] + '.json'), record['capture_sha256'])
        oracle.dense.validate_capture(old, record, case, 'artbeat', dense_sources)
        payloads[case['id']] = old
    output.mkdir(mode=0o700)
    # Trace every selected complete recording; compare all of them before taking
    # any hidden features. This is never a new crop/context or favorable subset.
    trace_hashes = {}
    for selected in plan:
        case, record = lookup[selected['id']]
        trace_path = output / (case['id'] + '.trace.private.json')
        print('Native parity trace: ' + case['id'], flush=True)
        subprocess.run([str(args.trace_executable.resolve()), '--suite', str(ROOT / 'evaluation/suites/artbeat-v1.json'),
            '--case', case['id'], '--audio-dir', str(args.audio_dir.resolve()), '--model-pack',
            str(ROOT / 'models/beat-this-full-v1.json'), '--model-dir', str(args.model_dir.resolve()),
            '--seconds', '60', '--output', str(trace_path)], check=True, cwd=ROOT,
            env=dict(os.environ, RTEN_NUM_THREADS='2'))
        trace = json.loads(trace_path.read_bytes())
        validate_trace(trace, case, record, payloads[case['id']], source_hashes)
        trace_hashes[case['id']] = sha(trace_path)
    model, inference, postprocess = load_model(args.upstream.resolve(), args.checkpoint, capture_lock, reference)
    state_before = fidelity.model_state_hash(model)
    started = time.perf_counter()
    captures = []
    for selected in plan:
        case_id = selected['id']
        print('Whole-recording pre-head capture: ' + case_id, flush=True)
        trace = fidelity.read_verified(output / (case_id + '.trace.private.json'), trace_hashes[case_id])
        mel = np.asarray(trace['mel_values'], np.float32).reshape(trace['mel_shape'])[0]
        captures.append(fidelity.run_case(model, inference, postprocess, mel, case_id, capture_lock, output, trace))
    require(fidelity.model_state_hash(model) == state_before, 'capture mutated model state')
    results = []
    for selected, captured in zip(plan, captures):
        path = output / (selected['id'] + '.private.npz')
        require(sha(path) == captured['private_archive']['sha256'], 'feature archive changed')
        with np.load(path, allow_pickle=False) as archive:
            require(all(capture.array_hash(archive[k]) == v for k, v in captured['array_sha256'].items()),
                    'feature array changed')
            result = evaluate_pair(archive['hidden'], archive['owner'], by_id[selected['id']]['data'], selected)
        results.append(dict(id=selected['id'], pair_sha256=oracle.exposure.canonical_hash(selected), **result))
    selected_ids = {p['id'] for p in plan}
    report = dict(schema_version=1, purpose=lock['purpose'], complete=True,
        contract=lock, lock_sha256=sha(LOCK_PATH), source_sha256={p: sha(HERE / p) for p in
            ('prehead_recurrence.py', 'prehead_recurrence_audit.py')}, helper_sha256=PINS | helpers | oracle.PINS,
        trace_sources=source_hashes, trace_sha256=trace_hashes,
        observation_provenance='exact known filename/null to manifest-id/hash spelling; every other field exact',
        capture_lock_sha256=sha(fidelity.LOCK_PATH),
        reference_revision=reference['reference_revision'], checkpoint_sha256=reference['checkpoint']['sha256'],
        versions=capture_lock['versions'], rten_threads=2, threads=capture_lock['threads'],
        interop_threads=capture_lock['interop_threads'], trace_executable_sha256=sha(args.trace_executable),
        selected_plan_sha256=oracle.replay.LOCK['selected_plan_sha256'],
        input_cases=[dict(c['identity'], cohort=c['cohort'], feature_capture=c['identity']['id'] in selected_ids,
            disposition='selected_pair' if c['identity']['id'] in selected_ids else
                        'expressive_untyped' if c['data']['untyped'] else 'no_registered_pair') for c in cases],
        captures=captures, pairs=results, model_state_unchanged=True,
        summary=dict(selected_pairs=7, eligible_pairs=sum(p['verdict']['status'] == 'eligible' for p in results),
            geometry_compatible_pairs=sum(p['geometry_compatible'] for p in results),
            robust_shape_pairs=sum(p['verdict']['shape_supported'] for p in results),
            robust_density_pairs=sum(p['verdict']['density_resolved'] for p in results)),
        elapsed_capture_and_comparison_s=time.perf_counter() - started,
        real_features_opened_for_discrimination=True, training=False, holdout_access=False,
        production_changed=False, new_user_parameters=False)
    report['decision'] = ('independent_automatic_gate_required' if
        report['summary']['eligible_pairs'] == report['summary']['robust_shape_pairs'] ==
        report['summary']['robust_density_pairs'] == 7 else 'do_not_promote_fixed_recurrence_rule')
    with (output / 'report.json').open('x', encoding='utf-8', newline='\n') as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write('\n')
    print(json.dumps(report['summary']), flush=True)
    print(report['decision'], flush=True)


if __name__ == '__main__':
    main()
