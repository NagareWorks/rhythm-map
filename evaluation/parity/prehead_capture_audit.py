"""Frozen CPU capture-fidelity gate, not musical feature selection or training.

Run four authored spectrograms and exactly the two original reference traces.
Only aggregate checks leave the private output directory. No automatic fetches.
"""
import argparse
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
from prehead_capture import require

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / 'prehead-capture-lock-v1.json'
CASE_IDS = ('artbeat-05-75-to-150', 'rubato-bach-bwv1007-01-ar-macleod2011')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_verified(path, expected):
    data = Path(path).read_bytes()
    require(sha(data) == expected, 'input byte identity changed')
    return json.loads(data)


def verified_plan():
    lock = json.loads(LOCK_PATH.read_bytes())
    reference = read_verified(HERE / 'reference-lock.json', lock['reference_lock_sha256'])
    baseline = read_verified(HERE / 'baseline-v1.json', lock['reference_report_sha256'])
    require(sha((HERE / 'compare_reference.py').read_bytes()) == lock['comparator_sha256'],
            'reference comparator changed')
    require(lock['schema_version'] == 1 and lock['purpose'] == 'capture_fidelity_not_discrimination' and
            lock['feature'] == 'transformer_blocks_output_after_rms_norm_before_task_heads' and
            (lock['feature_width'], lock['chunk_frames'], lock['border_frames'], lock['stride_frames'],
             lock['max_frames'], lock['dtype'], lock['device'], lock['ownership']) ==
            (512, 1500, 6, 1488, 4096, 'float32', 'cpu', 'keep_first') and
            (lock['threads'], lock['interop_threads']) == (2, 1), 'capture contract changed')
    require([(r['id'], r['frames'], r['kind']) for r in lock['authored_controls']] ==
            [('short_flat', 17, 'zero'), ('single_full', 1488, 'modular'),
             ('first_overlap', 1489, 'modular'), ('three_chunks', 2980, 'modular')],
            'authored controls changed')
    require(lock['budgets'] == dict(tap_vs_untapped='bit_exact', head_reconstruction_atol=2e-5,
            head_reconstruction_rtol=1e-6, mel_atol=1e-3, reference_logit_atol=2e-3,
            reference_rtol=1e-4, reference_event_atol_s=1e-5,
            event_identity='same_ordered_half_frame_indices'), 'parity budgets changed')
    require(tuple(c['case_id'] for c in baseline['cases']) == CASE_IDS and
            tuple(c['case'] for c in reference['cases']) == CASE_IDS and
            baseline['checkpoint_sha256'] == reference['checkpoint']['sha256'] and
            baseline['model_manifest_sha256'] == reference['model_manifest_sha256'],
            'reference population/identity changed')
    return lock, reference, baseline


def authored_mel(frames, kind):
    capture.layout(frames)
    require(kind in ('zero', 'modular'), 'unknown authored input')
    if kind == 'zero':
        return np.zeros((frames, 128), np.float32)
    # Deterministic artificial input, not an audio renderer or labeled music.
    t, f = np.arange(frames)[:, None], np.arange(128)[None, :]
    return (((17 * t + 29 * f) % 257) / 64).astype(np.float32)


def private_directory(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(ROOT) and not path.exists(), 'use a new private directory outside Git')
    if sys.platform == 'win32':
        require(path.drive.lower() != os.environ.get('SystemDrive', 'C:').lower(),
                'private captures must stay off the system drive')
    require(path.parent.is_dir(), 'private parent directory must exist')
    return path


def model_state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def hook_failure_control(torch):
    class Head(torch.nn.Module):
        def forward(self, x):
            if self._forward_pre_hooks:
                raise RuntimeError('authored tapped failure')
            return dict(beat=x[:, :, 0], downbeat=x[:, :, 0])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.task_heads = Head()

        def forward(self, x):
            return self.task_heads(x.repeat(1, 1, 4))

    model = Model().eval()
    try:
        capture.capture_chunk(model, np.zeros((17, 128), np.float32), 2e-5, 1e-6)
    except RuntimeError as error:
        require(str(error) == 'authored tapped failure', 'unexpected hook exception')
    else:
        raise ValueError('failure control did not fail')
    require(not model.task_heads._forward_pre_hooks, 'hook leaked after failure')
    return dict(injected_exception_observed=True, hook_removed=True)


def run_case(model, inference, postprocess, mel, case_id, lock, output, trace=None):
    import torch
    rows, split_mel = capture.layout(len(mel)), capture.split(mel)
    before_hash = capture.array_hash(mel)
    upstream_chunks, upstream_starts = inference.split_piece(torch.from_numpy(mel), 1500, 6, True)
    require(list(upstream_starts) == [r['start'] for r in rows], 'chunk starts differ from upstream')
    for ours, upstream in zip(split_mel, upstream_chunks):
        capture.difference(ours, upstream.numpy(), exact=True)
    chunks, baseline, chunk_checks = [], [], []
    for chunk in split_mel:
        data, base, checks = capture.capture_chunk(model, chunk,
            lock['budgets']['head_reconstruction_atol'], lock['budgets']['head_reconstruction_rtol'])
        chunks.append(data)
        baseline.append({key: torch.from_numpy(value) for key, value in base.items()})
        chunk_checks.append(checks)
    require(capture.array_hash(mel) == before_hash, 'mel input mutated')
    arrays = capture.stitch(len(mel), chunks)
    reference_heads = inference.aggregate_prediction(baseline, upstream_starts, len(mel),
                                                     1500, 6, 'keep_first', 'cpu')
    checks = {}
    for key, value in zip(('beat', 'downbeat'), reference_heads):
        checks[key + '_aggregate_vs_upstream'] = capture.difference(arrays[key], value.numpy(), exact=True)
    with torch.inference_mode():
        base_events = postprocess(beat=reference_heads[0], downbeat=reference_heads[1])
        tapped_events = postprocess(**{k: torch.from_numpy(arrays[k]) for k in ('beat', 'downbeat')})
    for key, ours, base in zip(('beat', 'downbeat'), tapped_events, base_events):
        checks[key + '_events_vs_untapped'] = capture.event_identity(ours, base, atol=0.0)
    if trace is not None:
        for key in ('beat', 'downbeat'):
            checks[key + '_vs_frozen_rten'] = capture.difference(arrays[key],
                np.asarray(trace[key + '_logits'], np.float32), atol=lock['budgets']['reference_logit_atol'],
                rtol=lock['budgets']['reference_rtol'])
        for key, ours in zip(('beats', 'downbeats'), tapped_events):
            checks[key + '_vs_frozen_port'] = capture.event_identity(
                ours, trace['upstream_' + key], lock['budgets']['reference_event_atol_s'])
        pcm = np.asarray(trace['mono_samples'], np.float32)
        from beat_this.preprocessing import LogMelSpect
        with torch.inference_mode():
            official_mel = LogMelSpect(device='cpu')(torch.from_numpy(pcm)).numpy()
        checks['official_frontend_vs_frozen_mel'] = capture.difference(
            official_mel, mel, atol=lock['budgets']['mel_atol'], rtol=lock['budgets']['reference_rtol'])
        # Also verify the old adapter's event projection; no new adapter run is claimed.
        for index, key in enumerate(('beats', 'downbeats')):
            retained = [b['time_s'] for b in trace['observations']['beats']
                        if index == 0 or b['downbeat_confidence'] >= 0.5]
            checks[key + '_vs_frozen_adapter'] = capture.event_identity(
                tapped_events[index], retained, lock['budgets']['reference_event_atol_s'])
    archive = capture.save_capture(output / (case_id + '.private.npz'), arrays)
    return dict(case_id=case_id, frames=len(mel), hidden_shape=list(arrays['hidden'].shape),
                mel_sha256=before_hash, chunk_count=len(rows),
                owner_frame_counts=[int(np.sum(arrays['owner'] == i)) for i in range(len(rows))],
                array_sha256={k: capture.array_hash(v) for k, v in arrays.items()},
                chunk_checks=chunk_checks, checks=checks, private_archive=archive,
                process_lifetime_peak_rss_bytes=capture.peak_rss_bytes(),
                pcm_sha256=capture.array_hash(np.asarray(trace['mono_samples'], np.float32)) if trace else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    lock, reference, baseline = verified_plan()
    require(len(args.trace) == 2, 'exactly the two original reference traces are required')
    output = private_directory(args.output_dir)
    upstream = args.upstream.resolve()
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(upstream), *arguments])
    require(git('rev-parse', 'HEAD').decode().strip() == reference['reference_revision'] and
            not git('status', '--porcelain'), 'upstream must be clean and pinned')
    traces = [read_verified(path, case['trace_sha256']) for path, case in zip(args.trace, baseline['cases'])]
    require(args.checkpoint.stat().st_size == reference['checkpoint']['size_bytes'], 'checkpoint size changed')
    checkpoint_bytes = args.checkpoint.read_bytes()
    require(sha(checkpoint_bytes) == reference['checkpoint']['sha256'], 'checkpoint bytes changed')
    versions = {key: importlib.metadata.version(key) for key in lock['versions']}
    require(versions == lock['versions'], 'reference dependency versions changed')
    import compare_reference as comparator
    for trace, case in zip(traces, baseline['cases']):
        comparator.validate_trace(trace, reference['model_manifest_sha256'])
        require(trace['case_id'] == case['case_id'] and trace['audio_sha256'] == case['audio_sha256'] and
                len(trace['mono_samples']) == case['sample_count'] and trace['mel_shape'][1] == case['mel_frames'],
                'reference scope changed')
    require(not any(k == 'beat_this' or k.startswith('beat_this.') for k in sys.modules),
            'upstream package already imported; use a fresh process')
    sys.path.insert(0, str(upstream))
    import torch
    from beat_this import inference
    from beat_this.model.beat_tracker import BeatThis
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.utils import replace_state_dict_key
    require(Path(inference.__file__).resolve().is_relative_to(upstream), 'wrong imported upstream')
    torch.set_num_threads(lock['threads'])
    torch.set_num_interop_threads(lock['interop_threads'])
    lifecycle = hook_failure_control(torch)
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
    del checkpoint_bytes
    hparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in inspect.signature(BeatThis).parameters}
    model = BeatThis(**hparams).eval()
    model.load_state_dict(replace_state_dict_key(checkpoint['state_dict'], 'model.', ''))
    del checkpoint
    require(type(model.task_heads).__name__ == 'SumHead' and
            type(model.transformer_blocks.norm).__name__ == 'RMSNorm' and
            tuple(model.task_heads.beat_downbeat_lin.weight.shape) == (2, 512), 'unexpected tap architecture')
    initial_state = model_state_hash(model)
    postprocess = Postprocessor(type='minimal', fps=50)
    output.mkdir(mode=0o700)
    start = time.perf_counter()
    authored, actual = [], []
    for item in lock['authored_controls']:
        print('Authored capture: ' + item['id'], flush=True)
        authored.append(run_case(model, inference, postprocess, authored_mel(item['frames'], item['kind']),
                                 item['id'], lock, output))
    for trace, case in zip(traces, baseline['cases']):
        print('Reference capture: ' + case['case_id'], flush=True)
        mel = np.asarray(trace['mel_values'], np.float32).reshape(trace['mel_shape'])[0]
        actual.append(dict(run_case(model, inference, postprocess, mel, case['case_id'], lock, output,
                                    trace), source_trace_sha256=case['trace_sha256']))
    require(model_state_hash(model) == initial_state, 'capture mutated model state')
    report = dict(schema_version=1, purpose=lock['purpose'], complete=True,
        lock_sha256=sha(LOCK_PATH.read_bytes()), reference_lock_sha256=lock['reference_lock_sha256'],
        reference_report_sha256=lock['reference_report_sha256'],
        source_sha256={p: sha((HERE / p).read_bytes()) for p in ('prehead_capture.py', 'prehead_capture_audit.py')},
        reference_revision=reference['reference_revision'], checkpoint_sha256=reference['checkpoint']['sha256'],
        upstream_source_sha256={p: sha(git('show', reference['reference_revision'] + ':' + p)) for p in
            ('beat_this/inference.py', 'beat_this/model/beat_tracker.py', 'beat_this/model/roformer.py',
             'beat_this/model/postprocessor.py', 'beat_this/preprocessing.py', 'beat_this/utils.py')},
        versions=versions, threads=lock['threads'], interop_threads=lock['interop_threads'],
        budgets=lock['budgets'], hook_failure_control=lifecycle, model_state_unchanged=True,
        authored=authored, references=actual, gate_elapsed_s=time.perf_counter() - start,
        process_lifetime_peak_rss_bytes=capture.peak_rss_bytes(),
        decision='capture_fidelity_only_no_discrimination_or_promotion',
        production_changed=False, training=False, holdout_access=False, full_cohort_capture=False)
    # No completion report exists on failure; partial private files are not an accepted capture.
    with (output / 'report.json').open('x', encoding='utf-8', newline='\n') as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write('\n')
    print('PREHEAD_CAPTURE_FIDELITY_PASSED', flush=True)


if __name__ == '__main__':
    main()
