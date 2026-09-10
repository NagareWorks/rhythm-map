"""Prepare the fixed private learning inputs through the existing Rust frontend."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

import data


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trace-executable', 'model-dir', 'rubato-audio', 'artbeat-audio', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reuse-traces', type=Path, action='append', default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    data.require(not output.exists() and not output.is_relative_to(data.ROOT), 'new private output outside Git required')
    if os.name == 'nt':
        data.require(output.drive.lower() != os.environ.get('SystemDrive', 'C:').lower(), 'keep artifacts off system drive')
    recipe = data.plan()
    # Persist the exact split/target/source contract before extracting any case.
    recipe['source_sha256'] = {p: data.sha(Path(__file__).parent / p) for p in ('data.py', 'model.py', 'prepare.py', 'README.md')}
    recipe['trace_executable_sha256'] = data.sha(args.trace_executable)
    output.mkdir()
    write_json(output / 'plan.json', recipe)
    started = time.monotonic()
    rows = []
    for index, item in enumerate(recipe['cases']):
        data.require(time.monotonic() - started < 3600, 'preparation wall-time budget exhausted; no complete manifest')
        audio_root = args.rubato_audio if item['cohort'] == 'rubato' else args.artbeat_audio
        audio_path = audio_root / item['audio_hint']
        data.require(data.sha(audio_path) == item['audio_sha256'], 'audio bytes changed')
        truth_path = data.ROOT / item['truth']
        data.require(data.sha(truth_path) == item['truth_sha256'], 'truth bytes changed')
        truth = json.loads(truth_path.read_bytes())
        trace, reused = None, False
        for root in args.reuse_traces:
            path = root / (item['id'] + '.trace.private.json')
            if not path.is_file():
                continue
            candidate = json.loads(path.read_bytes())
            # Only identical physical input and source/model contracts qualify.
            if (candidate['audio_sha256'] == item['audio_sha256'] and candidate['suite_sha256'] == item['suite_sha256']
                    and candidate['sample_rate'] == 22050 and candidate['observation_contract'] == 'beat-this-rten-observations-v2+decode-audio-v2'
                    and len(candidate['mono_samples']) == min(candidate['decoded_sample_count'], 60 * 22050)):
                trace, trace_path, reused = candidate, path, True
                break
        print(f"Prepare {index + 1}/40 {item['id']} ({item['role']}); cached trace={reused}", flush=True)
        if trace is None:
            trace_path = output / (item['id'] + '.trace.private.json')
            subprocess.run([str(args.trace_executable), '--suite', str(data.ROOT / 'evaluation/suites' / (item['suite'] + '.json')),
                '--case', item['id'], '--audio-dir', str(audio_root), '--model-pack', str(data.ROOT / 'models/beat-this-full-v1.json'),
                '--model-dir', str(args.model_dir), '--seconds', '60', '--output', str(trace_path)],
                check=True, cwd=data.ROOT, env=dict(os.environ, RTEN_NUM_THREADS='2'), timeout=240)
            trace = json.loads(trace_path.read_bytes())
        data.require(trace['case_id'] == item['id'] and trace['audio_sha256'] == item['audio_sha256']
                     and trace['suite_sha256'] == item['suite_sha256'] and trace['sample_rate'] == 22050
                     and trace['model_manifest_sha256'] == 'ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d', 'trace identity changed')
        mel = np.asarray(trace['mel_values'], np.float32).reshape(trace['mel_shape'])[0]
        duration = len(trace['mono_samples']) / 22050
        data.require(mel.ndim == 2 and mel.shape[1] == 128 and 1 <= len(mel) <= 3001 and np.isfinite(mel).all(), 'invalid mel')
        y, valid = data.targets([b['time_s'] for b in truth['beats']], len(mel), duration)
        raw = [b['time_s'] for b in trace['observations']['beats']]
        if len(raw) >= 2:
            baseline, baseline_valid = data.targets(raw, len(mel), duration)
        else:
            baseline, baseline_valid = np.zeros_like(y), np.zeros(len(y), bool)
        arrays = dict(mel=mel, target=y, valid=valid, baseline=baseline, baseline_valid=baseline_valid,
                      beat=np.asarray(trace['beat_logits'], np.float32), downbeat=np.asarray(trace['downbeat_logits'], np.float32))
        archive_path = output / (item['id'] + '.input.npz')
        with archive_path.open('xb') as stream:
            np.savez(stream, **arrays)
        rows.append(dict(item, frames=len(mel), duration_s=duration, original_duration_s=trace['decoded_sample_count'] / 22050,
            input_relation='crop' if len(trace['mono_samples']) < trace['decoded_sample_count'] else 'complete',
            valid_reference_frames=int(valid.sum()), input_sha256=data.sha(archive_path), trace_sha256=data.sha(trace_path), reused_trace=reused))
    write_json(output / 'inputs.json', dict(plan_sha256=data.sha(output / 'plan.json'), cases=rows,
        preparation_elapsed_s=time.monotonic() - started, complete=True, training=False, holdout_access=False))
    print('ALL_40_INPUTS_PREPARED', flush=True)


if __name__ == '__main__':
    main()
