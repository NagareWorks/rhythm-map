"""Run the fixed audit; previews/nominal labels are deliberately not fit inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import wave

import numpy as np

from . import protocol as p
from .render import duration_metrics, event_times, pitch_metrics, render, timing_metrics, witness, write_wav
from .timeline import Timeline


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def float_sha(pcm):
    return hashlib.sha256(np.asarray(pcm, dtype='<f4').tobytes()).hexdigest()


def read_source(root, item):
    path = (root / item['audio_hint']).resolve()
    if not path.is_relative_to(root.resolve()) or p.sha(path) != item['audio_sha256']:
        raise ValueError('audio identity or containment changed')
    with wave.open(str(path), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, p.SR, 'NONE'):
            raise ValueError('unexpected source PCM contract')
        samples = np.frombuffer(wav.readframes(60*p.SR), dtype='<i2').astype(np.float32) / 32768
    if len(samples) != 60*p.SR:
        raise ValueError('source prefix too short')
    return samples


def run(args):
    started = time.monotonic()
    sources, closure = p.selected_sources(), p.source_closure()
    if p.sha(args.ffmpeg) != args.ffmpeg_sha256:
        raise ValueError('FFmpeg identity changed')
    version = subprocess.run([str(args.ffmpeg), '-version'], check=True, capture_output=True, text=True, timeout=10).stdout
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rhythm-map.tempo-pair-renderer-audit.v1', source_closure=closure,
                  sources=sources, profiles=p.PROFILES, source_edges_s=p.SOURCE_EDGES, limits=p.LIMITS,
                  ffmpeg_sha256=args.ffmpeg_sha256, ffmpeg_version=version,
                  python=platform.python_version(), numpy=np.__version__,
                  independent_acceptance=False, holdout_access=False, model_calls=0, optimizer_steps=0,
                  training_eligible=False, music_alignment_verified=False, authored=[], previews=[])
    write_json(args.output / 'plan.json', report)

    def check_budget():
        if time.monotonic() - started > 900:
            raise TimeoutError('fixed audit wall budget exceeded')

    try:
        for backend in p.BACKENDS:
            for profile, rates in p.PROFILES.items():
                timeline = Timeline(p.SOURCE_EDGES, rates)
                for kind in p.WITNESSES:
                    check_budget()
                    pcm, events = witness(kind)
                    output, sizes = render(pcm, rates, backend, args.ffmpeg)
                    row = dict(backend=backend, profile=profile, witness=kind,
                               input_float_sha256=float_sha(pcm), output_float_sha256=float_sha(output),
                               duration=duration_metrics(sizes, rates))
                    if kind == 'tone':
                        row['pitch'] = pitch_metrics(output, sizes)
                    else:
                        observed = event_times(output)
                        row['timing'] = timing_metrics(timeline.to_output(events), observed)
                        row['observed_events_s'] = observed.tolist()
                    row['passed'] = all(v['passed'] for k, v in row.items() if k in ('duration', 'pitch', 'timing'))
                    report['authored'].append(row)
        report['authored_gate_by_backend'] = {
            b: all(r['passed'] for r in report['authored'] if r['backend'] == b) for b in p.BACKENDS}
        # Diagnostic preview generation does not confer label eligibility.
        for item in sources:
            check_budget()
            pcm = read_source(args.audio_root, item)
            truth = json.loads((p.ROOT / item['truth']).read_bytes())
            beats = [b for b in truth['beats'] if 0 <= b['time_s'] < 60]
            write_wav(args.output / (item['id'] + '.source.wav'), pcm)
            for backend in p.BACKENDS:
                for profile, rates in p.PROFILES.items():
                    check_budget()
                    output, sizes = render(pcm, rates, backend, args.ffmpeg)
                    timeline = Timeline(p.SOURCE_EDGES, rates)
                    name = f"{item['id']}.{backend}.{profile}"
                    audio_path = args.output / (name + '.wav')
                    write_wav(audio_path, output)
                    mapped = [dict(b, time_s=float(t)) for b, t in
                              zip(beats, timeline.to_output([b['time_s'] for b in beats]))]
                    label = dict(source_id=item['id'], source_truth_sha256=item['truth_sha256'],
                                 nominal_only=True, training_eligible=False,
                                 duration_s=float(timeline.output_edges[-1]), beats=mapped,
                                 source_edges_s=p.SOURCE_EDGES, rates=rates,
                                 nominal_output_edges_s=timeline.output_edges.tolist(),
                                 actual_output_edges_s=(np.r_[0, np.cumsum(sizes)]/p.SR).tolist(),
                                 provenance=item['provenance'], modification='prefix crop and piecewise tempo transform')
                    label_path = args.output / (name + '.nominal.json')
                    write_json(label_path, label)
                    report['previews'].append(dict(source_id=item['id'], backend=backend, profile=profile,
                        training_eligible=False, duration=duration_metrics(sizes, rates),
                        input_float_sha256=float_sha(pcm), output_float_sha256=float_sha(output),
                        source_preview_sha256=p.sha(args.output / (item['id'] + '.source.wav')),
                        wav_sha256=p.sha(audio_path), nominal_labels_sha256=p.sha(label_path),
                        peak_abs=float(np.max(np.abs(output)))))
        if p.source_closure() != closure or p.sha(args.ffmpeg) != args.ffmpeg_sha256:
            raise ValueError('source or renderer mutated during audit')
        report['completed'] = True
    except Exception as error:
        report['completed'] = False
        report['failure_type'] = type(error).__name__
        write_json(args.output / 'partial-report.json', report)
        raise
    report['elapsed_s'] = time.monotonic() - started
    write_json(args.output / 'report.json', report)
    print(json.dumps(dict(completed=True, authored_gate=report['authored_gate_by_backend'],
                         previews=len(report['previews']), training_eligible=False, elapsed_s=report['elapsed_s'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-root', required=True, type=Path)
    parser.add_argument('--ffmpeg', required=True, type=Path)
    parser.add_argument('--ffmpeg-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())
