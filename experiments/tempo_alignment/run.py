"""Verify retained PCM identities, then measure all declared musical points."""
import argparse
import json
from pathlib import Path
import platform
import time
import wave

import numpy as np

from experiments.tempo_pairs import protocol as old
from experiments.tempo_pairs.timeline import Timeline
from .alignment import DELAY_S, LIMITS, RESOLUTIONS, SR, extract, measure, paired_support, region, summarize

OLD_REPORT = 'experiments/tempo_pairs/results-v1.json'
OLD_SHA = 'b894c92a7c586b6f6d043b716b7e16a60a4cde0a82ae2eeb8fd42cc5b4f4a0cc'


def closure():
    names = ('__init__.py', 'alignment.py', 'run.py', 'PROTOCOL-v1.md')
    result = {f'experiments/tempo_alignment/{n}': old.sha(old.ROOT/'experiments/tempo_alignment'/n) for n in names}
    result[OLD_REPORT] = old.sha(old.ROOT/OLD_REPORT)
    return result


def load_pcm(path, expected_sha):
    if old.sha(path) != expected_sha:
        raise ValueError('retained WAV identity changed')
    with wave.open(str(path)) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, SR, 'NONE'):
            raise ValueError('unexpected PCM contract')
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(float)/32768


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def changed_context(timeline, center):
    return any(rate != 1 and center - 1 >= a and center + 1 <= b
               for a, b, rate in zip(timeline.output_edges[:-1], timeline.output_edges[1:], timeline.rates))


def negative_summary(rows):
    changed = [r['wrong_map'] for r in rows if r.get('changed_context')]
    physical = [r for r in changed if r['status'] != 'unsupported']
    false_aligned = sum(r['status'] == 'aligned' for r in physical)
    return dict(changed_context_points=len(changed), physically_supported_points=len(physical),
                false_aligned_points=false_aligned,
                false_aligned_fraction=false_aligned/len(physical) if physical else None,
                passed=len(physical) >= 10 and false_aligned/len(physical) <= .05)


def interior_gate(grid, beats):
    return all(s['interior']['supported_fraction'] is not None
               and s['interior']['supported_fraction'] >= LIMITS['min_interior_supported_fraction']
               for s in (grid, beats))


def run(args):
    begin = time.monotonic()
    if old.sha(old.ROOT/OLD_REPORT) != OLD_SHA:
        raise ValueError('old renderer outcome changed')
    prior = json.loads((old.ROOT/OLD_REPORT).read_bytes())
    if prior['source_closure'] != old.source_closure() or prior['sources'] != old.selected_sources():
        raise ValueError('old code, roles or provenance changed')
    selected = [r for r in prior['previews'] if r['backend'] == 'rubberband']
    if len(selected) != 15:
        raise ValueError('population changed')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='rhythm-map.tempo-alignment.v1', source_closure=closure(),
                  previous_report_sha256=OLD_SHA, limits=LIMITS, delay_samples=round(DELAY_S*SR),
                  independent_acceptance=False, training_eligible=False, holdout_access=False,
                  model_calls=0, optimizer_steps=0, python=platform.python_version(), numpy=np.__version__,
                  inputs=selected, cases=[])
    write_json(args.output/'plan.json', report)

    def budget():
        if time.monotonic() - begin >= 900:
            raise TimeoutError('fixed alignment budget exceeded')

    try:
        for identity in old.IDS:
            item = next(r for r in selected if r['source_id'] == identity)
            source = load_pcm(args.pairs/(identity+'.source.wav'), item['source_preview_sha256'])
            source_views = {n: extract(source, n) for n in RESOLUTIONS}
            for item in (r for r in selected if r['source_id'] == identity):
                budget()
                stem = f"{identity}.rubberband.{item['profile']}"
                pcm = load_pcm(args.pairs/(stem+'.wav'), item['wav_sha256'])
                label_path = args.pairs/(stem+'.nominal.json')
                if old.sha(label_path) != item['nominal_labels_sha256']:
                    raise ValueError('nominal label identity changed')
                labels = json.loads(label_path.read_bytes())
                timeline = Timeline(old.SOURCE_EDGES, old.PROFILES[item['profile']])
                if (labels['rates'] != list(timeline.rates) or labels['training_eligible'] is not False
                        or labels['nominal_only'] is not True or len(pcm) != sum(item['duration']['actual_segment_samples'])):
                    raise ValueError('nominal label or actual extent contract changed')
                delay = round(DELAY_S*SR)
                shifted = np.r_[np.zeros(delay), pcm[:-delay]]
                views = {n: extract(pcm, n) for n in RESOLUTIONS}
                control = {n: extract(shifted, n) for n in RESOLUTIONS}
                duration = float(timeline.output_edges[-1])
                rows = []
                for kind, centers in (('grid', np.arange(.5, duration, 1.)),
                                      ('beat', [b['time_s'] for b in labels['beats']])):
                    for index, center in enumerate(centers):
                        budget()
                        natural = measure(source_views, views, timeline, center)
                        delayed = measure(source_views, control, timeline, center)
                        row = dict(kind=kind, index=index, region=region(center, duration, timeline.output_edges[1:-1]),
                                   natural=natural, delayed=delayed, feature_capture_supported=paired_support(natural, delayed))
                        if kind == 'grid' and item['profile'] != 'identity_seams':
                            row['changed_context'] = changed_context(timeline, center)
                            row['wrong_map'] = measure(source_views, source_views, timeline, center)
                        rows.append(row)
                grid_rows = [r for r in rows if r['kind'] == 'grid']
                grid = summarize(grid_rows)
                beats = summarize([r for r in rows if r['kind'] == 'beat'])
                case = dict(source_id=identity, profile=item['profile'], grid=grid, beats=beats, points=rows,
                            interior_gate_passed=interior_gate(grid, beats), training_eligible=False)
                if item['profile'] != 'identity_seams':
                    case['negative_control'] = negative_summary(grid_rows)
                report['cases'].append(case)
                print(json.dumps(dict(source_id=identity, profile=item['profile'],
                      grid_supported=grid['interior']['supported_fraction'],
                      beats_supported=beats['interior']['supported_fraction'], elapsed_s=time.monotonic()-begin)), flush=True)
        budget()
        if report['source_closure'] != closure():
            raise ValueError('audit source changed during execution')
        report['negative_gate_passed'] = all(c['negative_control']['passed'] for c in report['cases'] if 'negative_control' in c)
        report['interior_gate_passed'] = all(c['interior_gate_passed'] for c in report['cases'])
        report['gate_passed'] = report['negative_gate_passed'] and report['interior_gate_passed']
        report['completed'] = True
    except Exception as error:
        report['completed'] = False
        report['failure_type'] = type(error).__name__
        write_json(args.output/'partial-report.json', report)
        raise
    report['elapsed_s'] = time.monotonic()-begin
    write_json(args.output/'report.json', report)
    print(json.dumps(dict(completed=True, gate_passed=report['gate_passed'], elapsed_s=report['elapsed_s'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())
