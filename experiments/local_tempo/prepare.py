"""Render and check new schedule PCM without a neural forward or optimizer."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.tempo_pairs import protocol as p
from experiments.tempo_pairs.render import render, witness, write_wav, duration_metrics, event_times, timing_metrics, pitch_metrics
from experiments.tempo_pairs.run import float_sha
from experiments.tempo_pairs.timeline import Timeline
from experiments.tempo_alignment import alignment as a
from experiments.tempo_alignment.run import load_pcm, changed_context, negative_summary
from . import data
from .data import require, sha, write_json


def prepare(args):
    started = time.monotonic()
    closure = data.closure()
    sources = p.selected_sources()
    require(sha(args.ffmpeg) == data.FFMPEG_SHA, 'renderer binary changed')
    require(not args.output.exists() and not args.output.resolve().is_relative_to(p.ROOT), 'fresh private output required')
    args.output.mkdir(parents=True)
    report = dict(schema='rhythm-map.local-tempo-preparation.v1', source_sha256=closure,
                  ffmpeg_sha256=data.FFMPEG_SHA, sources=sources, profiles=data.FRESH,
                  authored=[], pairs=[], model_calls=0, optimizer_steps=0, holdout_access=False,
                  independent_acceptance=False, production_change=False)
    write_json(args.output/'plan.json', report)

    def check():
        require(time.monotonic()-started < 890, 'preparation budget exceeded')

    try:
        for profile, rates in data.FRESH.items():
            timeline = Timeline(p.SOURCE_EDGES, rates)
            for kind in p.WITNESSES:
                check()
                pcm, events = witness(kind)
                output, sizes = render(pcm, rates, 'rubberband', args.ffmpeg)
                metrics = dict(duration=duration_metrics(sizes, rates))
                if kind == 'tone':
                    metrics['pitch'] = pitch_metrics(output, sizes)
                else:
                    metrics['timing'] = timing_metrics(timeline.to_output(events), event_times(output))
                report['authored'].append(dict(profile=profile, witness=kind, metrics=metrics,
                    output_float_sha256=float_sha(output), passed=all(v['passed'] for v in metrics.values())))
        require(all(r['passed'] for r in report['authored']), 'new-schedule authored renderer gate failed')
        old = json.loads((p.ROOT/'experiments/tempo_pairs/results-v1.json').read_bytes())
        for source in sources:
            identity = source['id']
            prior = next(r for r in old['previews'] if r['source_id'] == identity and r['backend'] == 'rubberband')
            pcm = load_pcm(args.pairs/(identity+'.source.wav'), prior['source_preview_sha256']).astype(np.float32)
            source_views = {n: a.extract(pcm, n) for n in a.RESOLUTIONS}
            for profile, rates in data.FRESH.items():
                check()
                timeline = Timeline(p.SOURCE_EDGES, rates)
                output, sizes = render(pcm, rates, 'rubberband', args.ffmpeg)
                require(sizes == [round(20*p.SR/r) for r in rates], 'new music segment extent drift')
                path = args.output/(identity+'.'+profile+'.wav')
                write_wav(path, output)
                digest = sha(path)
                actual = load_pcm(path, digest)
                views = {n: a.extract(actual, n) for n in a.RESOLUTIONS}
                delay = round(a.DELAY_S*p.SR)
                delayed_pcm = np.r_[np.zeros(delay), actual[:-delay]]
                delayed_views = {n: a.extract(delayed_pcm, n) for n in a.RESOLUTIONS}
                points = []
                for index, center in enumerate(np.arange(.5, timeline.output_edges[-1], 1.)):
                    check()
                    natural = a.measure(source_views, views, timeline, center)
                    delayed = a.measure(source_views, delayed_views, timeline, center)
                    row = data.fresh_point(identity, profile, index, natural, delayed,
                        a.paired_support(natural, delayed), a.region(center, timeline.output_edges[-1], timeline.output_edges[1:-1]))
                    row.update(changed_context=changed_context(timeline, center),
                               wrong_map=a.measure(source_views, source_views, timeline, center))
                    points.append(row)
                negative = negative_summary(points)
                changed = sum(r['admitted'] and r['rate'] != 1 for r in points)
                report['pairs'].append(dict(source_id=identity, profile=profile, source_wav_sha256=prior['source_preview_sha256'],
                    wav_sha256=digest, duration=duration_metrics(sizes, rates), points=points,
                    admitted_points=sum(r['admitted'] for r in points), admitted_changed_points=changed,
                    negative_control=negative, passed=negative['passed'] and changed >= 10))
                print(json.dumps(dict(event='prepared', source_id=identity, profile=profile,
                                     admitted_changed_points=changed, passed=report['pairs'][-1]['passed'])), flush=True)
        require(sum(len(p['points']) for p in report['pairs']) == 366, 'fresh grid denominator changed')
        require(closure == data.closure(), 'preparation source changed')
        report.update(completed=True, gate_passed=all(p['passed'] for p in report['pairs']), elapsed_s=time.monotonic()-started)
        write_json(args.output/'report.json', report)
        print(json.dumps(dict(completed=True, gate_passed=report['gate_passed'], elapsed_s=report['elapsed_s'])), flush=True)
    except BaseException as error:
        report.update(completed=False, failure_type=type(error).__name__, elapsed_s=time.monotonic()-started)
        write_json(args.output/'partial-report.json', report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pairs', 'ffmpeg', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    prepare(parser.parse_args())
