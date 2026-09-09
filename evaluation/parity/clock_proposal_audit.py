"""One fixed proposal-coverage audit on all forty exposed calibration inputs.

Generation of every fixed-grid query completes before annotations are loaded.
Later native-grid comparison is only a mechanical development proxy. No real
listener labels, packet clearance, classifier, model run or acceptance claim.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import annotation_exposure_audit as exposure
import clock_proposals as proposal

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def projection(payload):
    """Allowlist only. Dense capture identity is independently pinned by caller."""
    observations = payload['observations']
    proposal.require(payload['start_time_s'] == 0 and payload['frame_rate_hz'] == 50 and
                     observations['duration_s'] <= payload['frame_count'] / 50, 'unexpected capture support')
    return dict(duration_s=observations['duration_s'], beat_times_s=[b['time_s'] for b in observations['beats']],
                valid_spans_s=[[0, observations['duration_s']]])


def generate_capture(payload):
    observations = projection(payload)
    # Call boundary accepts neither IDs, scores nor annotation-selected windows.
    return [proposal.generate(observations, q) for q in proposal.query_grid(observations['duration_s'])]


def slice_name(data, query):
    """Labels are attached only after generation, including ramp/untyped rows."""
    if data['untyped']:
        return 'expressive_untyped'
    a, b = (t * 1000000 for t in query)
    segments = [s for s in data['segments'] if s['start'] < b and s['end'] > a]
    if any(s['kind'] == 'ramp' for s in segments):
        return 'ramp'
    if any(a <= c['at'] < b and c['kind'] == 'tempo_jump' for c in data['changes']):
        return 'step'
    if len(segments) == 1 and segments[0]['kind'] == 'constant' and segments[0]['start'] <= a < b <= segments[0]['end']:
        return 'constant'
    return 'other_or_boundary'


def summarize(rows):
    return dict(queries=len(rows), generator_status=dict(sorted(Counter(r['generated']['status'] for r in rows).items())),
                reference_status=dict(sorted(Counter(r['reference']['status'] for r in rows).items())),
                candidate_counts=dict(sorted(Counter(str(r['generated']['unique_count']) for r in rows).items())),
                full_readout_context_queries=sum(r['generated']['full_readout_context'] for r in rows),
                full_clock_context_queries=sum(r['generated'].get('full_clock_context', False) for r in rows),
                tick_budget_exceeded_candidates=sum(r['reference'].get('tick_budget_exceeded_candidates', 0) for r in rows))


def audit(roots):
    lock_path = HERE / 'clock-proposal-lock-v1.json'
    lock = json.loads(lock_path.read_bytes())
    proposal.require(all(sha(HERE / name) == value for name, value in lock['source_sha256'].items()), 'locked implementation changed')
    prior = exposure.read_json(HERE / exposure.SOURCE, exposure.SOURCE_HASH)
    metadata = exposure.metadata_projection(prior)  # No truth contents or old scores.
    proposal.require([c['cohort'] for c in metadata] == ['artbeat', 'rubato'] and
                     [len(c['cases']) for c in metadata] == [15, 25], 'calibration population changed')
    generated = []
    started = time.perf_counter()
    for cohort in metadata:
        for identity in cohort['cases']:
            path = Path(roots[cohort['cohort']]) / (identity['id'] + '.json')
            payload = exposure.read_json(path, identity['capture_sha256'])
            proposal.require(payload['case_id'] == identity['id'] and payload['frame_count'] == identity['frame_count'],
                             'capture identity differs')
            rows = generate_capture(payload)
            generated.append(dict(cohort=cohort['cohort'], identity=identity,
                                  observation_sha256=rows[0]['observation_sha256'], windows=rows))
    generation_s = time.perf_counter() - started
    generation_hash = proposal.canonical_hash(generated)
    # All candidates now fixed. No truth value was available to generate_capture.
    later_metadata, annotated = exposure.load_annotations()
    proposal.require(later_metadata == metadata, 'annotation identity projection changed')
    tracks, all_rows, private_rows = [], [], []
    for capture, case in zip(generated, annotated):
        proposal.require(capture['identity'] == case['identity'] and capture['cohort'] == case['cohort'], 'annotation association changed')
        reference_times = [t / 1000000 for t in case['data']['beats']]
        rows = [dict(generated=g, reference=proposal.reference_check(g, reference_times),
                     slice=slice_name(case['data'], g['query_s'])) for g in capture['windows']]
        private_rows.append(dict(cohort=capture['cohort'], identity=capture['identity'], rows=rows))
        tracks.append(dict(cohort=capture['cohort'], **capture['identity'],
                           observation_sha256=capture['observation_sha256'],
                           generated_sha256=proposal.canonical_hash(capture['windows']), **summarize(rows)))
        all_rows.extend(rows)
    proposal.require(proposal.canonical_hash(generated) == generation_hash, 'post-hoc scoring changed generated candidates')
    private = dict(generation_sha256=generation_hash, tracks=private_rows)
    report = dict(schema='rhythm-map.clock-proposal-audit.v1', contract=lock,
        lock_sha256=sha(lock_path), source_report_sha256=exposure.SOURCE_HASH,
        generation_sha256=generation_hash, private_content_sha256=proposal.canonical_hash(private),
        generation_elapsed_s=generation_s, capture_count=len(generated), input_cases=tracks,
        summary=summarize(all_rows), slices={s: summarize([r for r in all_rows if r['slice'] == s])
                                            for s in sorted({r['slice'] for r in all_rows})},
        generation_completed_before_annotation_load=True, original_forty_retained=True,
        new_annotations_collected=0, independently_labeled_coverage=None,
        neural_inference=False, training=False, holdout_access=False, production_change=False,
        selected_clock=None, decision='development_proxy_only_not_training_ready')
    return report, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artbeat-captures', 'rubato-captures', 'output', 'private-output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    proposal.require(args.output.resolve() != args.private_output.resolve() and
                     not args.output.exists() and not args.private_output.exists(), 'output exists or paths alias')
    report, private = audit(dict(artbeat=args.artbeat_captures, rubato=args.rubato_captures))
    for path, value in ((args.private_output, private), (args.output, report)):
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
