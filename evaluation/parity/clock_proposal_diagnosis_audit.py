"""Replay immutable proposal geometry and diagnose every old query, not tune it."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import clock_proposal_audit as prior
import clock_proposal_diagnosis as diagnosis

HERE = Path(__file__).resolve().parent
require = diagnosis.clocks.require
digest = diagnosis.clocks.canonical_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pinned(path, expected):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'pinned input bytes changed')
    return json.loads(data)


def build_report(private_path, roots):
    started = time.perf_counter()
    lock_path = HERE / 'clock-proposal-diagnosis-lock-v1.json'
    lock = json.loads(lock_path.read_bytes())
    require(all(sha(HERE / name) == value for name, value in lock['source_sha256'].items()), 'diagnostic code identity changed')
    old = read_pinned(HERE / 'clock-proposal-v1.json', lock['prior_report_sha256'])
    old_lock = read_pinned(HERE / 'clock-proposal-lock-v1.json', old['lock_sha256'])
    require(old_lock == old['contract'] and all(sha(HERE / n) == h for n, h in old_lock['source_sha256'].items()),
            'original generator or dependencies changed')
    private = read_pinned(private_path, lock['private_input_sha256'])
    require(digest(private) == old['private_content_sha256'] and
            private['generation_sha256'] == old['generation_sha256'], 'old candidate archive identity changed')
    _, annotations = prior.exposure.load_annotations()
    require(len(annotations) == len(old['input_cases']) == len(private['tracks']) == 40, 'input population changed')
    all_rows, tracks, private_rows = [], [], []
    for annotated, track, previous in zip(annotations, private['tracks'], old['input_cases']):
        identity, cohort = track['identity'], track['cohort']
        require(annotated['identity'] == identity and annotated['cohort'] == cohort == previous['cohort'] and
                all(previous[k] == v for k, v in identity.items()), 'track identity or ordering changed')
        payload = read_pinned(Path(roots[cohort]) / (identity['id'] + '.json'), identity['capture_sha256'])
        observations = prior.projection(payload)
        require(digest(observations) == previous['observation_sha256'], 'observation projection changed')
        generated = [row['generated'] for row in track['rows']]
        require(digest(generated) == previous['generated_sha256'] and
                [g['query_s'] for g in generated] == list(diagnosis.clocks.query_grid(observations['duration_s'])), 'query or candidate geometry changed')
        require(prior.summarize(track['rows']) == {k: previous[k] for k in prior.summarize(track['rows'])}, 'old per-track counts changed')
        reference = [t / 1000000 for t in annotated['data']['beats']]
        raw = observations['beat_times_s']
        candidates = [r['time_s'] for r in payload['observations']['beat_candidates']]
        rows = []
        for old_row in track['rows']:
            g = old_row['generated']
            require(prior.slice_name(annotated['data'], g['query_s']) == old_row['slice'], 'original slice changed')
            result = diagnosis.diagnose(g, reference, raw, candidates)
            require(result['original_reference'] == old_row['reference'], 'original query score changed')
            rows.append(dict(query_s=g['query_s'], slice=old_row['slice'], generator_status=g['status'], diagnosis=result))
        require(digest(generated) == previous['generated_sha256'], 'diagnosis rewrote a candidate')
        tracks.append(dict(cohort=cohort, **identity, **diagnosis.summarize(rows)))
        private_rows.append(dict(cohort=cohort, identity=identity, rows=rows))
        all_rows.extend(rows)
    require(digest(private) == old['private_content_sha256'], 'diagnosis mutated original archive')
    summary = diagnosis.summarize(all_rows)
    require(summary['queries'] == old['summary']['queries'] and summary['original_reference_status'] == old['summary']['reference_status'],
            'original overall denominator or scores changed')
    detailed = dict(original_generation_sha256=old['generation_sha256'], tracks=private_rows)
    report = dict(schema='rhythm-map.clock-proposal-diagnosis.v1', contract=lock, lock_sha256=sha(lock_path),
        prior_generation_sha256=old['generation_sha256'], private_content_sha256=digest(detailed),
        elapsed_s=time.perf_counter() - started, input_cases=tracks, summary=summary,
        slices={s: diagnosis.summarize([r for r in all_rows if r['slice'] == s]) for s in sorted({r['slice'] for r in all_rows})},
        all_original_candidates_and_scores_unchanged=True, posthoc_reference_assisted_diagnosis=True,
        new_candidates_generated=False, phase_shift_applied=False, selected_clock=None,
        new_annotations_collected=0, neural_inference=False, training=False, holdout_access=False,
        production_change=False, accuracy_improvement_claimed=False,
        decision='descriptive_decomposition_only_not_a_repaired_generator_or_training_gate')
    return report, detailed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prior-private', 'artbeat-captures', 'rubato-captures', 'output', 'private-output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve() != args.private_output.resolve() and not args.output.exists() and not args.private_output.exists(),
            'output already exists or aliases')
    report, detailed = build_report(args.prior_private, dict(artbeat=args.artbeat_captures, rubato=args.rubato_captures))
    for path, value in ((args.private_output, detailed), (args.output, report)):
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
