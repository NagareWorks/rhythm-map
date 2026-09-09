"""Model-free source/semantic audit and authored admission contract, not a detector."""
import argparse
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_SHA256 = 'd39d7f45dc4123847513abf618eba7c4151107560234621084109835981db14d'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_pinned(path, digest):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'changed bytes: ' + path.name)
    return json.loads(raw)


def protocol():
    source = read_pinned(HERE / 'rhythm-presence-source-v1.json', SOURCE_SHA256)
    reports = {name: read_pinned(HERE / name, digest)
               for name, digest in source['previous_reports'].items()}
    require(all(reports[name]['complete'] is True for name in
                ('musicfm-temporal-v1.json', 'musicfm-negative-v1.json')), 'incomplete prior report')
    require(reports['beat-this-semantics-v1.json']['adapter_accepted'] is False,
            'previous semantics disposition changed')
    return source, reports


def gate_cases(source, reports):
    """Truth stays in the evaluator manifest, never in a future producer input."""
    gate = source['gate']
    positives, negatives = gate['positive_ids'], gate['negative_ids']
    expected_ids = positives + negatives
    require(len(set(expected_ids)) == len(expected_ids), 'duplicate declared identity')
    cases = []
    for name in ('musicfm-temporal-v1.json', 'musicfm-negative-v1.json'):
        for row in reports[name]['cases']:
            require(row['id'] in expected_ids, 'unexpected retained case')
            cases.append(dict(id=row['id'], pcm_sha256=row['pcm_sha256'],
                              expected_status='supported' if row['id'] in positives else 'not_supported'))
    require(sorted(c['id'] for c in cases) == sorted(expected_ids), 'incomplete retained population')
    digests = [c['pcm_sha256'] for c in cases]
    require(len(set(digests)) == len(digests), 'duplicate PCM identity')
    require(all(len(d) == 64 and all(c in '0123456789abcdef' for c in d) for d in digests),
            'invalid PCM identity')
    return cases


def assess_admissibility(cases, predictions):
    """Check a complete ledger; this cannot certify a producer's data isolation."""
    expected = {row['pcm_sha256']: row['expected_status'] for row in cases}
    require(len(expected) == len(cases) and bool(cases), 'invalid case population')
    require(all(v in ('supported', 'not_supported') for v in expected.values()), 'invalid expectation')
    if predictions is None:
        return dict(status='not_run', passes=None, case_count=len(cases), unknown_count=None)
    require(type(predictions) is list, 'prediction ledger must be a list')
    by_hash = {}
    for row in predictions:
        require(type(row) is dict and set(row) == {'pcm_sha256', 'status'}, 'invalid prediction fields')
        digest, status = row['pcm_sha256'], row['status']
        require(type(digest) is str and digest in expected, 'unexpected PCM identity')
        require(digest not in by_hash, 'duplicate prediction')
        require(type(status) is str and status in ('supported', 'not_supported', 'unknown'),
                'invalid prediction status')
        by_hash[digest] = status
    require(set(by_hash) == set(expected), 'missing prediction')
    rows = [dict(pcm_sha256=digest, expected_status=expectation,
                 status=by_hash[digest], matches=by_hash[digest] == expectation)
            for digest, expectation in expected.items()]
    passes = all(row['matches'] for row in rows)
    return dict(status='authored_gate_pass_not_music_acceptance' if passes else 'authored_gate_failed',
                passes=passes, case_count=len(rows), unknown_count=sum(r['status'] == 'unknown' for r in rows),
                positives_supported=sum(r['expected_status'] == 'supported' and r['matches'] for r in rows),
                negatives_not_supported=sum(r['expected_status'] == 'not_supported' and r['matches'] for r in rows),
                cases=rows)


def probability_vector(values):
    require(len(values) > 0 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                                  for v in values), 'invalid probability vector')
    require(math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-12), 'probabilities do not sum to one')


def weighted_posterior(probabilities, weights, inverse=False):
    """Population weighted-CE identity only, not calibration of actual predictions."""
    probability_vector(probabilities)
    require(len(weights) == len(probabilities) and
            all(type(w) in (int, float) and math.isfinite(w) and w > 0 for w in weights), 'invalid weights')
    # Log arithmetic avoids overflow for otherwise valid large finite weights.
    logs = [(-math.inf if p == 0 else math.log(p) + (-1 if inverse else 1) * math.log(w))
            for p, w in zip(probabilities, weights)]
    maximum = max(logs)
    masses = [math.exp(v - maximum) for v in logs]
    total = math.fsum(masses)
    return [v / total for v in masses]


def histogram_information_bits(counts):
    """Generic Shannon concentration relative to uniform bins; no tracker port."""
    require(len(counts) > 1 and all(type(v) is int and v >= 0 for v in counts), 'invalid histogram')
    total = sum(counts)
    require(total > 0, 'empty histogram')
    entropy = -math.fsum((v / total) * math.log2(v / total) for v in counts if v)
    return math.log2(len(counts)) - entropy


def semantic_witnesses():
    regular, irregular = (10, 35, 60, 85), (10, 11, 12, 85)

    def frame_summary(ticks):
        # One accent, three plain beats, 96 off-tick frames, irrespective of order.
        labels = ['nonbeat'] * 100
        for index, tick in enumerate(ticks):
            labels[tick] = 'downbeat' if index == 0 else 'beat'
        return [labels.count(c) / len(labels) for c in ('beat', 'downbeat', 'nonbeat')]

    regular_summary, irregular_summary = frame_summary(regular), frame_summary(irregular)
    weighted = weighted_posterior(regular_summary, [60, 200, 1])
    # Report formatting only: libm's last bit can differ across Windows/Linux.
    # This precision never participates in the categorical admission decision.
    reported = lambda values: [round(v, 12) for v in values]
    return dict(
        frame_nonbeat=dict(regular_ticks=list(regular), irregular_ticks=list(irregular),
                           regular_summary=regular_summary, irregular_summary=irregular_summary,
                           same_summary=regular_summary == irregular_summary,
                           regular_offtick_fraction=regular_summary[2],
                           weighted_ce_population_posterior=reported(weighted),
                           recovered_frame_posterior=reported(weighted_posterior(weighted, [60, 200, 1], inverse=True)),
                           actual_backend_inference=False, region_presence_probability=None),
        agreement=dict(bins=40, concentrated_bits=round(histogram_information_bits([100] + [0] * 39), 12),
                       uniform_bits=round(histogram_information_bits([1] * 40), 12),
                       multiplied_counts_bits=round(histogram_information_bits([500] + [0] * 39), 12),
                       actual_tracker_inference=False, acoustic_input_used=False,
                       conclusion='agreement_of_identical_ticks_does_not_identify_their_acoustic_cause'),
        reported_precision_decimal_places=12,
        prior_only=dict(signal_log_likelihood_ratios=[0.0, 0.0], family_prior=[0.995, 0.005],
                        family_posterior=[0.995, 0.005], acoustic_presence_probability=None,
                        conclusion='equal_likelihoods_retain_prior_preference_not_signal_confidence'))


def build_report():
    source, reports = protocol()
    cases = gate_cases(source, reports)
    return dict(schema='rhythm-map.rhythm-presence-audit.v1', complete=True,
                source_sha256=SOURCE_SHA256,
                decision='no_direct_presence_producer_qualified_by_this_bounded_screen',
                candidate_dispositions=[dict(id=c['id'], disposition=c['disposition']) for c in source['candidates']],
                semantic_witnesses=semantic_witnesses(),
                prior_decisions={name: report.get('decision', report.get('purpose'))
                                 for name, report in reports.items()},
                gate_cases=cases, positive_count=sum(c['expected_status'] == 'supported' for c in cases),
                negative_count=sum(c['expected_status'] == 'not_supported' for c in cases),
                producer_gate=assess_admissibility(cases, None),
                new_model_acquisition=False, new_inference=False, music_access=False, holdout_access=False,
                training=False, training_necessity_proven=False, production_change=False, new_user_parameters=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path, help='fresh report path; never overwrites')
    group.add_argument('--check', type=Path, help='verify an existing report against the pinned sources')
    args = parser.parse_args()
    report = build_report()
    if args.check:
        require(json.loads(args.check.read_text(encoding='utf-8')) == report, 'retained report differs')
    else:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
