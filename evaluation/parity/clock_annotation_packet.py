"""Blinded two-pass packet contract; no UI, listening, clearance or training.

The private ledger binds provenance to the view. Do not distribute it to raters.
Hashes establish content identity, not trustworthy people, rights or chronology.
"""
import copy
import re

import clock_proposals as clocks
from clock_readout_proposal import consensus

require = clocks.require
digest = clocks.canonical_hash


def hash_string(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def identifier(value):
    return type(value) is str and bool(value) and value == value.strip()


def input_binding(value):
    require(type(value) is dict and set(value) == {'pcm_sha256', 'sample_rate', 'sample_count'}, 'invalid input binding')
    require(hash_string(value['pcm_sha256']) and all(type(value[k]) is int and value[k] > 0
            for k in ('sample_rate', 'sample_count')), 'invalid PCM identity')


def provenance(value):
    fields = {'work_id', 'recording_id', 'performance_id', 'creator_session_id',
              'shared_asset_ids', 'transform_family_ids', 'rights_evidence_sha256',
              'training_review', 'redistribution_review', 'encoder_overlap', 'input_relation'}
    require(type(value) is dict and set(value) == fields, 'invalid provenance fields')
    require(all(identifier(value[k]) for k in ('work_id', 'recording_id', 'performance_id', 'creator_session_id')),
            'missing provenance identity; unknown identities need explicit unresolved records')
    for key in ('shared_asset_ids', 'transform_family_ids'):
        ids = value[key]
        require(type(ids) is list and all(identifier(v) for v in ids) and len(set(ids)) == len(ids), 'invalid provenance links')
    require(hash_string(value['rights_evidence_sha256']), 'missing rights review evidence identity')
    require(all(value[k] in ('unreviewed', 'accepted', 'rejected') for k in ('training_review', 'redistribution_review')),
            'invalid separate rights review statuses')
    require(value['encoder_overlap'] in ('unknown', 'known_exposed', 'reviewed_no_known_overlap'), 'invalid encoder overlap')
    relation = value['input_relation']
    require(type(relation) is dict and set(relation) == {'kind', 'source_interval_s'}, 'invalid input relation')
    require(relation['kind'] in ('complete', 'crop', 'unknown'), 'invalid input relation kind')
    span = relation['source_interval_s']
    require((relation['kind'] == 'unknown' and span is None) or
            (relation['kind'] != 'unknown' and type(span) is list and len(span) == 2 and
             all(clocks.number(v) for v in span) and 0 <= span[0] < span[1]), 'invalid source interval')


def build(binding, private_provenance, observations, query, nonce):
    """Generate internally: a caller cannot pass oracle candidates or detector scores."""
    input_binding(binding)
    provenance(private_provenance)
    require(hash_string(nonce), 'a fixed external randomization nonce is required')
    require(abs(binding['sample_count'] / binding['sample_rate'] - observations['duration_s']) <= 1e-9,
            'PCM duration does not match declared analysis input')
    relation = private_provenance['input_relation']
    if relation['kind'] != 'unknown':
        a, b = relation['source_interval_s']
        require(abs(b - a - observations['duration_s']) <= 1e-9 and
                (relation['kind'] != 'complete' or a == 0), 'source crop relation does not match input')
    generated = clocks.generate(observations, query)
    first = dict(schema='rhythm-map.clock-first-view.v1', input=copy.deepcopy(binding), query_s=list(query))
    # Hash ordering is a fixed pseudorandom permutation, never a model ranking.
    ordered = sorted(generated['clocks'], key=lambda c: digest([nonce, c]))
    second = dict(schema='rhythm-map.clock-candidate-view.v1', first_view_sha256=digest(first),
                  input=copy.deepcopy(binding), query_s=list(query),
                  candidates=[dict(id=f'c{i:02d}', **copy.deepcopy(c)) for i, c in enumerate(ordered)])
    private = dict(provenance=copy.deepcopy(private_provenance), nonce=nonce,
                   observation_sha256=generated['observation_sha256'], generator_status=generated['status'],
                   generated_sha256=digest(generated), first_view_sha256=digest(first), second_view_sha256=digest(second))
    private['ledger_sha256'] = digest(private)
    return dict(first_view=first, candidate_view=second, private_ledger=private)


def validate_bundle(bundle):
    require(type(bundle) is dict and set(bundle) == {'first_view', 'candidate_view', 'private_ledger'}, 'invalid bundle fields')
    first, second, ledger = (bundle[k] for k in ('first_view', 'candidate_view', 'private_ledger'))
    require(set(first) == {'schema', 'input', 'query_s'} and first['schema'] == 'rhythm-map.clock-first-view.v1', 'invalid first view')
    input_binding(first['input'])
    query = first['query_s']
    require(type(query) is list and len(query) == 2 and all(clocks.number(v) for v in query) and
            0 <= query[0] < query[1] <= first['input']['sample_count'] / first['input']['sample_rate'], 'invalid packet query')
    require(set(second) == {'schema', 'first_view_sha256', 'input', 'query_s', 'candidates'} and
            second['schema'] == 'rhythm-map.clock-candidate-view.v1' and second['input'] == first['input'] and
            second['query_s'] == query and second['first_view_sha256'] == digest(first), 'candidate view context changed')
    candidates = second['candidates']
    require(type(candidates) is list and len(candidates) <= clocks.CAP, 'invalid packet population')
    for i, candidate in enumerate(candidates):
        require(type(candidate) is dict and set(candidate) == {'id', 'knots'} and candidate['id'] == f'c{i:02d}', 'nonopaque or invalid candidate')
        clock = {'knots': candidate['knots']}
        clocks.validate_clock(clock)
        require(clock['knots'][0][0] <= query[0] < query[1] <= clock['knots'][-1][0], 'candidate misses query support')
        require(not any(clocks.equivalent(clock, {'knots': old['knots']}) for old in candidates[:i]), 'duplicate physical clock')
    require(set(ledger) == {'provenance', 'nonce', 'observation_sha256', 'generator_status',
                           'generated_sha256', 'first_view_sha256', 'second_view_sha256', 'ledger_sha256'}, 'invalid ledger fields')
    provenance(ledger['provenance'])
    relation = ledger['provenance']['input_relation']
    if relation['kind'] != 'unknown':
        a, b = relation['source_interval_s']
        require(abs(b - a - first['input']['sample_count'] / first['input']['sample_rate']) <= 1e-9 and
                (relation['kind'] != 'complete' or a == 0), 'source crop relation does not match input')
    require(all(hash_string(ledger[k]) for k in ('nonce', 'observation_sha256', 'generated_sha256',
                    'first_view_sha256', 'second_view_sha256', 'ledger_sha256')), 'invalid ledger hash')
    require(ledger['first_view_sha256'] == digest(first) and ledger['second_view_sha256'] == digest(second) and
            ledger['ledger_sha256'] == digest({k: v for k, v in ledger.items() if k != 'ledger_sha256'}), 'packet/ledger identity mismatch')
    require(ledger['generator_status'] in ('ready', 'empty', 'overflow', 'invalid_query_support',
            'unbracketed_query', 'observation_budget_exceeded') and
            (ledger['generator_status'] == 'ready') == bool(candidates), 'generator availability mismatch')


def first_pass(bundle, record):
    validate_bundle(bundle)
    require(type(record) is dict and set(record) == {'listener_id', 'view_sha256', 'context_sufficient',
            'followability', 'accepted_clocks', 'transition_intervals_s'}, 'invalid first-pass fields')
    require(identifier(record['listener_id']) and record['view_sha256'] == digest(bundle['first_view']), 'first-pass identity mismatch')
    require(record['context_sufficient'] is None or type(record['context_sufficient']) is bool, 'invalid context judgement')
    require(record['followability'] in ('followable', 'not_followable', 'uncertain'), 'invalid followability judgement')
    accepted = record['accepted_clocks']
    require(type(accepted) is list and len(accepted) <= clocks.CAP, 'invalid first-pass clocks')
    query = bundle['first_view']['query_s']
    for clock in accepted:
        clocks.validate_clock(clock)
        require(clock['knots'][0][0] <= query[0] < query[1] <= clock['knots'][-1][0], 'first-pass clock misses query')
    require((record['followability'] == 'followable') == bool(accepted), 'followable first pass needs its own clock, before proposals')
    intervals = record['transition_intervals_s']
    require(type(intervals) is list and all(type(s) is list and len(s) == 2 and
            all(clocks.number(v) for v in s) and query[0] <= s[0] <= s[1] <= query[1] for s in intervals), 'invalid transition uncertainty interval')
    return digest(record)


def verify_replay(bundle, observations):
    """Bind candidate construction to supplied observations, not just self-hashes."""
    validate_bundle(bundle)
    ledger = bundle['private_ledger']
    require(build(bundle['first_view']['input'], ledger['provenance'], observations,
                  bundle['first_view']['query_s'], ledger['nonce']) == bundle, 'observation-only packet replay changed')


def reconcile(bundle, records, votes):
    validate_bundle(bundle)
    require(type(records) is list and type(votes) is list and len(records) == len(votes) == 2, 'two independent complete records required')
    hashes = [first_pass(bundle, record) for record in records]
    require(records[0]['listener_id'] != records[1]['listener_id'], 'duplicate listener')
    projected = []
    for record, identity, vote in zip(records, hashes, votes):
        require(type(vote) is dict and set(vote) == {'listener_id', 'view_sha256', 'first_pass_sha256', 'labels'}, 'invalid second-pass fields')
        require(vote['listener_id'] == record['listener_id'] and vote['first_pass_sha256'] == identity and
                vote['view_sha256'] == digest(bundle['candidate_view']), 'second pass not bound to its original first pass/view')
        projected.append({k: vote[k] for k in ('listener_id', 'view_sha256', 'labels')})
    ids = [c['id'] for c in bundle['candidate_view']['candidates']]
    if not ids:
        require(all(v['labels'] == {} for v in votes), 'labels for absent candidates')
        result = dict(targets={}, reasons={}, known=0, unknown=0)
    else:
        result = consensus(ids, projected)
    # Context uncertainty is unavailable evidence even if overlay votes are known.
    if any(r['context_sufficient'] is not True for r in records):
        result = dict(targets=dict.fromkeys(ids), reasons=dict.fromkeys(ids, 'context_unavailable'), known=0, unknown=len(ids))
    followable = all(r['followability'] == 'followable' and r['context_sufficient'] is True for r in records)
    result['coverage_status'] = ('supported_candidate' if any(v == 1 for v in result['targets'].values()) else
        'generator_miss' if followable and (not ids or all(v == 0 for v in result['targets'].values())) else 'unassessed')
    result['training_authorized'] = False
    return result
