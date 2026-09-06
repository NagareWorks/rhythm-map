"""Replay seven frozen annotation-selected pairs, not an automatic tempo decoder.

Regenerate and verify the public plan before any private capture access. Source
identity and rational ledgers are reused unchanged; no score chooses exposure.
One failed window rejects both sides from paired comparison, never the denominator.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

import annotation_exposure_audit as exposure
import window_response_residual_audit as residual

HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / 'paired-response-replay-lock-v1.json'
LOCK = json.loads(LOCK_PATH.read_text())
PACKET_HASH = '6bf7aa6b88aae92b7b91767b1e63a7fe78b4b4275080dba42edd4c39ceef23e0'
packets, dense = residual.packets, residual.dense
ROLES = ('constant', 'change')
SOURCES, CLOCKS = residual.SOURCES, residual.CLOCKS


def verified_plan():
    frozen = exposure.read_json(HERE / 'annotation-exposure-v1.json', LOCK['plan_report_sha256'])
    dense.require(exposure.build_report() == frozen, 'annotation plan audit changed')
    dense.require(frozen['selected_window_frames'] == LOCK['window_frames'] and
                  frozen['selected_pair_count'] == LOCK['selected_pairs'] and
                  frozen['plan_sha256'] == LOCK['selected_plan_sha256'], 'registered plan identity changed')
    metadata, cases = exposure.load_annotations()
    plan = []
    for case in cases:
        pair = exposure.profile(case['data'], LOCK['window_frames'])['pair']
        if pair is not None:
            plan.append(dict(cohort=case['cohort'], id=case['identity']['id'], **pair))
    dense.require(exposure.canonical_hash(plan) == frozen['plan_sha256'] and
                  [p['id'] for p in plan] == frozen['selected_case_ids'] and
                  all(p['cohort'] == 'artbeat' for p in plan), 'selected coordinates changed')
    return metadata, cases, plan


def verified_helpers():
    prior, _ = dense.read_json(HERE / 'window-response-residuals-v1.json', exposure.SOURCE_HASH)
    required = dict(prior['helper_sha256'])
    required.update({'window_response_residual_audit.py': prior['script_sha256'],
                     'window-response-residuals-lock-v1.json': prior['lock_sha256']})
    packet_report, _ = dense.read_json(HERE / 'backend-response-packets-v1.json', PACKET_HASH)
    required.update(packet_report['helper_sha256'])
    dense.require(packet_report['script_sha256'] == required['backend_response_packet_audit.py'], 'packet script identity differs')
    required['backend-response-packets-lock-v1.json'] = packet_report['lock_sha256']
    required.update({'annotation_exposure_audit.py': '824832bb891637a8eafb40bd05c05e4fb5c5000577c7f08a81a10bdc2892c681',
                     'annotation-exposure-lock-v1.json': '2f9fcbc4a75ac6c30acad4d927d681d686a6a2e9d93531831f8e8e9f934c1271'})
    dense.require(all(dense.sha((HERE / p).read_bytes()) == h for p, h in required.items()), 'frozen helper changed')
    dense.require(list(SOURCES) == LOCK['sources'] and list(CLOCKS) == LOCK['clock_families'], 'source/clock set changed')
    return required, packet_report


def prepare_window(source_packets, times, start, end):
    dense.require(type(start) is int and type(end) is int and start >= 0 and end - start == LOCK['window_frames'],
                  'window outside frozen duration contract')
    selected = {s: residual.crop_packets(source_packets[s], start, end) for s in SOURCES}
    row = dict(frames=end - start, owned_responses={s: len(selected[s]) for s in SOURCES})
    grids, status = residual.supplied_clocks(times, start, end)
    if status == 'eligible' and any(len(p) > 128 for p in selected.values()):
        status = 'response_budget_exceeded'
    if status == 'eligible' and not all(residual.support_inside(p, end - start) for p in selected.values()):
        status = 'packet_support_crosses_window'
    return row | {'status': status}, selected, grids


def contrast(left, right):
    matched = left['matched'] - right['matched']
    missing = left['unmatched_ticks'] - right['unmatched_ticks']
    if matched == missing == 0:
        relation = 'equal'
    elif matched >= 0 and missing <= 0:
        relation = 'left_dominates'
    elif matched <= 0 and missing >= 0:
        relation = 'right_dominates'
    else:
        relation = 'tradeoff'
    return dict(matched_delta=matched, unmatched_ticks_delta=missing, relation=relation)


def evaluate_window(row, selected, grids):
    size = row['frames']
    results = {s: {name: packets.packet_ledger([True] * size, selected[s], grids[name])['result'] for name in CLOCKS}
               for s in SOURCES}
    inventories = {s: {name: residual.ledger_inventory(results[s][name], selected[s], grids['annotated'], size) for name in CLOCKS}
                   for s in SOURCES}
    causes, marks = residual.annotated_miss_causes(results['raw']['annotated'], results['candidate']['annotated'],
                                                  selected['candidate'], grids['annotated'])
    # Preserve the older comparator convention for the reused summary helper.
    comparisons = {s: {name: residual.compare_counts(results[s]['continuation'], results[s][name])
                       for name in CLOCKS if name != 'continuation'} for s in SOURCES}
    return row | dict(inventories=inventories, comparisons=comparisons, annotated_tick_causes=causes,
                      annotated_miss_candidate_marks=marks,
                      identical_annotated_continuation=grids['annotated'] == grids['continuation'],
                      versus_continuation={s: {name: contrast(inventories[s][name], inventories[s]['continuation'])
                                              for name in CLOCKS if name != 'continuation'} for s in SOURCES})


def analyze_pair(source_packets, times, pair):
    length = pair['window_frames']
    a, b = pair['constant_start_frame'], pair['change_start_frame']
    dense.require(length == LOCK['window_frames'] and a + length <= b, 'pair overlap or duration changed')
    prepared = {role: prepare_window(source_packets, times, start, start + length) for role, start in zip(ROLES, (a, b))}
    for source in SOURCES:
        ids = [(p['source'], p['source_index']) for role in ROLES for p in prepared[role][1][source]]
        dense.require(len(ids) == len(set(ids)), 'packet owned by both pair members or duplicate source identity')
    eligible = all(value[0]['status'] == 'eligible' for value in prepared.values())
    return dict(status='eligible' if eligible else 'pair_rejected',
                windows={role: evaluate_window(*value) if eligible else value[0] for role, value in prepared.items()})


def summarize_pairs(pairs):
    eligible = [p for p in pairs if p['status'] == 'eligible']
    return dict(selected_pairs=len(pairs), eligible_pairs=len(eligible), rejected_pairs=len(pairs) - len(eligible),
                selected_windows=2 * len(pairs), compared_windows=2 * len(eligible),
                status=dict(sorted(Counter(p['status'] for p in pairs).items())),
                window_status={role: dict(sorted(Counter(p['windows'][role]['status'] for p in pairs).items())) for role in ROLES},
                all_selected_owned_responses={s: sum(p['windows'][r]['owned_responses'][s] for p in pairs for r in ROLES) for s in SOURCES},
                by_context={role: residual.summarize([p['windows'][role] for p in eligible]) for role in ROLES},
                paired_relations={s: {name: dict(sorted(Counter(
                    '/'.join(p['windows'][r]['versus_continuation'][s][name]['relation'] for r in ROLES)
                    for p in eligible).items())) for name in CLOCKS if name != 'continuation'} for s in SOURCES})


def build_report(evidence_path, capture_dir):
    helpers, packet_report = verified_helpers()
    metadata, cases, plan = verified_plan()  # complete before any private read
    by_id = {c['identity']['id']: c for c in cases}
    count, evidence_hash = dense.INPUTS['artbeat']
    evidence, _ = dense.read_json(evidence_path, evidence_hash)
    prior = next(c for c in packet_report['cohorts'] if c['cohort'] == 'artbeat')
    summary, summary_hash = dense.read_json(Path(capture_dir) / 'summary.json', prior['capture_summary_sha256'])
    records = dense.validate_summary(summary, evidence['cases'], count)
    dense.require([c['id'] for c in evidence['cases']] == [c['id'] for c in metadata[0]['cases']], 'evidence ordered cohort changed')
    evidence_by_id = {c['id']: (c, r, old) for c, r, old in zip(evidence['cases'], records, prior['cases'])}
    sources = {k: dense.sha((dense.ROOT / p).read_bytes()) for k, p in dense.SOURCES.items()}
    pairs = []
    for selected in plan:
        identity, data = (by_id[selected['id']][k] for k in ('identity', 'data'))
        case, record, old = evidence_by_id[selected['id']]
        dense.require(case['truth_sha256'] == identity['truth_sha256'] and record['capture_sha256'] == identity['capture_sha256'] and
                      old['id'] == selected['id'], 'selected input identity differs')
        times = [Fraction(t, exposure.FRAME_US) for t in data['beats']]
        dense.require(residual.annotation_coordinates(case['truth_times_s']) == times, 'private evidence truth differs from selected annotations')
        payload, _ = dense.read_json(Path(capture_dir) / f"{selected['id']}.json", identity['capture_sha256'])
        beats = dense.validate_capture(payload, record, case, 'artbeat', sources)
        dense.require(len(beats) == identity['frame_count'], 'selected frame extent changed')
        assembled = packets.assemble(beats, payload['downbeat_logits'], payload['observations']['duration_s'], payload['observations'])
        dense.require(packets.summarize(assembled) == old['source_inventory'], 'whole-capture packet inventory changed')
        result = analyze_pair({'raw': assembled['raw'], 'candidate': assembled['candidates']}, times, selected)
        selected_counts = {s: sum(result['windows'][r]['owned_responses'][s] for r in ROLES) for s in SOURCES}
        whole = {'raw': len(assembled['raw']), 'candidate': len(assembled['candidates'])}
        dense.require(all(selected_counts[s] <= whole[s] for s in SOURCES), 'pair ownership exceeds capture')
        pairs.append(dict(cohort='artbeat', **identity, pair_sha256=exposure.canonical_hash({k: v for k, v in selected.items() if k not in ('id', 'cohort')}),
                          whole_capture_responses=whole, outside_selected_windows={s: whole[s] - selected_counts[s] for s in SOURCES}, **result))
    selected_ids = {p['id'] for p in plan}
    return dict(schema_version=1, purpose=LOCK['purpose'], contract=LOCK,
                script_sha256=dense.sha(Path(__file__).read_bytes()), lock_sha256=dense.sha(LOCK_PATH.read_bytes()),
                helper_sha256=helpers, source_hashes=sources,
                predecessor_sha256={'annotation-exposure-v1.json': LOCK['plan_report_sha256'],
                                    'window-response-residuals-v1.json': exposure.SOURCE_HASH,
                                    'backend-response-packets-v1.json': PACKET_HASH},
                selected_plan_sha256=LOCK['selected_plan_sha256'], frozen_evidence_sha256=evidence_hash,
                capture_summary_sha256=summary_hash, metadata_projection_sha256=exposure.canonical_hash(metadata),
                input_cases=[dict(cohort=c['cohort'], **c['identity'],
                                 status='selected_pair' if c['identity']['id'] in selected_ids else 'not_selected_by_frozen_annotation_plan',
                                 private_capture_read=c['identity']['id'] in selected_ids) for c in cases],
                private_capture_count=len(pairs), complete=True, truth_assisted=True, observation_ledger_replayed=True,
                selected_tempo=None, source_pooling=False, phase_search=False, neural_inference=False, decoder_replayed=False,
                fitted_mapping=False, holdout_opened=False, training_run=False, production_output_changed=False,
                new_user_parameters=False, accuracy_improvement_claimed=False, pairs=pairs, summary=summarize_pairs(pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artbeat-evidence', type=Path, required=True)
    parser.add_argument('--artbeat-captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.artbeat_evidence, args.artbeat_captures)
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
