"""Replay the conditional report from retained features, without model execution.

This checks archive identity, clocks and report accounting. It cannot reproduce
or replace the recorded neural reference/hook/repeat checks from stitched arrays.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import musicfm_paired as paired
import musicfm_paired_probe as probe


def verify(report, feature_directory):
    plan, base, _, cases, selected = probe.protocol()
    require = probe.loader.require
    require(report['schema'] == 'rhythm-map.musicfm-paired.v1' and report['complete'] is True and
            report['extraction_fidelity_pass'] is True and report['conditional_only'] is True and
            report['decision'] == 'conditional_diagnostic_only_no_product_admission' and
            all(report[k] is False for k in ('training', 'holdout_access', 'production_change')),
            'not a completed conditional diagnostic')
    require(report['helper_sha256'] == plan['helpers'] and
            report['protocol_sha256'] == probe.loader.digest(probe.LOCK.read_bytes()) and
            report['known_no_rhythm_failure_report_sha256'] == plan['helpers']['musicfm-negative-v1.json'],
            'report protocol differs')
    negative = probe.loader.strict_json((probe.HERE / 'musicfm-negative-v1.json').read_bytes())
    runtime = probe.loader.runtime_lock()
    require(report['checkpoint_sha256'] == base['checkpoint']['sha256'] and
            report['runtime_versions'] == {p['name']: p['version'] for p in runtime['packages']} and
            report['initial_model_state_sha256'] == report['final_model_state_sha256'] ==
            negative['final_model_state_sha256'], 'recorded model/runtime identity differs')
    require([c['id'] for c in report['captures']] == plan['selected_ids'], 'capture population differs')
    by_id = {c['identity']['id']: c for c in cases}
    expected_cases = [dict(c['identity'], cohort=c['cohort'],
        disposition='selected_pair' if c['identity']['id'] in plan['selected_ids'] else
        'expressive_untyped' if c['data']['untyped'] else 'no_registered_pair') for c in cases]
    require(report['input_cases'] == expected_cases, 'forty-ID accounting differs')
    pairs = []
    for selected_pair, recorded in zip(selected, report['captures']):
        name = selected_pair['id']
        h = np.load(feature_directory / (name + '.features.npy'), allow_pickle=False)
        owner = np.load(feature_directory / (name + '.owners.npy'), allow_pickle=False)
        require(probe.loader.digest(h.tobytes()) == recorded['feature_sha256'] and
                probe.loader.digest(owner.tobytes()) == recorded['owner_sha256'], 'archive identity differs')
        require(recorded['sample_rate'] == 24000 and len(h) == recorded['tokens'] ==
                probe.temporal.tokens(recorded['samples']) and recorded['state_and_input_unchanged'] is True and
                recorded['private_archive_roundtrip_exact'] is True, 'complete feature coverage differs')
        contexts = probe.temporal.chunks(recorded['samples'])
        require(len(recorded['contexts']) == len(contexts), 'context coverage differs')
        for actual, expected in zip(recorded['contexts'], contexts):
            require(all(actual[k] == v for k, v in expected.items()) and
                    actual['reference_hook_repeat_exact'] is True and actual['all_blocks_and_projection'] is True and
                    np.all(owner[expected['own_start']:expected['own_stop']] == expected['id']),
                    'recorded context/owner contract differs')
        pairs.append(dict(id=name, pair_sha256=paired.oracle.exposure.canonical_hash(selected_pair),
            **paired.pair_result(h, owner, by_id[name]['data'], selected_pair)))
    require(pairs == report['pairs'] and paired.summarize(pairs, plan['selected_ids']) == report['summary'],
            'conditional replay differs')
    return report['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--feature-directory', type=Path, required=True)
    args = parser.parse_args()
    report = probe.loader.strict_json(args.report.read_bytes())
    print(json.dumps(verify(report, args.feature_directory), indent=2))


if __name__ == '__main__':
    main()
