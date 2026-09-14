"""Route already retained gradients once; no CNN, clock or optimizer execution."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import tarfile
import time

import numpy as np
import torch

from experiments.gradient_audit.audit import cosine
from experiments.gradient_audit.register import sources as old_sources, write_json
from experiments.observed_adjoint.transport import unpack_gradient
from experiments.phase_attribution.run import ROOT, sha, verify
from experiments.scaled_adjoint.scaled import accumulate, require
from experiments.gradient_routing.route import GROUPS, partition, route

HERE = Path(__file__).resolve().parent
PARENT = ROOT / 'experiments/gradient_audit/results-v1.json'
ARCHIVE_SHA256 = 'befd8d11cadc88e9b2ce8699f0a1efb731d3f5bb6a7dc3ed603456e7a766f884'
PRIVATE_REPORT_SHA256 = '19a3b3d540fa968aa8ec37a63712d5953b9f53cbf1b14430d4050b9ac54d4a91'


def sources():
    parent = json.loads(PARENT.read_bytes())
    require(parent['source_sha256'] == old_sources(), 'old closed experiment changed')
    paths = set(parent['source_sha256']) | {PARENT.relative_to(ROOT).as_posix()}
    paths.update(p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py'))
    paths.add((HERE / 'PROTOCOL-v1.md').relative_to(ROOT).as_posix())
    return {n: sha(ROOT / n) for n in sorted(paths)}


def make_plan():
    parent = json.loads(PARENT.read_bytes())
    require(parent['private_report_sha256'] == PRIVATE_REPORT_SHA256, 'wrong parent report')
    return dict(schema='rhythm-map.gradient-routing-plan.v1', source_sha256=sources(),
                archive_sha256=ARCHIVE_SHA256, private_report_sha256=PRIVATE_REPORT_SHA256,
                population=[{k: r[k] for k in ('id', 'checkpoint', 'role', 'work', 'gradient_packet_sha256')}
                            for r in parent['cases']], groups=list(GROUPS), seconds=120,
                model_forwards=0, clock_forwards=0, optimizer_updates=0,
                holdout_access=False, prior_used_for_routing=False)


def inspect(gradients):
    phase, count, prior = (gradients[k] for k in ('phase', 'count', 'prior'))
    original = accumulate([phase, count])
    routed, receipt = route(phase, count)
    groups = {k: partition(v) for k, v in
              dict(original=original, routed=routed, count=count, prior=prior).items()}
    for name, row in receipt.items():
        row.update(original_count_cosine=cosine(groups['original'][name], groups['count'][name]),
                   original_prior_cosine=cosine(groups['original'][name], groups['prior'][name]),
                   routed_prior_cosine=cosine(groups['routed'][name], groups['prior'][name]))
    return dict(groups=receipt, all_parameters=dict(original_count_cosine=cosine(original, count),
                routed_count_cosine=cosine(routed, count), original_prior_cosine=cosine(original, prior),
                routed_prior_cosine=cosine(routed, prior)))


def audit(archive, output, plan_path, expected_plan_sha256):
    verify(plan_path, expected_plan_sha256)
    plan = json.loads(plan_path.read_bytes())
    require(plan == make_plan(), 'registered routing plan differs')
    verify(archive, plan['archive_sha256'])
    require(not output.exists() and not output.resolve().is_relative_to(ROOT), 'new private output required')
    output.mkdir(parents=True)
    expected = {'output/run/' + r['id'] + '.' + r['checkpoint'] + '.gradients.pt': r
                for r in plan['population']}
    require(len(expected) == len(plan['population']) == 80, 'wrong packet population')
    rows, fit, seen = {}, {}, set()
    report_seen = False
    started, current = time.monotonic(), None
    try:
        # Stream without extracting paths from the archive or retaining field arrays.
        with tarfile.open(archive, 'r|gz') as stream:
            for member in stream:
                require(time.monotonic() - started < plan['seconds'], 'packet audit budget exceeded')
                current = member.name
                if current == 'output/run/report.json':
                    require(not report_seen and member.isfile(), 'duplicate or non-file parent report')
                    require(hashlib.sha256(stream.extractfile(member).read()).hexdigest() == PRIVATE_REPORT_SHA256,
                            'private parent report changed')
                    report_seen = True
                if current not in expected:
                    continue
                require(current not in seen and member.isfile() and member.size <= 64 * 1024**2,
                        'duplicate, non-file or oversized gradient packet')
                row = expected[current]
                data = stream.extractfile(member).read()
                require(hashlib.sha256(data).hexdigest() == row['gradient_packet_sha256'], 'gradient hash differs')
                packet = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
                require((packet['id'], packet['checkpoint']) == (row['id'], row['checkpoint']), 'packet identity differs')
                gradients = {k: unpack_gradient(v) for k, v in packet['gradients']['parameter'].items()}
                require(set(gradients) == {'phase', 'count', 'total', 'prior'}, 'wrong gradient terms')
                key = (row['id'], row['checkpoint'])
                rows[key] = {**row, 'diagnostic': inspect(gradients)}
                if row['role'] == 'fit':
                    fit[key] = {k: gradients[k] for k in ('phase', 'count', 'prior')}
                write_json(output / (row['id'] + '.' + row['checkpoint'] + '.json'), rows[key])
                seen.add(current)
        require(seen == set(expected) and report_seen, 'incomplete packet or report archive')
        ordered = [rows[(r['id'], r['checkpoint'])] for r in plan['population']]
        fit_rows = [r for r in plan['population'] if r['role'] == 'fit' and r['checkpoint'] == 'audio']
        counts = Counter(r['work'] for r in fit_rows)
        require(len(fit_rows) == 20 and len(counts) == 9, 'fit population changed')
        aggregate = {}
        for kind in ('audio', 'zero'):
            gradients = {k: accumulate([fit[(r['id'], kind)][k].multiply(1 / (9 * counts[r['work']]))
                                        for r in fit_rows]) for k in ('phase', 'count', 'prior')}
            aggregate[kind] = inspect(gradients)
        require(plan == make_plan(), 'sources changed during audit')
        result = dict(schema='rhythm-map.gradient-routing.v1', plan_sha256=expected_plan_sha256,
                      source_sha256=plan['source_sha256'], archive_sha256=plan['archive_sha256'],
                      runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                      cases=ordered, fit_aggregate=aggregate, elapsed_s=time.monotonic() - started,
                      model_forwards=0, clock_forwards=0, optimizer_updates=0, holdout_access=False,
                      prior_used_for_routing=False, decision='mechanism_only_adamw_boundary_not_admitted')
        write_json(output / 'report.json', result)
        return result
    except BaseException as error:
        write_json(output / 'failure.json', dict(current_member=current, complete_cases=len(seen),
                   elapsed_s=time.monotonic() - started, error_type=type(error).__name__, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--expected-plan-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.register is not None:
        require(not any((args.archive, args.plan, args.output, args.expected_plan_sha256)), 'register only')
        require(not args.register.resolve().is_relative_to(ROOT), 'private plan required')
        write_json(args.register, make_plan())
        print(sha(args.register), flush=True)
    else:
        require(all((args.archive, args.plan, args.output, args.expected_plan_sha256)), 'complete audit args required')
        result = audit(args.archive, args.output, args.plan, args.expected_plan_sha256)
        print(json.dumps(dict(cases=len(result['cases']), elapsed_s=result['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
