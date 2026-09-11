"""Source/runtime registration without reading music or checkpoints."""
import argparse
import json
from pathlib import Path

from experiments.coupled_clock.clock import require
from experiments.phase_attribution.run import ROOT, PINS as PRIOR_PINS, REPORT, sha, verify
from experiments.scaled_phase_fit.register import configure, runtime_identity

HERE = Path(__file__).resolve().parent
PINS = {**PRIOR_PINS, ROOT / 'experiments/phase_attribution/results-v1.json':
        '3e001e734e09e322e914f9830fbd053dd26644afe32426e72ed8b3b2b99c1d38'}


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sources():
    for path, digest in PINS.items():
        verify(path, digest)
    old = json.loads(REPORT.read_bytes())
    paths = set(old['source_sha256'])
    attribution = json.loads((ROOT / 'experiments/phase_attribution/results-v1.json').read_bytes())
    for name, digest in {**old['source_sha256'], **attribution['source_sha256']}.items():
        verify(ROOT / name, digest)
        paths.add(name)
    paths.update(p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py'))
    paths.add((HERE / 'PROTOCOL-v1.md').relative_to(ROOT).as_posix())
    return {name: sha(ROOT / name) for name in sorted(paths)}


def make_plan():
    configure('cuda')
    runtime = runtime_identity('cuda')
    old = json.loads(REPORT.read_bytes())
    original_runtime = json.loads((ROOT / 'experiments/scaled_phase_fit/plan-v1.json').read_bytes())['runtime']
    require(runtime == original_runtime, 'selected-checkpoint target runtime differs')
    return dict(schema='rhythm-map.gradient-audit-plan.v1', runtime=runtime, source_sha256=sources(),
                prerequisite_sha256={p.relative_to(ROOT).as_posix(): d for p, d in PINS.items()},
                population=[{k: r[k] for k in ('id', 'role', 'work', 'frames', 'packet_sha256', 'prediction_sha256')}
                            for r in old['cases']],
                weights={k: old['fits'][i]['weight_sha256'] for i, k in enumerate(('audio', 'zero'))},
                selected_epoch={k: old['fits'][i]['selected_epoch'] for i, k in enumerate(('audio', 'zero'))},
                audit_seconds=1200, watchdog_seconds=1500, record_checkpoint_pairs=80,
                cnn_field_forwards=320, clock_forwards=240, training=False, automatic_retry=False,
                encoder_inference=False, holdout_access=False, production_change=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private plan required')
    write_json(args.output, make_plan())
    print(sha(args.output), flush=True)


if __name__ == '__main__':
    main()
