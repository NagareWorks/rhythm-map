"""Pre-fit registration; no music packets or old learned weights are opened."""
import argparse
import json
from pathlib import Path

from experiments.scaled_phase_fit import register as previous
from experiments.coupled_clock.prepare import ROOT, sha, write_json
from experiments.coupled_clock.clock import require

HERE = Path(__file__).resolve().parent
PINS = {
    'experiments/scaled_phase_fit/plan-v1.json': 'c159b0f90369793d67cae5d8fc8c94264252d152dfb2e3de6eec257967f55193',
    'experiments/scaled_phase_fit/results-v1.json': '73454ce089cd3821c51d622ccbd491c56ecc53765e9552f6910d4a56e6c840a5',
    'experiments/scaled_phase_fit/execution-v1.json': '73d72d2c600b9dd00589355c2f850adf62c800037758b202438c294902122593',
    'experiments/gradient_routing/results-v1.json': 'a6d31925ac506abea605fa367132c62b7a8296a38daf0cea20626589480bfcf9',
    'experiments/protected_update/authored-windows-v1.json': 'bd578d4adb1cc43916ce1fc74a6c704c995bdd9d47daca0117a5c995fc823c6a',
}


def source_hashes():
    require(all(sha(ROOT / name) == digest for name, digest in PINS.items()), 'closed prerequisite identity changed')
    old = json.loads((ROOT / 'experiments/scaled_phase_fit/plan-v1.json').read_bytes())
    require(old['source_sha256'] == previous.source_hashes(), 'closed fit source changed')
    authored = json.loads((ROOT / 'experiments/protected_update/authored-windows-v1.json').read_bytes())
    require(authored['gate_passed'] and authored['musical_updates'] == 0 and not authored['admission'],
            'optimizer authored prerequisite failed')
    require(all(sha(ROOT / name) == digest for name, digest in authored['source_sha256'].items()),
            'protected boundary source changed')
    paths = set(previous.source_hashes()) | set(authored['source_sha256'])
    paths.update(p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py'))
    paths.add((HERE / 'PROTOCOL-v1.md').relative_to(ROOT).as_posix())
    return {name: sha(ROOT / name) for name in sorted(paths)}


def make_plan(device):
    old = previous.make_plan(device)
    return dict(old, schema='rhythm-map.protected-phase-plan.v1', source_sha256=source_hashes(),
                prerequisite_sha256={**old['prerequisite_sha256'], **PINS},
                optimizer=dict(name='ProtectedAdamW', learning_rate=.001, weight_decay=.0001, max_norm=1.),
                recorder='experiments.protected_phase_fit.recorder.MusicalRecorder',
                transport='separate_original_phase_and_count_observed_transports',
                complete_record_gradients=800, complete_records=400, protected_step_receipts=100,
                post_step_audits='same_complete_weighted_batch_original_losses_no_update_decision',
                changed_mechanism='count_priority_gradient_routing_plus_native_displacement_protection',
                old_weights_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private plan required')
    write_json(args.output, make_plan(args.device))
    print('PROTECTED_PHASE_PLAN_READY ' + sha(args.output), flush=True)


if __name__ == '__main__':
    main()
