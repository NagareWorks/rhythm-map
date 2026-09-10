"""Replay three CLOSED heads under four fixed inputs; never update any weight."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.clock_readout.model import ClockReadout, HALO
from experiments.coupled_clock.model import CoupledClockReadout
from experiments.coupled_clock.clock import integrate, require
from experiments.coupled_clock.prepare import replay_v1
from experiments.coupled_clock.measurement import clock_fields, independent_fields, measure, macro
from experiments.trajectory_clock.attribution import ROOT, INPUT_SHA, sha, write_json
from experiments.clock_evidence.diagnostic import VARIANTS, intervention, interior, field_response, paired_summary

PARENTS = {
    'direct': ('clock_readout', '990f7a2c71b2d3d61ab77c81daa69c3c5add3a78ead43d776fbf03d25ed07736'),
    'coupled': ('coupled_clock', 'e33a44db4ce20a35f0676aa3f4fc47806da83f4a2ab80c1cfb555ce5343cda9a'),
    'trajectory': ('trajectory_clock', 'a20e960c8fa7d3ffffdfab24e9e2d73fb2a0ab78069cc8a33d7e1cfd42d5a8e1'),
}


def source_hashes():
    folders = ('clock_readout', 'coupled_clock', 'trajectory_clock', 'clock_evidence')
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
            for folder in folders for p in sorted((ROOT / 'experiments' / folder).glob('*.py'))}


def load_model(name, path, expected):
    require(sha(path) == expected, 'weight identity changed')
    model = (ClockReadout() if name == 'direct' else CoupledClockReadout()).eval()
    with np.load(path, allow_pickle=False) as archive:
        model.load_state_dict({key: torch.from_numpy(archive[key].copy()) for key in archive.files})
    model.requires_grad_(False)
    return model


def predict(model, features, direct):
    with torch.inference_mode():
        if direct:
            # Exact original 200-frame ownership and raw-feature zero padding.
            fields = replay_v1(model, features)
            return fields, fields
        fields = model.fields(torch.from_numpy(features)[None])
        q = integrate(fields[:, :-1, 0], fields[:, 0, 1]).cycles[0].numpy()
        return fields[0].numpy(), q


def musical(prediction, direct, reference, common):
    return measure(independent_fields(prediction) if direct else clock_fields(prediction), reference, common)


def summarize(rows):
    result = {}
    for role in ('fit', 'development', 'diagnostic'):
        result[role] = {}
        for name in PARENTS:
            subset = [r for r in rows if r['role'] == role and r['model'] == name]
            metrics = {v: macro([dict(work=r['work'], **r['metrics']) for r in subset], v)
                       for v in VARIANTS + ('fitted_zero', 'raw')}
            result[role][name] = dict(metrics=metrics,
                paired={v: paired_summary(subset, v) for v in VARIANTS[1:] + ('fitted_zero',)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'direct-results', 'coupled-results', 'trajectory-results', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'new private output required')
    require(sha(args.inputs / 'inputs.json') == INPUT_SHA, 'input manifest changed')
    manifest = json.loads((args.inputs / 'inputs.json').read_bytes())
    require(manifest['complete'] and not manifest['holdout_access'] and len(manifest['cases']) == 40,
            'unexpected input population')
    paths = {name: getattr(args, name + '_results') for name in PARENTS}
    parents, models = {}, {}
    for name, (folder, expected) in PARENTS.items():
        public = ROOT / 'experiments' / folder / 'results-v1.json'
        require(sha(public) == sha(paths[name] / 'report.json') == expected, 'parent report identity changed')
        report = json.loads(public.read_bytes())
        require(not report['holdout_access'] and not report['production_change']
                and [(r['id'], r['role'], r['work'], r['frames']) for r in report['cases']] ==
                    [(r['id'], r['role'], r['work'], r['frames']) for r in manifest['cases']], 'parent population differs')
        parents[name] = report
        models[name] = {fit['kind']: load_model(name, paths[name] / (fit['kind'] + '.weights.npz'), fit['weight_sha256'])
                        for fit in report['fits']}
        require(set(models[name]) == {'audio', 'zero-audio'}, 'missing matched fitted control')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir()
    execution = dict(schema='rhythm-map.clock-evidence-execution.v1', inputs_sha256=INPUT_SHA,
        parent_report_sha256={k: v[1] for k, v in PARENTS.items()}, source_sha256=source_hashes(),
        variants=list(VARIANTS), fitted_zero_control=True, device='cpu', torch_version=torch.__version__,
        numpy_version=np.__version__, readout_halo_frames=HALO,
        shift_rule='floor(frames/2); one value fixed by geometry, no offset search',
        training=False, encoder_inference=False, holdout_access=False, production_change=False,
        independent_acceptance=False, weights_fixed=True)
    write_json(args.output / 'execution.json', execution)
    started, rows, replays, peak_equivariance = time.monotonic(), [], [], 0.
    try:
        for index, item in enumerate(manifest['cases']):
            packet = args.inputs / (item['id'] + '.npz')
            require(sha(packet) == item['packet_sha256'], 'packet identity changed')
            with np.load(packet, allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            require(arrays['hidden'].shape == (item['frames'], 512), 'feature geometry differs')
            common = arrays['valid'] & arrays['raw_valid']
            safe = interior(item['frames'])
            require(safe.any(), 'no seam-free interior')
            for name in PARENTS:
                direct = name == 'direct'
                previous = parents[name]['cases'][index]
                saved = None
                if not direct:
                    path = paths[name] / (item['id'] + '.prediction.npz')
                    require(sha(path) == previous['prediction_sha256'], 'retained prediction changed')
                    with np.load(path, allow_pickle=False) as archive:
                        saved = {k: archive[k].copy() for k in archive.files}
                outputs, fields, metrics, responses = {}, {}, {}, {}
                for variant in VARIANTS:
                    x = intervention(arrays['hidden'], variant)
                    fields[variant], outputs[variant] = predict(models[name]['audio'], x, direct)
                    require(np.isfinite(outputs[variant]).all(), 'nonfinite required output')
                    metrics[variant] = musical(outputs[variant], direct, arrays['reference'], common)
                _, outputs['fitted_zero'] = predict(models[name]['zero-audio'], intervention(arrays['hidden'], 'zero'), direct)
                metrics['fitted_zero'] = musical(outputs['fitted_zero'], direct, arrays['reference'], common)
                metrics['raw'] = measure(clock_fields(arrays['raw']), arrays['reference'], common)
                expected = arrays['v1'] if direct else saved['audio']
                np.testing.assert_allclose(outputs['natural'], expected, atol=1e-5, rtol=2e-5)
                replay_error = float(np.max(np.abs(outputs['natural'] - expected)))
                if not direct:
                    np.testing.assert_allclose(outputs['fitted_zero'], saved['zero'], atol=1e-5, rtol=2e-5)
                    replay_error = max(replay_error, float(np.max(np.abs(outputs['fitted_zero'] - saved['zero']))))
                replays.append(replay_error)
                # Local-field translation check only. It does not align the
                # output clock, change musical masks or use truth as input.
                shifted = np.roll(fields['natural'], item['frames'] // 2, axis=0)
                np.testing.assert_allclose(fields['half_roll'][safe], shifted[safe], atol=1e-5, rtol=2e-5)
                equivariance = float(np.max(np.abs(fields['half_roll'][safe] - shifted[safe])))
                peak_equivariance = max(peak_equivariance, equivariance)
                for variant in VARIANTS[1:]:
                    responses[variant] = field_response(fields['natural'], fields[variant], safe & common, direct)
                path = args.output / (item['id'] + '.' + name + '.npz')
                with path.open('xb') as stream:
                    np.savez(stream, **outputs, **{'fields_' + k: v for k, v in fields.items()})
                rows.append(dict(id=item['id'], role=item['role'], work=item['work'], model=name,
                    frames=item['frames'], packet_sha256=item['packet_sha256'], output_sha256=sha(path),
                    replay_max_absolute_difference=replay_error, local_roll_max_absolute_difference=equivariance,
                    metrics=metrics, field_response=responses))
            print(json.dumps(dict(event='recording_complete', recording=index + 1, total=40)), flush=True)
        require(source_hashes() == execution['source_sha256'], 'source changed during diagnostic')
        report = dict(schema='rhythm-map.clock-evidence.v1', execution_sha256=sha(args.output / 'execution.json'),
            inputs_sha256=INPUT_SHA, parent_report_sha256=execution['parent_report_sha256'],
            cases=rows, summary=summarize(rows), elapsed_s=time.monotonic() - started,
            replay_max_absolute_difference=max(replays), local_roll_max_absolute_difference=peak_equivariance,
            recordings=40, checkpoints=6, audio_checkpoint_sequence_evaluations=480, fitted_zero_sequence_evaluations=120,
            training=False, encoder_inference=False, weights_fixed=True, holdout_access=False,
            independent_acceptance=False, production_change=False, promotion_gate=False,
            interpretation='post-fit input-dependence diagnostic; interventions may be out of distribution; no causal training-mechanism proof')
        write_json(args.output / 'report.json', report)
        print(json.dumps(dict(event='diagnostic_complete', recordings=40, elapsed_s=report['elapsed_s'])), flush=True)
    except Exception as error:
        write_json(args.output / 'failure.json', dict(exception_type=type(error).__name__, automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
