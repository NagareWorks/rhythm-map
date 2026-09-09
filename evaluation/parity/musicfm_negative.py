"""Frozen negative/robustness checks, not a no-rhythm decoder or confidence head."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import musicfm_availability as availability
import musicfm_fidelity as fidelity
import musicfm_fidelity_compat as compat
import musicfm_loader as loader
import musicfm_temporal as temporal
import musicfm_temporal_probe as previous

LOCK_SHA = '542fedd633dd92c6d8ec36d74942eab26e21e73972c9938827fd1ef18a475c28'
FAMILIES = (('fast', temporal.Fraction(25, 4)), ('slow', temporal.Fraction(20)))


def protocol():
    raw = (HERE / 'musicfm-negative-lock-v1.json').read_bytes()
    loader.require(loader.digest(raw) == LOCK_SHA, 'negative protocol changed')
    plan = loader.strict_json(raw)
    loader.require(loader.digest(Path(availability.__file__).read_bytes()) == plan['availability_helper_sha256'],
                   'availability helper changed')
    raw = (HERE / 'musicfm-availability-v1.json').read_bytes()
    loader.require(loader.digest(raw) == plan['availability_report_sha256'], 'availability report changed')
    activity, old = availability.protocol()
    _, base, compatibility = previous.protocol()
    return plan, activity, old, base, compatibility


def inputs(plan):
    import numpy as np
    controls = availability.controls()
    selected = {}
    for row in plan['cases']:
        pcm = controls[row['id']]
        loader.require(pcm.dtype == np.float32 and pcm.shape == (plan['samples_per_case'],) and
                       np.isfinite(pcm).all() and loader.digest(pcm.tobytes()) == row['pcm_sha256'],
                       'frozen control PCM changed')
        selected[row['id']] = pcm
    loader.require(len(selected) == plan['new_case_count'], 'incomplete control population')
    return selected


def negative_verdict(row):
    if row['status'] in ('zero_norm_support', 'no_common_support'):
        loader.require(row['clocks'] is None, 'unavailable row has clocks')
        return dict(abstains=True, reason=row['status'], clock_spread=None)
    loader.require(row['status'] == 'eligible' and set(row['clocks']) == set(temporal.NAMES),
                   'invalid negative clock ledger')
    values = [entry['score'] for entry in row['clocks'].values()]
    loader.require(all(type(v) in (int, float) and math.isfinite(v) for v in values), 'invalid clock score')
    spread = max(values) - min(values)
    return dict(abstains=spread <= temporal.TIE,
                reason='numerical_tie' if spread <= temporal.TIE else 'spurious_clock_preference',
                clock_spread=spread)


def judge(scores, old, plan):
    loader.require(list(scores) == [c['id'] for c in plan['cases']] and
                   all(set(s) == {'fast', 'slow'} for s in scores.values()), 'incomplete score population')
    pairs, negatives = [], []
    for case in plan['cases']:
        name = case['id']
        for speed, _ in FAMILIES:
            if case['kind'] == 'constant':
                change = 'step-' + speed
                pairs.append(dict(constant=name, change=change,
                                  **temporal.verdict(scores[name][speed], old['discrimination']['scores'][change][speed])))
            else:
                negatives.append(dict(id=name, family=speed, **negative_verdict(scores[name][speed])))
    loader.require(len(pairs) == plan['pair_denominator'] and len(negatives) == plan['negative_family_denominator'],
                   'verdict denominator changed')
    return dict(scores=scores, pairs=pairs, negatives=negatives, paired_denominator=len(pairs),
                negative_case_denominator=plan['negative_case_denominator'], negative_family_denominator=len(negatives),
                shape_supported=sum(p['shape_supported'] for p in pairs),
                density_resolved=sum(p['density_resolved'] for p in pairs),
                negative_families_abstaining=sum(n['abstains'] for n in negatives),
                rule_survives_authored=all(p['shape_supported'] and p['density_resolved'] for p in pairs) and
                all(n['abstains'] for n in negatives))


def prepare_activity(executable, output, cases, plan, activity):
    rows = []
    for name, pcm in cases.items():
        packet, mask, sha = availability.run_native(executable, output, name, pcm, activity)
        gates = {speed: availability.availability(mask, plan['window_start'], period) for speed, period in FAMILIES}
        loader.require(all(g['status'] == plan['expected_availability'] for g in gates.values()),
                       'prior activity availability changed')
        rows.append(dict(id=name, pcm_sha256=sha, gates=gates,
                         native_source_sha256=packet['source_sha256']))
    return rows


def extract(model, torch, cases, output, report, old):
    import numpy as np
    state = fidelity.state_hash(model, torch)
    loader.require(state == old['initial_model_state_sha256'] == old['final_model_state_sha256'],
                   'previous model state differs')
    metadata = json.dumps(model.stat, sort_keys=True) + model.conformer.config.to_json_string()
    cpu_rng, py_rng = torch.random.get_rng_state().clone(), random.getstate()
    report['initial_model_state_sha256'] = state
    extracted = {}
    with ExitStack() as stack:
        def forbidden(*_args, **_kwargs):
            raise ValueError('training/quantizer entrypoint forbidden')
        for method in ('forward', 'get_targets', 'masking', 'tokenize'):
            stack.enter_context(patch.object(model, method, forbidden))
        for name, pcm in cases.items():
            start, sha = time.perf_counter(), loader.digest(pcm.tobytes())
            chunks, features, ledger = temporal.chunks(len(pcm)), [], []
            for chunk in chunks:
                x = torch.from_numpy(pcm[chunk['sample_start']:chunk['sample_stop']]).unsqueeze(0)
                reference = fidelity.reference_pass(model, x, torch)
                captured = previous.captured_chunk(model, x, torch)
                repeat = fidelity.reference_pass(model, x, torch)
                loader.require(torch.equal(reference, captured) and torch.equal(reference, repeat), 'context parity failed')
                features.append(reference[0].numpy().copy())
                ledger.append(dict(**chunk, feature_sha256=fidelity.tensor_hash(reference, torch),
                                   reference_hook_repeat_exact=True, all_blocks_and_projection=True))
            hidden, owner = temporal.stitch(features, chunks, len(pcm))
            loader.require(loader.digest(pcm.tobytes()) == sha and fidelity.state_hash(model, torch) == state and
                           metadata == json.dumps(model.stat, sort_keys=True) + model.conformer.config.to_json_string() and
                           all(not m.training for m in model.modules()) and
                           torch.equal(cpu_rng, torch.random.get_rng_state()) and random.getstate() == py_rng,
                           'state/input/mode/RNG changed')
            archives = {}
            for kind, value in (('features', hidden), ('owners', owner)):
                path = output / (name + '.' + kind + '.npy')
                with path.open('xb') as stream:
                    np.save(stream, value, allow_pickle=False)
                identity = dict(size_bytes=path.stat().st_size, sha256=loader.digest(path.read_bytes()))
                with loader.verified_file(path, identity['size_bytes'], identity['sha256']) as stream:
                    loader.require(np.array_equal(np.load(stream, allow_pickle=False, max_header_size=4096), value),
                                   'private archive round-trip failed')
                archives[kind] = identity
            extracted[name] = hidden, owner
            report['cases'].append(dict(id=name, samples=len(pcm), tokens=len(hidden), pcm_sha256=sha,
                                        feature_sha256=loader.digest(hidden.tobytes()), owner_sha256=loader.digest(owner.tobytes()),
                                        contexts=ledger, archives=archives, state_and_input_unchanged=True,
                                        private_archive_roundtrip_exact=True, elapsed_s=time.perf_counter() - start))
            print(json.dumps(dict(case=name, extraction_complete=True)), flush=True)
    report['final_model_state_sha256'] = fidelity.state_hash(model, torch)
    return extracted


def score_extracted(extracted, gates, plan):
    loader.require(list(extracted) == [c['id'] for c in plan['cases']] and
                   [g['id'] for g in gates] == list(extracted), 'incomplete extraction/availability population')
    scores = {}
    for gate in gates:
        name = gate['id']
        hidden, owner = extracted[name]
        scores[name] = {}
        for speed, period in FAMILIES:
            row = availability.score_if_available(gate['gates'][speed], lambda: temporal.evaluate(
                hidden[plan['window_start']:plan['window_stop']], owner[plan['window_start']:plan['window_stop']],
                0, temporal.Fraction(25, 2), period, 50))
            loader.require(row['feature_access'], 'activity changed after preflight')
            scores[name][speed] = {k: v for k, v in row.items() if k != 'feature_access'}
    return scores


def worker(args, output, plan, activity, old, base, compatibility):
    from musicfm_loader_probe import bound_windows_process
    from prehead_capture import peak_rss_bytes
    loader.require(bound_windows_process() == base['resources']['process_memory_limit_bytes'], 'resource limit drift')
    resources = fidelity.resources(output, base['resources'])
    runtime = loader.runtime_lock()
    loader.require(sys.platform == runtime['platform'] and platform.machine() == runtime['machine'] and
                   platform.python_version() == runtime['python'], 'runtime mismatch')
    versions = loader.runtime_versions({p['name']: p['version'] for p in runtime['packages']})
    output.mkdir()
    report = dict(schema='rhythm-map.musicfm-negative.v1', complete=False, decision='not_completed',
                  protocol_sha256=LOCK_SHA, source_sha256=loader.digest(Path(__file__).read_bytes()),
                  previous_availability_report_sha256=plan['availability_report_sha256'],
                  previous_temporal_report_sha256=activity['previous_report_sha256'],
                  native_binary_sha256=loader.digest(Path(args.native_executable).read_bytes()),
                  resources=resources, runtime_versions=versions, cases=[],
                  pretrained_inference_started=False, extraction_fidelity_pass=False,
                  music_access=False, holdout_access=False, training=False, production_change=False,
                  new_user_parameters=False, independent_music_acceptance=False, commercial_distribution_approved=False)
    stage, start = 'activity_before_model_access', time.perf_counter()
    try:
        cases = inputs(plan)
        report['activity'] = prepare_activity(Path(args.native_executable), output, cases, plan, activity)
        with loader.no_network():
            import torch
            torch.set_num_threads(2)
            torch.manual_seed(142)
            report['authored_trace_controls'] = previous.authored_trace_controls(torch)
            stage = 'construction'
            model = loader.construct_meta(args.upstream, args.config, args.stats)
            schema = loader.tensor_schema(model)
            loader.require(loader.digest(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()) ==
                           base['expected_state_schema_sha256'] and
                           loader.digest(model.conformer.config.to_json_string().encode()) ==
                           base['expected_effective_config_sha256'], 'constructor differs')
            stage = 'verified_compatibility_load'
            checkpoint = loader.restricted_load(args.checkpoint, base['checkpoint'], torch.load)
            mapped, audit = compat.normalize(checkpoint, schema, compatibility, torch)
            loader.require(audit['unchanged_tensor_objects'] == compatibility['unchanged_non_counter_tensors'],
                           'state inventory drift')
            model.load_state_dict(mapped, strict=True, assign=True)
            loader.require(all(v.device.type == 'cpu' for v in model.state_dict().values()), 'unmaterialized state')
            del mapped, checkpoint
            model.eval()
            report['compatibility_audit'] = audit
            stage = 'authored_context_capture'
            report['pretrained_inference_started'] = True
            extracted = extract(model, torch, cases, output, report, old)
            report['extraction_fidelity_pass'] = True
            stage = 'frozen_negative_and_robustness_verdict'
            report['discrimination'] = judge(score_extracted(extracted, report['activity'], plan), old, plan)
        report.update(complete=True, unintended_python_network_attempts=0,
                      decision='authored_composition_survives_not_music_accuracy' if
                      report['discrimination']['rule_survives_authored'] else 'close_composed_rule_before_music')
    except Exception as error:
        report.update(decision='stopped_at_' + stage, failure_type=type(error).__name__)
        raise
    finally:
        report.update(elapsed_s=time.perf_counter() - start, peak_rss_bytes=peak_rss_bytes())
        with (output / 'report.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'config', 'stats', 'checkpoint', 'native-executable', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan, activity, old, base, compatibility = protocol()
    output = fidelity.private_output(args.output)
    env = fidelity.offline_environment(output)
    if args.worker:
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_XET',
                    'USE_TF', 'USE_FLAX', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'):
            loader.require(os.environ.get(key) == env[key], 'unsealed offline environment')
        worker(args, output, plan, activity, old, base, compatibility)
    else:
        subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                       env=env, check=True, timeout=plan['timeout_seconds'])


if __name__ == '__main__':
    main()
