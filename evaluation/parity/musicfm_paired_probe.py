"""One offline conditional MusicFM replay; no new selector, training or holdout."""
import argparse
from contextlib import ExitStack
import json
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
import musicfm_fidelity as fidelity
import musicfm_fidelity_compat as compat
import musicfm_loader as loader
import musicfm_paired as paired
import musicfm_temporal as temporal
import musicfm_temporal_probe as extraction

LOCK = HERE / 'musicfm-paired-lock-v1.json'


def protocol():
    plan = loader.strict_json(LOCK.read_bytes())
    loader.require(plan['schema'] == 'rhythm-map.musicfm-paired-lock.v1' and
                   plan['conditional_only'] is True and plan['training'] is False and
                   plan['holdout_access'] is False and plan['production_change'] is False,
                   'paired scope changed')
    for name, sha in plan['helpers'].items():
        loader.require(loader.digest((HERE / name).read_bytes()) == sha, 'paired helper changed: ' + name)
    _, base, compatibility = extraction.protocol()
    _, _, _, _, cases, selected = paired.prior.verified_plan()
    loader.require([p['id'] for p in selected] == plan['selected_ids'], 'selected population changed')
    return plan, base, compatibility, cases, selected


def read_pcm(directory, selected, prior_report):
    """Only byte-verified complete PCM already captured by the earlier audit."""
    import numpy as np
    result, sources = {}, []
    for pair in selected:
        name = pair['id']
        path = directory / (name + '.trace.private.json')
        raw = path.read_bytes()
        expected = prior_report['trace_sha256'][name]
        loader.require(loader.digest(raw) == expected, 'cached complete trace changed')
        trace = loader.strict_json(raw)
        loader.require(trace['case_id'] == name and trace['sample_rate'] == 22050 and
                       trace['suite_id'] == 'artbeat-v1' and trace['prefix_seconds'] == 60 and
                       trace['decoded_sample_count'] == len(trace['mono_samples']) < 60 * 22050,
                       'not a complete registered input')
        x = np.asarray(trace['mono_samples'], np.float32)
        loader.require(x.ndim == 1 and len(x) > 1025 and np.isfinite(x).all(), 'invalid cached PCM')
        result[name] = x
        sources.append(dict(id=name, trace_sha256=expected, original_audio_sha256=trace['audio_sha256'],
                            input_samples=len(x), input_sample_rate=22050, input_pcm_sha256=loader.digest(x.tobytes())))
    return result, sources


def resample(x, torch):
    """One explicit bridge from locked 22.05 kHz PCM, not a frontend ablation."""
    import torchaudio.functional as audio
    loader.require(x.ndim == 1 and x.dtype == torch.float32 and x.device.type == 'cpu' and
                   bool(torch.isfinite(x).all()), 'invalid resampling input')
    y = audio.resample(x, 22050, 24000, lowpass_filter_width=6, rolloff=0.99,
                       resampling_method='sinc_interp_hann')
    loader.require(y.dtype == torch.float32 and len(y) == (len(x) * 160 + 146) // 147 and
                   bool(torch.isfinite(y).all()), 'resampling length/type/finiteness mismatch')
    return y


def resampler_controls(torch):
    rows = []
    for count in (1025, 22050, 22051):
        zero = torch.zeros(count)
        loader.require(torch.count_nonzero(resample(zero, torch)) == 0, 'silence changed')
        x = zero.clone()
        at = count // 2
        x[at] = 0.5
        a, b = resample(x, torch), resample(x, torch)
        peak = int(torch.argmax(torch.abs(a)))
        loader.require(torch.equal(a, b) and abs(peak - at * 160 / 147) <= 1,
                       'resampler repeat/origin control failed')
        loader.require(torch.equal(x, torch.nn.functional.one_hot(torch.tensor(at), count).float() * 0.5),
                       'resampler mutated input')
        rows.append(dict(samples=count, output_samples=len(a), repeat_exact=True,
                         peak_error_samples=abs(peak - at * 160 / 147), silence_exact=True))
    return rows


def capture_cases(model, torch, pcm, output, report):
    import numpy as np
    initial = fidelity.state_hash(model, torch)
    expected = loader.strict_json((HERE / 'musicfm-negative-v1.json').read_bytes())['final_model_state_sha256']
    loader.require(initial == expected, 'model differs from retained negative experiment')
    report['initial_model_state_sha256'] = initial
    metadata = json.dumps(model.stat, sort_keys=True) + model.conformer.config.to_json_string()
    cpu_rng, py_rng = torch.random.get_rng_state().clone(), random.getstate()
    captured = {}
    with ExitStack() as stack:
        def forbidden(*_args, **_kwargs):
            raise ValueError('training/quantizer entrypoint forbidden')
        for method in ('forward', 'get_targets', 'masking', 'tokenize'):
            stack.enter_context(patch.object(model, method, forbidden))
        for name, original in pcm.items():
            started = time.perf_counter()
            original_sha = loader.digest(original.tobytes())
            x = resample(torch.from_numpy(original), torch)
            input_sha = fidelity.tensor_hash(x, torch)
            plan = temporal.chunks(len(x))
            features, ledger = [], []
            for row in plan:
                chunk = x[row['sample_start']:row['sample_stop']].unsqueeze(0)
                reference = fidelity.reference_pass(model, chunk, torch)
                observed = extraction.captured_chunk(model, chunk, torch)
                repeated = fidelity.reference_pass(model, chunk, torch)
                loader.require(torch.equal(reference, observed) and torch.equal(reference, repeated),
                               'context reference/hook/repeat differs')
                features.append(reference[0].numpy().copy())
                ledger.append(dict(**row, feature_sha256=fidelity.tensor_hash(reference, torch),
                                   reference_hook_repeat_exact=True, all_blocks_and_projection=True))
                print(json.dumps(dict(case=name, context=row['id'], contexts=len(plan), complete=True)), flush=True)
            hidden, owner = temporal.stitch(features, plan, len(x))
            loader.require(loader.digest(original.tobytes()) == original_sha and
                           fidelity.tensor_hash(x, torch) == input_sha and fidelity.state_hash(model, torch) == initial and
                           metadata == json.dumps(model.stat, sort_keys=True) + model.conformer.config.to_json_string() and
                           all(not m.training for m in model.modules()) and
                           torch.equal(cpu_rng, torch.random.get_rng_state()) and random.getstate() == py_rng,
                           'input/state/mode/RNG changed')
            for suffix, values in (('features', hidden), ('owners', owner)):
                path = output / (name + '.' + suffix + '.npy')
                np.save(path, values, allow_pickle=False)
                loader.require(np.array_equal(np.load(path, allow_pickle=False), values), 'archive roundtrip failed')
            captured[name] = hidden, owner
            report['captures'].append(dict(id=name, samples=len(x), sample_rate=24000, tokens=len(hidden),
                pcm_sha256=input_sha, feature_sha256=loader.digest(hidden.tobytes()),
                owner_sha256=loader.digest(owner.tobytes()), contexts=ledger, state_and_input_unchanged=True,
                private_archive_roundtrip_exact=True, elapsed_s=time.perf_counter() - started))
    report['final_model_state_sha256'] = fidelity.state_hash(model, torch)
    return captured


def worker(args, output, plan, base, compatibility, cases, selected):
    from musicfm_loader_probe import bound_windows_process
    from prehead_capture import peak_rss_bytes
    loader.require(bound_windows_process() == base['resources']['process_memory_limit_bytes'], 'resource limit drift')
    resources = fidelity.resources(output, base['resources'])
    runtime = loader.runtime_lock()
    loader.require(sys.platform == runtime['platform'] and platform.machine() == runtime['machine'] and
                   platform.python_version() == runtime['python'], 'runtime mismatch')
    versions = loader.runtime_versions({p['name']: p['version'] for p in runtime['packages']})
    output.mkdir()
    report = dict(schema='rhythm-map.musicfm-paired.v1', complete=False, decision='not_completed',
                  protocol_sha256=loader.digest(LOCK.read_bytes()), helper_sha256=plan['helpers'],
                  resources=resources, runtime_versions=versions, captures=[],
                  conditional_only=True, music_access=True, holdout_access=False, training=False,
                  production_change=False, checkpoint_sha256=base['checkpoint']['sha256'],
                  known_no_rhythm_failure_report_sha256=plan['helpers']['musicfm-negative-v1.json'])
    stage, started = 'verified_cached_inputs', time.perf_counter()
    try:
        old = loader.strict_json((HERE / 'prehead-recurrence-v1.json').read_bytes())
        pcm, report['input_sources'] = read_pcm(Path(args.trace_dir), selected, old)
        with loader.no_network():
            import torch
            torch.set_num_threads(2)
            torch.manual_seed(142)
            stage = 'authored_instrumentation'
            report['resampler_controls'] = resampler_controls(torch)
            report['authored_trace_controls'] = extraction.authored_trace_controls(torch)
            stage = 'construction_and_verified_load'
            model = loader.construct_meta(args.upstream, args.config, args.stats)
            schema = loader.tensor_schema(model)
            loader.require(loader.digest(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()) ==
                           base['expected_state_schema_sha256'] and
                           loader.digest(model.conformer.config.to_json_string().encode()) ==
                           base['expected_effective_config_sha256'], 'constructor differs')
            checkpoint = loader.restricted_load(args.checkpoint, base['checkpoint'], torch.load)
            mapped, report['compatibility_audit'] = compat.normalize(checkpoint, schema, compatibility, torch)
            model.load_state_dict(mapped, strict=True, assign=True)
            loader.require(all(v.device.type == 'cpu' for v in model.state_dict().values()), 'unmaterialized state')
            del mapped, checkpoint
            model.eval()
            stage = 'complete_context_capture'
            captured = capture_cases(model, torch, pcm, output, report)
            stage = 'conditional_paired_comparison'
            by_id = {c['identity']['id']: c for c in cases}
            report['pairs'] = [dict(id=p['id'], pair_sha256=paired.oracle.exposure.canonical_hash(p),
                **paired.pair_result(*captured[p['id']], by_id[p['id']]['data'], p)) for p in selected]
            report['summary'] = paired.summarize(report['pairs'], plan['selected_ids'])
            report['input_cases'] = [dict(c['identity'], cohort=c['cohort'],
                disposition='selected_pair' if c['identity']['id'] in captured else
                'expressive_untyped' if c['data']['untyped'] else 'no_registered_pair') for c in cases]
        report.update(complete=True, extraction_fidelity_pass=True, unintended_python_network_attempts=0,
                      decision='conditional_diagnostic_only_no_product_admission')
    except Exception as error:
        report.update(decision='stopped_at_' + stage, failure_type=type(error).__name__)
        raise
    finally:
        report.update(elapsed_s=time.perf_counter() - started, peak_rss_bytes=peak_rss_bytes())
        with (output / 'report.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'config', 'stats', 'checkpoint', 'trace-dir', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan, base, compatibility, cases, selected = protocol()
    output = fidelity.private_output(args.output)
    env = fidelity.offline_environment(output)
    if args.worker:
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_XET',
                    'USE_TF', 'USE_FLAX', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'):
            loader.require(os.environ.get(key) == env[key], 'unsealed offline environment')
        worker(args, output, plan, base, compatibility, cases, selected)
    else:
        subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                       env=env, check=True, timeout=plan['timeout_seconds'])


if __name__ == '__main__':
    main()
