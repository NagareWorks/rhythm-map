"""Bounded authored complete-input extraction and fixed temporal-rule gate."""
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
import musicfm_temporal as temporal

LOCK_SHA = 'eba21f993020759e11af8dab76af626e28f952edaecadf559941bf490e996772'
TEMPORAL_SHA = 'e924d58f2e69f963a4885fca1d8544acb00840022301238b49a9a79abb01b43f'


def protocol():
    raw = (HERE / 'musicfm-temporal-lock-v1.json').read_bytes()
    loader.require(loader.digest(raw) == LOCK_SHA, 'temporal protocol changed')
    plan = loader.strict_json(raw)
    for name, sha in dict(plan['helpers'], **{'musicfm_temporal.py': TEMPORAL_SHA}).items():
        loader.require(loader.digest((HERE / name).read_bytes()) == sha, 'temporal helper changed: ' + name)
    base, compatibility = fidelity.protocol(), compat.compatibility()
    loader.require(plan['base_fidelity_sha256'] == fidelity.LOCK_SHA and
                   plan['compatibility_sha256'] == compat.COMPAT_SHA, 'base protocol changed')
    return plan, base, compatibility


def captured_chunk(model, x, torch):
    """New explicit 8s context contract; the old <=2s fidelity helper is untouched."""
    loader.require(x.ndim == 2 and x.shape[0] == 1 and x.dtype == torch.float32 and x.device.type == 'cpu' and
                   1025 <= x.shape[1] <= temporal.CHUNK and bool(torch.isfinite(x).all()), 'invalid context PCM')
    count = temporal.tokens(x.shape[1])
    loader.require(len(model.conformer.layers) == 12 and
                   not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()), 'invalid hook state')
    calls, saved = [0] * 12, {}
    with ExitStack() as stack:
        def hook(index):
            def observe(_module, _args, value):
                calls[index] += 1
                if index == 6:
                    fidelity.finite_feature(value[0], count, torch)
                    saved['block7'] = value[0].detach().clone()
            return observe
        for i, block in enumerate(model.conformer.layers):
            stack.callback(block.register_forward_hook(hook(i)).remove)

        def encoder(_module, _args, value):
            loader.require('encoder' not in saved and type(value['hidden_states']) is tuple and
                           len(value['hidden_states']) == 13, 'invalid encoder trace')
            for hidden in value['hidden_states']:
                fidelity.finite_feature(hidden, count, torch)
            loader.require(torch.equal(value['hidden_states'][-1], value['last_hidden_state']), 'invalid last hidden')
            saved['encoder'] = value['hidden_states'][7].detach().clone()

        def projection(_module, _args, value):
            loader.require('projection' not in saved and tuple(value.shape) == (1, count, 4096) and
                           bool(torch.isfinite(value).all()), 'invalid projection trace')
            saved['projection'] = True
        stack.callback(model.conformer.register_forward_hook(encoder).remove)
        stack.callback(model.linear.register_forward_hook(projection).remove)
        result = fidelity.reference_pass(model, x, torch)
    fidelity.finite_feature(result, count, torch)
    loader.require(calls == [1] * 12 and set(saved) == {'block7', 'encoder', 'projection'} and
                   torch.equal(result, saved['block7']) and torch.equal(result, saved['encoder']), 'context tap differs')
    return result


def score_cases(cases):
    loader.require(tuple(cases) == temporal.CASES, 'incomplete authored population')
    scores, pairs = {}, []
    for name, (hidden, owner) in cases.items():
        scores[name] = {}
        for label, period in (('fast', temporal.Fraction(25, 4)), ('slow', temporal.Fraction(20))):
            scores[name][label] = temporal.evaluate(hidden[100:200], owner[100:200], 0,
                                                   temporal.Fraction(25, 2), period, 50)
    for name in temporal.CASES[:3]:
        for speed in ('fast', 'slow'):
            pairs.append(dict(constant=name, change='step-' + speed,
                              **temporal.verdict(scores[name][speed], scores['step-' + speed][speed])))
    silence = []
    for row in scores['silence'].values():
        # Unavailable evidence is safe abstention; eligible silence must not prefer a clock.
        silence.append(row['status'] != 'eligible' or
                       max(v['score'] for v in row['clocks'].values()) -
                       min(v['score'] for v in row['clocks'].values()) <= temporal.TIE)
    return dict(scores=scores, pairs=pairs, paired_denominator=6,
                shape_supported=sum(p['shape_supported'] for p in pairs),
                density_resolved=sum(p['density_resolved'] for p in pairs),
                silence_abstains=all(silence),
                rule_survives_authored=all(p['density_resolved'] and p['shape_supported'] for p in pairs) and all(silence))


def authored_trace_controls(torch):
    """Long-context trace success/failure fixture, without pretrained weights."""
    import types

    class Block(torch.nn.Module):
        def forward(self, x):
            return x + 1, None

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block() for _ in range(12)])
            self.embed_positions = types.SimpleNamespace(cached_sequence_length=None,
                                                        cached_rotary_positional_embedding=None)

        def forward(self, x):
            states = [x]
            for block in self.layers:
                x = block(x)[0]
                states.append(x)
            return dict(hidden_states=tuple(states), last_hidden_state=x)

    class Projection(torch.nn.Module):
        def forward(self, x):
            return x.repeat(1, 1, 4)

    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conformer, self.linear = Encoder(), Projection()
            self.mode = 'valid'

        def get_latent(self, x, layer_ix):
            value = self.conformer(torch.zeros((1, temporal.tokens(x.shape[1]), 1024)))
            if self.mode == 'exception':
                raise ValueError('authored interruption')
            if self.mode != 'no_projection':
                self.linear(value['last_hidden_state'])
            return value['hidden_states'][6 if self.mode == 'wrong_layer' else layer_ix]

    fixture = Fixture().eval()
    x = torch.zeros((1, temporal.CHUNK))
    loader.require(torch.equal(captured_chunk(fixture, x, torch), torch.full((1, 200, 1024), 7.0)),
                   'authored long-context tap failed')
    for mode in ('exception', 'no_projection', 'wrong_layer'):
        fixture.mode = mode
        try:
            captured_chunk(fixture, x, torch)
        except ValueError:
            pass
        else:
            raise ValueError('negative long-context fixture accepted: ' + mode)
        loader.require(not any(m._forward_hooks for m in fixture.modules()), 'authored hook leak')
    return dict(long_context_exact=True, negative_traces_refused=True, exception_cleanup=True, pretrained=False)


def evaluate_cases(model, torch, output, report):
    import numpy as np
    state = fidelity.state_hash(model, torch)
    metadata = json.dumps(model.stat, sort_keys=True) + model.conformer.config.to_json_string()
    cpu_rng, py_rng = torch.random.get_rng_state().clone(), random.getstate()
    report['initial_model_state_sha256'] = state
    cases = {}
    with ExitStack() as stack:
        def forbidden(*_args, **_kwargs):
            raise ValueError('training/quantizer entrypoint forbidden')
        for method in ('forward', 'get_targets', 'masking', 'tokenize'):
            stack.enter_context(patch.object(model, method, forbidden))
        for name in temporal.CASES:
            start = time.perf_counter()
            pcm = temporal.authored_pcm(name)
            pcm_hash = loader.digest(pcm.tobytes())
            plan, features, ledger = temporal.chunks(len(pcm)), [], []
            for row in plan:
                x = torch.from_numpy(pcm[row['sample_start']:row['sample_stop']]).unsqueeze(0)
                reference = fidelity.reference_pass(model, x, torch)
                captured = captured_chunk(model, x, torch)
                repeated = fidelity.reference_pass(model, x, torch)
                loader.require(torch.equal(reference, captured) and torch.equal(reference, repeated), 'context parity failed')
                features.append(reference[0].numpy().copy())
                ledger.append(dict(**row, feature_sha256=fidelity.tensor_hash(reference, torch),
                                   reference_hook_repeat_exact=True, all_blocks_and_projection=True))
            hidden, owner = temporal.stitch(features, plan, len(pcm))
            # Context changes are measured, never equated to numerical inference error.
            overlaps = []
            for i in range(1, len(plan)):
                left, right = plan[i]['token_start'], min(plan[i - 1]['token_start'] + len(features[i - 1]),
                                                        plan[i]['token_start'] + len(features[i]))
                a = features[i - 1][left - plan[i - 1]['token_start']:right - plan[i - 1]['token_start']]
                b = features[i][:right - left]
                overlaps.append(dict(contexts=[i - 1, i], tokens=right - left,
                                     max_abs_delta=float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
                                     exact=bool(np.array_equal(a, b))))
            loader.require(loader.digest(pcm.tobytes()) == pcm_hash and fidelity.state_hash(model, torch) == state and
                           metadata == json.dumps(model.stat, sort_keys=True) + model.conformer.config.to_json_string() and
                           all(not m.training for m in model.modules()) and
                           torch.equal(cpu_rng, torch.random.get_rng_state()) and random.getstate() == py_rng,
                           'state/input/mode/RNG changed')
            np.save(output / (name + '.features.npy'), hidden, allow_pickle=False)
            np.save(output / (name + '.owners.npy'), owner, allow_pickle=False)
            loader.require(np.array_equal(np.load(output / (name + '.features.npy'), allow_pickle=False), hidden) and
                           np.array_equal(np.load(output / (name + '.owners.npy'), allow_pickle=False), owner),
                           'private archive round-trip failed')
            cases[name] = hidden, owner
            report['cases'].append(dict(id=name, samples=len(pcm), tokens=len(hidden), pcm_sha256=pcm_hash,
                                        feature_sha256=loader.digest(hidden.tobytes()), owner_sha256=loader.digest(owner.tobytes()),
                                        contexts=ledger, overlaps=overlaps, state_and_input_unchanged=True,
                                        private_archive_roundtrip_exact=True, elapsed_s=time.perf_counter() - start))
            print(json.dumps(dict(case=name, complete=True)), flush=True)
    report['discrimination'] = score_cases(cases)
    report['final_model_state_sha256'] = fidelity.state_hash(model, torch)


def worker(args, output, plan, base, compatibility):
    from musicfm_loader_probe import bound_windows_process
    from prehead_capture import peak_rss_bytes
    loader.require(bound_windows_process() == base['resources']['process_memory_limit_bytes'], 'resource limit drift')
    resources = fidelity.resources(output, base['resources'])
    runtime = loader.runtime_lock()
    loader.require(sys.platform == runtime['platform'] and platform.machine() == runtime['machine'] and
                   platform.python_version() == runtime['python'], 'runtime mismatch')
    versions = loader.runtime_versions({p['name']: p['version'] for p in runtime['packages']})
    output.mkdir()
    report = dict(schema='rhythm-map.musicfm-temporal.v1', complete=False, decision='not_completed',
                  protocol_sha256=LOCK_SHA, temporal_source_sha256=TEMPORAL_SHA,
                  probe_source_sha256=loader.digest(Path(__file__).read_bytes()),
                  resources=resources, runtime_versions=versions, cases=[],
                  music_access=False, holdout_access=False, training=False, production_change=False)
    stage, start = 'construction', time.perf_counter()
    try:
        with loader.no_network():
            import torch
            torch.set_num_threads(2)
            torch.manual_seed(142)
            report['authored_trace_controls'] = authored_trace_controls(torch)
            model = loader.construct_meta(args.upstream, args.config, args.stats)
            schema = loader.tensor_schema(model)
            loader.require(loader.digest(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()) ==
                           base['expected_state_schema_sha256'] and
                           loader.digest(model.conformer.config.to_json_string().encode()) ==
                           base['expected_effective_config_sha256'], 'constructor differs')
            stage = 'verified_compatibility_load'
            checkpoint = loader.restricted_load(args.checkpoint, base['checkpoint'], torch.load)
            mapped, audit = compat.normalize(checkpoint, schema, compatibility, torch)
            loader.require(audit['unchanged_tensor_objects'] == compatibility['unchanged_non_counter_tensors'], 'state inventory drift')
            model.load_state_dict(mapped, strict=True, assign=True)
            loader.require(all(v.device.type == 'cpu' for v in model.state_dict().values()), 'unmaterialized state')
            del mapped, checkpoint
            model.eval()
            report['compatibility_audit'] = audit
            stage = 'authored_context_capture'
            evaluate_cases(model, torch, output, report)
        report.update(complete=True, extraction_fidelity_pass=True, unintended_python_network_attempts=0,
                      decision='authored_rule_survives_not_music_accuracy' if report['discrimination']['rule_survives_authored']
                      else 'close_fixed_rule_before_music')
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
    for name in ('upstream', 'config', 'stats', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan, base, compatibility = protocol()
    output = fidelity.private_output(args.output)
    env = fidelity.offline_environment(output)
    if args.worker:
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_XET',
                    'USE_TF', 'USE_FLAX', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'):
            loader.require(os.environ.get(key) == env[key], 'unsealed offline environment')
        worker(args, output, plan, base, compatibility)
    else:
        subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                       env=env, check=True, timeout=plan['timeout_seconds'])


if __name__ == '__main__':
    main()
