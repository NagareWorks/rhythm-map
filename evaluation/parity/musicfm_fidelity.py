"""Bounded Windows FMA fidelity probe, authored PCM only; never a music CLI."""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import musicfm_loader as loader

LOCK_SHA = 'e7ad8b06df669a709fd2da330fe230d05551f5405f4591dad865c288db4fde01'


def protocol():
    raw = (HERE / 'musicfm-fidelity-lock-v1.json').read_bytes()
    loader.require(loader.digest(raw) == LOCK_SHA, 'fidelity protocol changed')
    plan = loader.strict_json(raw)
    for name, sha in plan['helpers'].items():
        loader.require(loader.digest((HERE / name).read_bytes()) == sha, 'helper changed: ' + name)
    loader.runtime_lock()
    return plan


def private_output(path):
    output = Path(path).resolve()
    loader.require(sys.platform == 'win32' and output.drive.lower() != os.environ.get('SystemDrive', 'C:').lower(),
                   'private output must be off the Windows system drive')
    loader.require(not output.is_relative_to(HERE.parents[1]) and not output.exists() and output.parent.is_dir(),
                   'output must be fresh, outside Git, with an existing parent')
    return output


def offline_environment(output):
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_XET='1', USE_TF='0', USE_FLAX='0',
               HF_HOME=str(output / 'hf-cache'), HF_HUB_CACHE=str(output / 'hf-cache' / 'hub'),
               TRANSFORMERS_CACHE=str(output / 'hf-cache' / 'hub'), PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    return env


def resources(output, limits):
    import ctypes
    from ctypes import wintypes

    class Memory(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
            (k, ctypes.c_uint64) for k in ('total_phys', 'avail_phys', 'total_pagefile',
                                         'avail_pagefile', 'total_virtual', 'avail_virtual', 'avail_extended')]

    value = Memory()
    value.length = ctypes.sizeof(value)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = (ctypes.POINTER(Memory),)
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    loader.require(kernel.GlobalMemoryStatusEx(ctypes.byref(value)), 'memory preflight unavailable')
    observed = dict(available_physical_bytes=value.avail_phys, available_commit_bytes=value.avail_pagefile,
                    free_disk_bytes=shutil.disk_usage(output.parent).free)
    validate_resources(observed, limits)
    return observed


def validate_resources(observed, limits):
    for key, minimum in [('available_physical_bytes', 'min_available_physical_bytes'),
                         ('available_commit_bytes', 'min_available_commit_bytes'),
                         ('free_disk_bytes', 'min_free_disk_bytes')]:
        loader.require(type(observed.get(key)) is int and observed[key] >= limits[minimum],
                       'insufficient resource: ' + key)


def pcm(case, torch):
    loader.token_geometry(case['samples'])
    loader.require(case['kind'] in ('silence', 'impulse', 'tone'), 'unknown authored PCM recipe')
    x = torch.zeros((1, case['samples']), dtype=torch.float32)
    if case['kind'] == 'impulse':
        x[0, case['samples'] // 2] = 0.5
    elif case['kind'] == 'tone':
        x[0] = 0.2 * torch.sin(torch.arange(case['samples'], dtype=torch.float32) * (2 * torch.pi * 440 / 24000))
    return x


def tensor_hash(tensor, torch):
    loader.require(tensor.device.type == 'cpu' and tensor.layout == torch.strided, 'non-CPU/dense state')
    # Byte views avoid copying the entire checkpoint into a second snapshot.
    data = memoryview(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()).cast('B')
    digest = hashlib.sha256()
    for offset in range(0, len(data), 1024 * 1024):
        digest.update(data[offset:offset + 1024 * 1024])
    return digest.hexdigest()


def state_hash(model, torch):
    rows = {}
    for kind, items in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]:
        for name, value in items:
            rows[kind + ':' + name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                          sha256=tensor_hash(value, torch))
    return loader.digest(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode())


@contextmanager
def preserve_transients(model, torch):
    positional = model.conformer.embed_positions
    previous_length = positional.cached_sequence_length
    previous_cache = positional.cached_rotary_positional_embedding
    cache_copy = previous_cache.clone() if previous_cache is not None else None
    python_rng = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        unchanged = previous_cache is None or torch.equal(previous_cache, cache_copy)
        positional.cached_sequence_length = previous_length
        positional.cached_rotary_positional_embedding = previous_cache
        random.setstate(python_rng)
        loader.require(unchanged, 'existing rotary cache mutated in place')


def finite_feature(value, tokens, torch):
    loader.require(type(value) is torch.Tensor and value.dtype == torch.float32 and value.device.type == 'cpu' and
                   tuple(value.shape) == (1, tokens, 1024) and bool(torch.isfinite(value).all()),
                   'feature type/shape/finiteness mismatch')


def captured_pass(model, x, torch):
    """Observe the existing get_latent entrypoint, without replacing its math."""
    tokens = loader.token_geometry(x.shape[1])['tokens']
    captured, calls = {}, [0] * 12
    loader.require(len(model.conformer.layers) == 12, 'wrong block count')
    loader.require(not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()),
                   'pre-existing hooks are not supported')
    with ExitStack() as stack:
        def block_hook(index):
            def record(_module, _args, output):
                calls[index] += 1
                if index == 6:
                    finite_feature(output[0], tokens, torch)
                    captured['block7'] = output[0].detach().clone()
            return record

        for index, block in enumerate(model.conformer.layers):
            stack.callback(block.register_forward_hook(block_hook(index)).remove)

        def encoder_hook(_module, _args, output):
            loader.require('encoder' not in captured, 'encoder executed more than once')
            hidden = output['hidden_states']
            loader.require(type(hidden) is tuple and len(hidden) == 13, 'hidden tuple mismatch')
            for value in hidden:
                finite_feature(value, tokens, torch)
            loader.require(torch.equal(hidden[-1], output['last_hidden_state']), 'last hidden tuple mismatch')
            captured['encoder'] = hidden[7].detach().clone()

        def projection_hook(_module, _args, output):
            loader.require('projection' not in captured and tuple(output.shape) == (1, tokens, 4096) and
                           bool(torch.isfinite(output).all()), 'projection call/shape/finiteness mismatch')
            captured['projection'] = True

        stack.callback(model.conformer.register_forward_hook(encoder_hook).remove)
        stack.callback(model.linear.register_forward_hook(projection_hook).remove)
        with preserve_transients(model, torch), torch.inference_mode():
            result = model.get_latent(x, layer_ix=7).detach().clone()
    loader.require(calls == [1] * 12 and set(captured) == {'block7', 'encoder', 'projection'},
                   'incomplete execution trace')
    finite_feature(result, tokens, torch)
    loader.require(torch.equal(result, captured['block7']) and torch.equal(result, captured['encoder']),
                   'layer-7 tap differs from reference tuple')
    return result


def reference_pass(model, x, torch):
    with preserve_transients(model, torch), torch.inference_mode():
        return model.get_latent(x, layer_ix=7).detach().clone()


def authored_hook_controls(torch):
    """Small real-Torch trace fixture; NOT the MusicFM architecture or weights."""
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
            hidden = [x]
            for block in self.layers:
                x = block(x)[0]
                hidden.append(x)
            return dict(hidden_states=tuple(hidden), last_hidden_state=x)

    class Projection(torch.nn.Module):
        def forward(self, x):
            return x.repeat(1, 1, 4)

    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conformer, self.linear = Encoder(), Projection()
            self.fail = False
            self.wrong_layer = False
            self.skip_projection = False
            self.register_buffer('authored_scalar', torch.tensor(1, dtype=torch.int64))

        def get_latent(self, x, layer_ix):
            value = self.conformer(torch.zeros((1, loader.token_geometry(x.shape[1])['tokens'], 1024)))
            if self.fail:
                raise RuntimeError('authored interruption')
            if not self.skip_projection:
                self.linear(value['last_hidden_state'])
            return value['hidden_states'][6 if self.wrong_layer else layer_ix]

    model = Fixture().eval()
    x = torch.zeros((1, 1200))
    expected = torch.full((1, 2, 1024), 7.0)
    loader.require(torch.equal(captured_pass(model, x, torch), expected), 'authored tap mismatch')
    model.fail = True
    try:
        captured_pass(model, x, torch)
    except RuntimeError as error:
        loader.require(str(error) == 'authored interruption', 'unexpected authored error')
    else:
        raise ValueError('authored exception was swallowed')
    loader.require(not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()),
                   'hooks leaked after failure')
    model.fail = False
    for attribute in ('wrong_layer', 'skip_projection'):
        setattr(model, attribute, True)
        try:
            captured_pass(model, x, torch)
        except ValueError:
            pass
        else:
            raise ValueError('authored negative trace accepted: ' + attribute)
        setattr(model, attribute, False)
        loader.require(not any(m._forward_hooks for m in model.modules()), 'negative trace leaked hooks')
    before = state_hash(model, torch)
    model.authored_scalar.add_(1)
    loader.require(state_hash(model, torch) != before, 'scalar buffer mutation not detected')
    py_rng, cpu_rng = random.getstate(), torch.random.get_rng_state().clone()
    positions = model.conformer.embed_positions
    with preserve_transients(model, torch):
        positions.cached_sequence_length = 2
        positions.cached_rotary_positional_embedding = torch.ones(2)
        random.seed(42)
        torch.manual_seed(42)
    loader.require(positions.cached_sequence_length is None and positions.cached_rotary_positional_embedding is None and
                   random.getstate() == py_rng and torch.equal(torch.random.get_rng_state(), cpu_rng),
                   'authored cache/RNG restoration failed')
    return dict(exact_layer7=True, all_blocks_and_projection=True, exception_cleanup=True,
                wrong_layer_and_missing_projection_rejected=True, scalar_state_mutation_detected=True,
                transient_cache_and_rng_restored=True, pretrained=False)


def evaluate_cases(model, plan, torch, report):
    from unittest.mock import patch
    initial = state_hash(model, torch)
    metadata = json.dumps(model.stat, sort_keys=True) + model.conformer.config.to_json_string()
    report['initial_model_state_sha256'] = initial
    cpu_rng, py_rng = torch.random.get_rng_state().clone(), random.getstate()
    with ExitStack() as stack:
        def forbidden(*_args, **_kwargs):
            raise ValueError('training/quantizer entrypoint forbidden')
        for name in ('forward', 'get_targets', 'masking', 'tokenize'):
            stack.enter_context(patch.object(model, name, forbidden))
        for case in plan['cases']:
            start = time.perf_counter()
            x = pcm(case, torch)
            before = tensor_hash(x, torch)
            reference = reference_pass(model, x, torch)
            captured = captured_pass(model, x, torch)
            repeated = reference_pass(model, x, torch)
            finite_feature(reference, loader.token_geometry(case['samples'])['tokens'], torch)
            loader.require(torch.equal(reference, captured) and torch.equal(reference, repeated),
                           'reference/capture/repeat are not bit-exact')
            loader.require(tensor_hash(x, torch) == before, 'authored PCM mutated')
            loader.require(state_hash(model, torch) == initial and
                           metadata == json.dumps(model.stat, sort_keys=True) + model.conformer.config.to_json_string(),
                           'model state changed')
            loader.require(all(not m.training for m in model.modules()) and
                           torch.equal(cpu_rng, torch.random.get_rng_state()) and random.getstate() == py_rng,
                           'mode/RNG changed')
            report['cases'].append(dict(**case, **loader.token_geometry(case['samples']),
                                        pcm_sha256=before, feature_sha256=tensor_hash(reference, torch),
                                        reference_hook_repeat_exact=True, all_twelve_blocks_executed=True,
                                        projection_executed=True, state_and_input_unchanged=True,
                                        elapsed_s=time.perf_counter() - start))
            print(json.dumps(dict(case=case['id'], complete=True)), flush=True)
    report['final_model_state_sha256'] = state_hash(model, torch)


def worker(args, output, plan):
    from musicfm_loader_probe import bound_windows_process
    limit = bound_windows_process()
    loader.require(limit == plan['resources']['process_memory_limit_bytes'], 'resource envelope changed')
    observed = resources(output, plan['resources'])
    lock = loader.runtime_lock()
    loader.require(sys.platform == lock['platform'] and platform.machine() == lock['machine'] and
                   platform.python_version() == lock['python'], 'runtime platform/Python mismatch')
    versions = loader.runtime_versions({p['name']: p['version'] for p in lock['packages']})
    output.mkdir()
    report = dict(schema='rhythm-map.musicfm-fidelity.v1', complete=False, decision='not_completed',
                  protocol_sha256=LOCK_SHA, source_sha256=loader.digest(Path(__file__).read_bytes()),
                  helper_sha256=plan['helpers'], resources=observed, resource_limits=plan['resources'],
                  runtime_versions=versions, checkpoint_bytes_verified=False, checkpoint_loaded=False,
                  pretrained_inference_started=False, music_access=False, holdout_access=False,
                  training=False, production_change=False, commercial_distribution_approved=False, cases=[])
    stage, start = 'identity', time.perf_counter()
    try:
        with loader.no_network():
            import torch
            torch.set_num_threads(plan['resources']['torch_threads'])
            torch.manual_seed(142)
            stage = 'authored_hook_controls'
            report['authored_hook_controls'] = authored_hook_controls(torch)
            stage = 'checkpoint_identity'
            with loader.verified_file(args.checkpoint, plan['checkpoint']['size_bytes'], plan['checkpoint']['sha256']):
                pass
            report['checkpoint_bytes_verified'] = True
            stage = 'construct_meta'
            model = loader.construct_meta(args.upstream, args.config, args.stats)
            schema = loader.tensor_schema(model)
            schema_sha = loader.digest(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode())
            loader.require(schema_sha == plan['expected_state_schema_sha256'] and
                           loader.digest(model.conformer.config.to_json_string().encode()) ==
                           plan['expected_effective_config_sha256'], 'constructor identity differs')
            report['state_schema_sha256'] = schema_sha
            stage = 'restricted_checkpoint_load'
            load_start = time.perf_counter()
            loader.load_fma(model, args.checkpoint)
            report.update(checkpoint_loaded=True, checkpoint_load_s=time.perf_counter() - load_start)
            stage = 'authored_pretrained_fidelity'
            report['pretrained_inference_started'] = True
            evaluate_cases(model, plan, torch, report)
        report.update(complete=True, decision='authored_fidelity_pass_not_musical_accuracy',
                      unintended_python_network_attempts=0)
    except Exception as error:
        report.update(decision='stopped_at_' + stage, failure_type=type(error).__name__)
        raise
    finally:
        from prehead_capture import peak_rss_bytes
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
    plan = protocol()
    output = private_output(args.output)
    env = offline_environment(output)
    if args.worker:
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_XET',
                    'USE_TF', 'USE_FLAX', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'):
            loader.require(os.environ.get(key) == env[key], 'unsealed offline environment')
        worker(args, output, plan)
    else:
        subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                       env=env, check=True, timeout=plan['resources']['timeout_seconds'])


if __name__ == '__main__':
    main()
