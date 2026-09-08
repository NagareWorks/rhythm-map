"""Explicit FMA serialization-compatibility gate; preserves the strict refusal."""
import argparse
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import musicfm_fidelity as fidelity
import musicfm_loader as loader

COMPAT_SHA = '1936f0360cb58fc84bd1497835097b25f077a4f778436b3361c6ba93bda15222'


def compatibility():
    raw = (HERE / 'musicfm-compatibility-lock-v1.json').read_bytes()
    loader.require(loader.digest(raw) == COMPAT_SHA, 'compatibility protocol changed')
    plan = loader.strict_json(raw)
    loader.require(plan['base_protocol_sha256'] == fidelity.LOCK_SHA and
                   loader.digest(Path(fidelity.__file__).read_bytes()) == plan['fidelity_source_sha256'],
                   'original fidelity implementation changed')
    return plan


def counter_integer(value):
    loader.require(type(value) is float and math.isfinite(value) and 0 <= value < 2 ** 63 and value.is_integer(),
                   'counter must be a finite nonnegative int64-representable integer')
    return int(value)


def normalize(checkpoint, expected, plan, torch):
    """Only two named aliases and eighteen named scalar counters can change."""
    loader.require(type(checkpoint) in (dict, OrderedDict) and
                   type(checkpoint.get('state_dict')) in (dict, OrderedDict), 'unknown checkpoint root')
    original = checkpoint['state_dict']
    state, renamed, converted, unchanged = {}, [], [], 0
    for key, value in original.items():
        loader.require(type(key) is str and key.startswith('model.'), 'unknown state prefix')
        name = key[6:]
        target = plan['aliases'].get(name, name)
        loader.require(target not in state and target in expected, 'alias collision or unexpected key')
        if name in plan['aliases']:
            loader.require('model.' + target not in original, 'old and new alias both present')
            renamed.append(name)
        if name in plan['counter_keys']:
            loader.require(type(value) is torch.Tensor and value.device.type == 'cpu' and
                           value.dtype == torch.float32 and tuple(value.shape) == () and
                           value.layout == torch.strided and not value.is_quantized, 'counter schema mismatch')
            count = counter_integer(value.item())
            converted_value = value.to(torch.int64)
            loader.require(converted_value.item() == count and torch.equal(converted_value.to(torch.float32), value),
                           'counter conversion is not exact')
            state[target] = converted_value
            converted.append(name)
        else:
            state[target] = value  # Preserve tensor identity; no learned-value cast/copy/reshape.
            unchanged += 1
    loader.require(set(renamed) == set(plan['aliases']) and set(converted) == set(plan['counter_keys']),
                   'incomplete explicit compatibility inventory')
    mapped = loader.checked_state({'state_dict': {'model.' + k: v for k, v in state.items()}}, expected, torch)
    return mapped, dict(renamed=sorted(renamed), converted_counters=sorted(converted),
                        unchanged_tensor_objects=unchanged, learned_tensor_casts=False, learned_tensor_reshapes=False)


def authored_compatibility_controls(torch, plan):
    # The official legacy->modern weight_norm bridge, exercised without model weights.
    legacy = torch.nn.utils.weight_norm(torch.nn.Conv1d(4, 8, 3, groups=2), dim=2).eval()
    modern = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(4, 8, 3, groups=2), dim=2).eval()
    legacy_state = legacy.state_dict()
    modern.load_state_dict(legacy_state, strict=True)  # Official PyTorch compatibility hook.
    x = torch.arange(36, dtype=torch.float32).reshape(1, 4, 9) / 32
    with torch.inference_mode():
        loader.require(torch.equal(legacy(x), modern(x)) and torch.equal(legacy.weight, modern.weight),
                       'legacy/modern weight_norm equivalence failed')
    bn = torch.nn.BatchNorm1d(4).eval()
    with torch.inference_mode():
        reference = bn(x)
        bn.num_batches_tracked = torch.tensor(17.0)
        before = bn(x)
        bn.num_batches_tracked = bn.num_batches_tracked.to(torch.int64)
        loader.require(torch.equal(reference, before) and torch.equal(reference, bn(x)), 'eval BatchNorm changed')

    source = {'model.' + k: torch.tensor(17.0) for k in plan['counter_keys']}
    source.update({'model.' + k: torch.ones((2, 3)) for k in plan['aliases']})
    source['model.authored_weight'] = torch.tensor([0.25, 0.5])
    expected = {plan['aliases'].get(k[6:], k[6:]): dict(shape=list(v.shape),
                dtype='torch.int64' if k[6:] in plan['counter_keys'] else str(v.dtype)) for k, v in source.items()}
    mapped, audit = normalize({'state_dict': source}, expected, plan, torch)
    loader.require(mapped['authored_weight'] is source['model.authored_weight'] and
                   len(audit['renamed']) == 2 and len(audit['converted_counters']) == 18, 'authored migration failed')
    for value in (float('nan'), float('inf'), -1.0, 0.5, float(2 ** 63)):
        try:
            counter_integer(value)
        except ValueError:
            pass
        else:
            raise ValueError('invalid authored counter accepted')
    for corrupt in ('collision', 'missing_counter', 'missing_alias', 'wrong_counter_dtype'):
        altered = dict(source)
        if corrupt == 'collision':
            altered['model.' + next(iter(plan['aliases'].values()))] = torch.ones((2, 3))
        elif corrupt == 'missing_counter':
            altered.pop('model.' + plan['counter_keys'][0])
        elif corrupt == 'missing_alias':
            altered.pop('model.' + next(iter(plan['aliases'])))
        else:
            altered['model.' + plan['counter_keys'][0]] = torch.tensor(17, dtype=torch.int64)
        try:
            normalize({'state_dict': altered}, expected, plan, torch)
        except ValueError:
            pass
        else:
            raise ValueError('invalid authored migration accepted: ' + corrupt)
    return dict(weight_norm_weights_and_outputs_exact=True, batchnorm_eval_exact=True,
                explicit_alias_counter_controls=True, invalid_counter_and_inventory_refused=True, pretrained=False)


def worker(args, output, base, compat):
    from musicfm_loader_probe import bound_windows_process
    from prehead_capture import peak_rss_bytes
    loader.require(bound_windows_process() == base['resources']['process_memory_limit_bytes'], 'limit drift')
    observed = fidelity.resources(output, base['resources'])
    lock = loader.runtime_lock()
    loader.require(sys.platform == lock['platform'] and platform.machine() == lock['machine'] and
                   platform.python_version() == lock['python'], 'runtime mismatch')
    versions = loader.runtime_versions({p['name']: p['version'] for p in lock['packages']})
    output.mkdir()
    report = dict(schema='rhythm-map.musicfm-fidelity-compat.v1', complete=False, decision='not_completed',
                  protocol_sha256=fidelity.LOCK_SHA, compatibility_sha256=COMPAT_SHA,
                  source_sha256=loader.digest(Path(__file__).read_bytes()),
                  original_fidelity_source_sha256=compat['fidelity_source_sha256'],
                  resources=observed, resource_limits=base['resources'], runtime_versions=versions,
                  checkpoint_bytes_verified=False, checkpoint_loaded=False, pretrained_inference_started=False,
                  music_access=False, holdout_access=False, training=False, production_change=False,
                  commercial_distribution_approved=False, cases=[])
    stage, start = 'authored_compatibility', time.perf_counter()
    try:
        with loader.no_network():
            import torch
            torch.set_num_threads(2)
            torch.manual_seed(142)
            report['authored_compatibility_controls'] = authored_compatibility_controls(torch, compat)
            stage = 'constructor'
            model = loader.construct_meta(args.upstream, args.config, args.stats)
            expected = loader.tensor_schema(model)
            loader.require(loader.digest(json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()) ==
                           base['expected_state_schema_sha256'] and
                           loader.digest(model.conformer.config.to_json_string().encode()) ==
                           base['expected_effective_config_sha256'], 'constructor differs')
            stage = 'restricted_normalization'
            checkpoint = loader.restricted_load(args.checkpoint, base['checkpoint'], torch.load)
            report['checkpoint_bytes_verified'] = True
            mapped, audit = normalize(checkpoint, expected, compat, torch)
            loader.require(audit['unchanged_tensor_objects'] == compat['unchanged_non_counter_tensors'],
                           'non-counter inventory drift')
            report['compatibility_audit'] = audit
            model.load_state_dict(mapped, strict=True, assign=True)
            loader.require(all(v.device.type == 'cpu' for v in model.state_dict().values()), 'unmaterialized state')
            del mapped, checkpoint
            model.eval()
            report['checkpoint_loaded'] = True
            stage = 'authored_pretrained_fidelity'
            report['pretrained_inference_started'] = True
            fidelity.evaluate_cases(model, base, torch, report)
        report.update(complete=True, decision='compatibility_and_authored_fidelity_pass_not_musical_accuracy',
                      unintended_python_network_attempts=0)
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
    base, compat = fidelity.protocol(), compatibility()
    output = fidelity.private_output(args.output)
    env = fidelity.offline_environment(output)
    if args.worker:
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_XET',
                    'USE_TF', 'USE_FLAX', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'):
            loader.require(os.environ.get(key) == env[key], 'unsealed offline environment')
        worker(args, output, base, compat)
    else:
        subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                       env=env, check=True, timeout=base['resources']['timeout_seconds'])


if __name__ == '__main__':
    main()
