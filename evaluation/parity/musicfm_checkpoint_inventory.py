"""Post-refusal metadata inventory: no state assignment, repair, or inference."""
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import musicfm_fidelity as fidelity
import musicfm_loader as loader


def differences(actual, expected):
    common = sorted(set(actual) & set(expected))
    return dict(missing=sorted(set(expected) - set(actual)), unexpected=sorted(set(actual) - set(expected)),
                mismatches=[dict(key=k, actual=actual[k], expected=expected[k]) for k in common
                            if actual[k] != expected[k]],
                exact_entries=sum(actual[k] == expected[k] for k in common),
                actual_count=len(actual), expected_count=len(expected))


def worker(args, output, plan):
    from musicfm_loader_probe import bound_windows_process
    from prehead_capture import peak_rss_bytes
    loader.require(bound_windows_process() == plan['resources']['process_memory_limit_bytes'], 'limit drift')
    observed = fidelity.resources(output, plan['resources'])
    lock = loader.runtime_lock()
    loader.require(sys.platform == lock['platform'] and platform.machine() == lock['machine'] and
                   platform.python_version() == lock['python'], 'runtime mismatch')
    versions = loader.runtime_versions({p['name']: p['version'] for p in lock['packages']})
    output.mkdir()
    with loader.no_network():
        import torch
        torch.set_num_threads(2)
        model = loader.construct_meta(args.upstream, args.config, args.stats)
        expected = loader.tensor_schema(model)
        checkpoint = loader.restricted_load(args.checkpoint, plan['checkpoint'], torch.load)
        loader.require(type(checkpoint) in (dict, OrderedDict) and
                       type(checkpoint.get('state_dict')) in (dict, OrderedDict), 'unknown checkpoint root')
        actual, prefixes = {}, []
        for key, value in checkpoint['state_dict'].items():
            loader.require(type(key) is str and type(value) is torch.Tensor and value.device.type == 'cpu' and
                           value.layout == torch.strided and not value.is_quantized, 'unknown state entry')
            loader.require(bool(torch.isfinite(value).all()), 'nonfinite state entry')
            if not key.startswith('model.'):
                prefixes.append(key)
            name = key[6:] if key.startswith('model.') else key
            loader.require(name not in actual, 'duplicate mapped state key')
            actual[name] = dict(shape=list(value.shape), dtype=str(value.dtype))
        report = dict(schema='rhythm-map.musicfm-checkpoint-inventory.v1', complete=True,
                      scope='metadata-only-after-strict-load-refusal', protocol_sha256=fidelity.LOCK_SHA,
                      source_sha256=loader.digest(Path(__file__).read_bytes()),
                      checkpoint_identity=plan['checkpoint'], resources=observed, runtime_versions=versions,
                      all_state_tensors_finite=True, unexpected_prefixes=prefixes,
                      comparison=differences(actual, expected), state_assigned=False,
                      pretrained_inference=False, music_access=False, training=False, production_change=False,
                      peak_rss_bytes=peak_rss_bytes())
    with (output / 'report.json').open('x', encoding='utf-8', newline='\n') as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write('\n')
    print(json.dumps(report['comparison']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'config', 'stats', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan, output = fidelity.protocol(), fidelity.private_output(args.output)
    env = fidelity.offline_environment(output)
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
