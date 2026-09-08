"""Windows-only, bounded loader prerequisite probe; no MusicFM checkpoint input.

Real pretrained inference and musical discrimination require later gates. The
small checkpoints below are authored tensors, not downloaded model weights.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import musicfm_loader as loader

JOB_HANDLE = None


def bound_windows_process():
    """Apply a 4 GiB committed-memory limit before importing neural libraries."""
    import ctypes
    from ctypes import wintypes
    loader.require(sys.platform == 'win32', 'this pinned runtime probe is Windows-only')

    class Basic(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                    ('flags', wintypes.DWORD), ('min_working_set', ctypes.c_size_t),
                    ('max_working_set', ctypes.c_size_t), ('active_processes', wintypes.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                    ('scheduling', wintypes.DWORD)]

    class Counters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

    class Extended(ctypes.Structure):
        _fields_ = [('basic', Basic), ('io', Counters), ('process_memory', ctypes.c_size_t),
                    ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                    ('peak_job', ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.CreateJobObjectW(None, None)
    loader.require(bool(handle), 'cannot create memory-limit job')
    limits = Extended()
    limits.basic.flags = 0x100 | 0x2000  # PROCESS_MEMORY | KILL_ON_JOB_CLOSE
    limits.process_memory = 4 * 1024 ** 3
    loader.require(kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)),
                   'cannot configure memory limit')
    loader.require(kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()),
                   'cannot apply memory limit')
    global JOB_HANDLE
    JOB_HANDLE = handle  # Keep open until OS process teardown; no child workloads.
    return limits.process_memory


def authored_checkpoint_controls(directory, torch):
    tiny = torch.nn.Linear(2, 3).eval()
    with torch.no_grad():
        tiny.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(3, 2) / 8)
        tiny.bias.zero_()
    path = directory / 'authored-state.pt'
    with path.open('xb') as target:
        torch.save({'state_dict': {'model.' + k: v for k, v in tiny.state_dict().items()}}, target)
    raw = path.read_bytes()
    identity = dict(size_bytes=len(raw), sha256=loader.digest(raw))
    restored = loader.restricted_load(path, identity, torch.load)
    mapped = loader.checked_state(restored, loader.tensor_schema(tiny), torch)
    with torch.device('meta'):
        target = torch.nn.Linear(2, 3)
    target.load_state_dict(mapped, strict=True, assign=True)
    loader.require(all(torch.equal(v, target.state_dict()[k]) for k, v in tiny.state_dict().items()),
                   'authored state round trip changed')

    rejected = []
    for label, value in [('wrong_shape', torch.zeros(6)), ('wrong_dtype', torch.zeros((3, 2), dtype=torch.float64)),
                         ('nonfinite', torch.full((3, 2), float('nan')))]:
        case = {'state_dict': dict(restored['state_dict'], **{'model.weight': value})}
        try:
            loader.checked_state(case, loader.tensor_schema(tiny), torch)
        except ValueError:
            rejected.append(label)
    loader.require(len(rejected) == 3, 'authored state rejection failed')

    # A callable outside the restricted allowlist must never execute, even for
    # a hash-verified authored file. eval here would raise a distinct exception.
    class UnsafeAuthored:
        def __reduce__(self):
            return eval, ('1/0',)

    unsafe_path = directory / 'authored-unsupported-pickle.pt'
    with unsafe_path.open('xb') as stream:
        torch.save({'state_dict': UnsafeAuthored()}, stream)
    raw = unsafe_path.read_bytes()
    import pickle
    try:
        loader.restricted_load(unsafe_path, dict(size_bytes=len(raw), sha256=loader.digest(raw)), torch.load)
    except pickle.UnpicklingError:
        rejected.append('unsupported_pickle_callable')
    loader.require(rejected[-1] == 'unsupported_pickle_callable', 'unsafe authored pickle was not rejected')
    return dict(roundtrip_exact=True, rejected=rejected, pretrained_weights=False)


def frontend_controls(model, upstream, config, statistics, torch):
    # Same pinned operators but tiny channel counts. These test coordinate
    # geometry, NOT MusicFM encoder features or musical-clock discrimination.
    _config, _stats, source = loader.verified_inputs(upstream, config, statistics)
    with loader.source_module(source) as module:
        tiny = module.Conv2dSubsampling(1, 2, 4, strides=[2, 2], n_bands=128).eval()
    tiny_state = {key: value.clone() for key, value in tiny.state_dict().items()}
    frontend = model.preprocessor_melspec_2048
    spectrogram = frontend.mel_stft.spectrogram
    mel_scale = frontend.mel_stft.mel_scale
    expected = dict(n_fft=2048, win_length=2048, hop_length=240, pad=0, power=2.0,
                    normalized=False, center=True, pad_mode='reflect')
    loader.require(all(getattr(spectrogram, k) == v for k, v in expected.items()) and
                   torch.equal(spectrogram.window, torch.hann_window(2048)) and
                   frontend.mel_stft.sample_rate == 24000 and mel_scale.n_mels == 128 and
                   mel_scale.f_min == 0.0 and mel_scale.f_max == 12000.0 and
                   mel_scale.norm is None and mel_scale.mel_scale == 'htk' and
                   frontend.is_db and frontend.amplitude_to_db.stype == 'power' and
                   frontend.amplitude_to_db.top_db is None, 'frontend defaults changed')
    rows = []
    with torch.inference_mode():
        for n, kind in ((1025, 'silence'), (1199, 'impulse'), (1200, 'tone'),
                        (1919, 'tone'), (1920, 'impulse'), (24001, 'tone'), (48000, 'silence')):
            x = torch.zeros((1, n), dtype=torch.float32)
            if kind == 'impulse':
                x[0, n // 2] = 0.5
            elif kind == 'tone':
                x[0] = 0.2 * torch.sin(torch.arange(n, dtype=torch.float32) * (2 * torch.pi * 440 / 24000))
            original = x.clone()
            mel = model.preprocessing(x, ['melspec_2048'])['melspec_2048']
            geometry = loader.token_geometry(n)
            loader.require(tuple(mel.shape) == (1, 128, geometry['mel_frames']), 'mel geometry mismatch')
            normalized = model.normalize({'melspec_2048': mel.clone()})['melspec_2048']
            direct = (mel - model.stat['melspec_2048_mean']) / model.stat['melspec_2048_std']
            loader.require(torch.equal(normalized, direct), 'global normalization mismatch')
            encoded = tiny(normalized)
            repeated = tiny(normalized)
            loader.require(tuple(encoded.shape) == (1, geometry['tokens'], 4) and
                           bool(torch.isfinite(encoded).all()) and torch.equal(encoded, repeated) and
                           torch.equal(x, original), 'authored geometry/repeatability/input mismatch')
            rows.append(dict(samples=n, kind=kind, **geometry, normalization_exact=True,
                             tiny_convolution_repeat_exact=True, input_unchanged=True))
    loader.require(all(torch.equal(value, tiny.state_dict()[key]) for key, value in tiny_state.items()),
                   'tiny convolution state changed')
    return rows


def notice_inventory(lock):
    """Identify notices retained by the installed wheels, not a product SBOM."""
    import importlib.metadata as metadata
    rows, missing = [], []
    for package in lock['packages']:
        distribution = metadata.distribution(package['name'])
        notices = []
        for file in distribution.files or []:
            name = str(file).replace('\\', '/')
            if '.dist-info/' in name and any(part in name.lower() for part in ('license', 'notice', 'copying')):
                raw = distribution.locate_file(file).read_bytes()
                notices.append(dict(path=name, size_bytes=len(raw), sha256=loader.digest(raw)))
        if not notices:
            missing.append(package['name'])
        rows.append(dict(name=package['name'], version=distribution.version, notices=notices,
                         redistribution_audit_complete=False))
    loader.require(missing == lock['expected_missing_wheel_notice_files'], 'wheel notice inventory changed')
    return rows


def worker(args):
    limit = bound_windows_process()
    lock = loader.runtime_lock()
    loader.require(sys.platform == lock['platform'] and platform.machine() == lock['machine'] and
                   platform.python_version() == lock['python'], 'runtime platform/Python mismatch')
    versions = loader.runtime_versions({p['name']: p['version'] for p in lock['packages']})
    directory = Path(args.output)
    directory.mkdir(parents=False, exist_ok=False)
    with loader.no_network():
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(142)
        begin = time.perf_counter()
        model = loader.construct_meta(args.upstream, args.config, args.stats)
        elapsed = time.perf_counter() - begin
        schema = loader.tensor_schema(model)
        loader.require(schema['linear.weight'] == dict(shape=[4096, 1024], dtype='torch.float32'),
                       'task projection schema mismatch')
        schema_bytes = json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()
        controls = authored_checkpoint_controls(directory, torch)
        frontend = frontend_controls(model, args.upstream, args.config, args.stats, torch)
        loader.require(loader.tensor_schema(model) == schema, 'model schema changed during probe')
        from prehead_capture import peak_rss_bytes
        report = dict(schema='rhythm-map.musicfm-loader-prerequisite.v1', complete=True,
                      decision='loader_prerequisites_only_not_pretrained_fidelity',
                      pretrained_checkpoint_acquired=False, pretrained_checkpoint_loaded=False,
                      pretrained_musicfm_inference=False, music_access=False, training=False,
                      production_changed=False, runtime_versions=versions,
                      source_sha256={name: loader.digest((HERE / name).read_bytes()) for name in
                                     ('musicfm_loader.py', 'musicfm_loader_probe.py', 'musicfm-runtime-win64.json',
                                      'musicfm-runtime-win64.txt', 'musicfm-artifact-screen-v1.json')},
                      process_memory_limit_bytes=limit, peak_rss_bytes=peak_rss_bytes(),
                      torch_threads=torch.get_num_threads(), construct_meta_s=elapsed,
                      meta_parameter_elements=sum(p.numel() for p in model.parameters()),
                      real_buffer_elements=sum(v.numel() for v in model.buffers() if v.device.type == 'cpu'),
                      state_key_count=len(schema), state_schema_sha256=loader.digest(schema_bytes),
                      effective_config_sha256=loader.digest(model.conformer.config.to_json_string().encode()),
                      authored_checkpoints=controls, authored_frontend_geometry=frontend,
                      frontend_defaults_verified=True, tiny_convolution_state_unchanged=True,
                      dependency_notices=notice_inventory(lock), unintended_python_network_attempts=0)
    with (directory / 'report.json').open('x', encoding='utf-8', newline='\n') as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write('\n')
    print(json.dumps(dict(complete=True, state_key_count=report['state_key_count'],
                         meta_parameter_elements=report['meta_parameter_elements'],
                         peak_rss_bytes=report['peak_rss_bytes'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'config', 'stats', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    loader.require(sys.platform == 'win32' and output.drive.lower() != os.environ.get('SystemDrive', 'C:').lower(),
                   'private Windows output must be off the system drive')
    loader.require(not output.is_relative_to(HERE.parents[1]) and not output.exists(),
                   'output must be fresh and outside the repository')
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_XET='1',
                       USE_TF='0', USE_FLAX='0', PYTHONDONTWRITEBYTECODE='1')
    # Do not let import-time cache-version bookkeeping touch a shared/system cache.
    environment['HF_HOME'] = str(output / 'hf-cache')
    environment['HF_HUB_CACHE'] = str(output / 'hf-cache' / 'hub')
    environment['TRANSFORMERS_CACHE'] = str(output / 'hf-cache' / 'hub')
    environment.pop('PYTHONPATH', None)
    if args.worker:
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY',
                    'HF_HUB_DISABLE_XET', 'USE_TF', 'USE_FLAX', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'):
            loader.require(os.environ.get(key) == environment[key], 'worker environment not sealed')
        worker(args)
    else:
        subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                       env=environment, check=True, timeout=180)


if __name__ == '__main__':
    main()
