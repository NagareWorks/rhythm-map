"""Research-only, offline, identity-checked MusicFM-FMA construction/loading.

No decoder, acquisition, native adapter or production option lives here. The
network guard detects accidental Python networking; it is not a hostile-code
sandbox. Run real checkpoints only in an externally resource-bounded process.
"""
from collections import OrderedDict
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import random
import socket
import stat
import sys
import traceback
import types
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SCREEN_SHA = '29fb352b002125b84bbac6e962c8e4126a5d34636399d40a211dd95705b3a283'
CONFIG_NAME = 'facebook/wav2vec2-conformer-rope-large-960h-ft'
SOURCE = {
    'modules/random_quantizer.py': (3055, '2ffae390fbf7fefa8a62000ba0490af92ecc001d3b5715fb98f7a1e1578c2aa2'),
    'modules/features.py': (1869, 'aa8e99711f4a4eab522ff89fe86fc27e281e373ed0ae416743736da229f171da'),
    'modules/conv.py': (3154, '585705a6450db374e8c411034e846491f7f48d4ed0516fbf8d3b84855ef0d7d1'),
    'model/musicfm_25hz.py': (8605, '9645e51938e7a73689a0792ecb15ee957e0723582c8f040949eadd25036ec804'),
}
MAX_SAMPLES = 48000  # Authored, unpadded clips only in this first loader gate.
RUNTIME_SHA = '753b87d3630b52e58e5544df217b23ea80daf76aa8035ecb47a6aa9b35981f62'
REQUIREMENTS_SHA = 'ffafd956af668b5467b8785a73fb7d05df0435a460d108bf3b0f67ef598b69e1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError('nonfinite JSON number')

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    require(type(value) is dict, 'expected JSON object')
    return value


@contextmanager
def verified_file(path, size_bytes, sha256):
    """Hash a regular file, then give the deserializer the SAME open handle.

    Caller-owned immutable files are required; this does not lock out hostile
    concurrent writes. The exact digest and byte count are checked before use.
    """
    path = Path(path)
    require(type(size_bytes) is int and 0 < size_bytes <= 1_400_000_000 and
            isinstance(sha256, str) and len(sha256) == 64 and
            all(c in '0123456789abcdef' for c in sha256), 'invalid file identity')
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), 'expected regular non-link file')
    require(info.st_size == size_bytes, 'file size mismatch')
    with path.open('rb') as stream:
        accumulator, count = hashlib.sha256(), 0
        while block := stream.read(min(1024 * 1024, size_bytes + 1 - count)):
            count += len(block)
            require(count <= size_bytes, 'file grew beyond size budget')
            accumulator.update(block)
        require(count == size_bytes and accumulator.hexdigest() == sha256, 'file byte identity mismatch')
        stream.seek(0)
        yield stream


def small_file(path, identity):
    require(identity['size_bytes'] <= 65536, 'small-file budget exceeded')
    with verified_file(path, identity['size_bytes'], identity['sha256']) as stream:
        return stream.read()


def screen():
    data = (HERE / 'musicfm-artifact-screen-v1.json').read_bytes()
    require(digest(data) == SCREEN_SHA, 'feasibility screen changed')
    return strict_json(data)


def token_geometry(samples):
    require(type(samples) is int and 1024 < samples <= MAX_SAMPLES,
            'authored clip must have 1025..48000 samples at 24 kHz')
    mel = samples // 240
    tokens = (mel + 3) // 4
    return dict(mel_frames=mel, tokens=tokens, origin_samples=0, stride_samples=960,
                last_center_samples=(tokens - 1) * 960)


def validate_config(config):
    required = dict(model_type='wav2vec2-conformer', hidden_size=1024,
                    num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
                    position_embeddings_type='rotary', rotary_embedding_base=10000,
                    conv_depthwise_kernel_size=31, layerdrop=0.0, layer_norm_eps=1e-5)
    for key, expected in required.items():
        require(type(config.get(key)) is type(expected) and config[key] == expected,
                'unexpected external config: ' + key)
    require('auto_map' not in config, 'remote code configuration is forbidden')
    return config


def verified_inputs(upstream, config_path, stats_path):
    plan = screen()
    config = validate_config(strict_json(small_file(config_path, plan['configuration_dependency'])))
    spec = next(row for row in plan['artifacts'] if row['path'] == 'fma_stats.json')
    stats_bytes = small_file(stats_path, spec)
    statistics = strict_json(stats_bytes)
    require(statistics.get('melspec_2048_mean') == spec['melspec_2048_mean'] and
            statistics.get('melspec_2048_std') == spec['melspec_2048_std'], 'FMA statistics mismatch')
    source = {}
    for name, (size, sha) in SOURCE.items():
        path = Path(upstream) / name
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and not path.is_symlink() and
                info.st_size <= 2 * size, 'source file type/size mismatch')
        # Git checkouts may use CRLF. This canonicalization is explicit, not
        # applied to artifacts, config, statistics, or checkpoint bytes.
        data = path.read_bytes().replace(b'\r\n', b'\n')
        require(len(data) == size and digest(data) == sha, 'source identity mismatch: ' + name)
        source[name] = data
    return config, stats_bytes, source


@contextmanager
def no_network():
    """Fail on attempted Python socket/DNS operations, including swallowed errors."""
    attempted = []

    def denied(*_args, **_kwargs):
        # Function names diagnose initialization without leaking socket targets.
        attempted.append(' > '.join(frame.name for frame in traceback.extract_stack(limit=12)[:-1]))
        raise RuntimeError('MusicFM research worker attempted networking')

    with ExitStack() as stack:
        # urllib3 probes IPv6 by binding localhost during import. Suppress the
        # capability probe rather than allowing even a loopback socket bind.
        stack.enter_context(patch.object(socket, 'has_ipv6', False))
        for name in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'create_connection'):
            stack.enter_context(patch.object(socket, name, denied))
        for name in ('connect', 'connect_ex', 'sendto', 'bind', 'listen', 'sendmsg'):
            if hasattr(socket.socket, name):
                stack.enter_context(patch.object(socket.socket, name, denied))
        try:
            yield
        finally:
            require(not attempted, 'network attempt detected; result rejected: ' + '; '.join(attempted))


@contextmanager
def source_module(source):
    """Import only verified in-memory source, never arbitrary checkout modules."""
    require(not any(n == 'musicfm' or n.startswith('musicfm.') for n in sys.modules),
            'MusicFM module namespace already occupied')
    require(set(source) == set(SOURCE), 'incomplete source inventory')
    names = []
    try:
        for name in ('musicfm', 'musicfm.modules', 'musicfm.model'):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
            names.append(name)
        for filename in SOURCE:
            name = 'musicfm.' + filename[:-3].replace('/', '.')
            module = types.ModuleType(name)
            module.__file__ = '<verified-musicfm>/' + filename
            module.__package__ = name.rsplit('.', 1)[0]
            sys.modules[name] = module
            names.append(name)
            exec(compile(source[filename], module.__file__, 'exec'), module.__dict__)
        yield sys.modules['musicfm.model.musicfm_25hz']
    finally:
        for name in reversed(names):
            sys.modules.pop(name, None)


def runtime_versions(expected):
    actual = {name: importlib.metadata.version(name) for name in expected}
    require(actual == expected, 'runtime dependency versions differ from lock')
    return actual


def runtime_lock():
    # These checked-in text files, like source, have canonical LF identities.
    data = (HERE / 'musicfm-runtime-win64.json').read_bytes().replace(b'\r\n', b'\n')
    requirements = (HERE / 'musicfm-runtime-win64.txt').read_bytes().replace(b'\r\n', b'\n')
    require(digest(data) == RUNTIME_SHA and digest(requirements) == REQUIREMENTS_SHA,
            'runtime lock changed')
    lock = strict_json(data)
    actual = [line for line in requirements.decode().splitlines() if line and not line.startswith('#')]
    expected = [f"{p['name']}=={p['version']} --hash=sha256:{p['sha256']}" for p in lock['packages']]
    require(actual == expected and len({p['name'] for p in lock['packages']}) == len(expected),
            'runtime requirements/inventory mismatch')
    return lock


def construct_meta(upstream, config_path, stats_path):
    """Construct the exact architecture without allocating trained parameters.

    Mel filter/window buffers need real CPU computation in torchaudio. Only
    those tiny buffers are constructed on CPU; network parameters remain meta.
    Callers must validate runtime identity and apply their resource envelope.
    """
    config_data, stats_bytes, source = verified_inputs(upstream, config_path, stats_path)
    with no_network():
        import torch
        from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerConfig
        with source_module(source) as module:
            original_frontend = module.MelSTFT
            calls = []

            def config_from_local(name, *args, **kwargs):
                require(name == CONFIG_NAME and not args and not kwargs and not calls,
                        'unexpected configuration request')
                calls.append(name)
                return Wav2Vec2ConformerConfig.from_dict(config_data)

            def open_stats(path, mode):
                require(path == str(stats_path) and mode == 'r', 'unexpected upstream file open')
                return io.StringIO(stats_bytes.decode('utf-8'))

            def cpu_frontend(*args, **kwargs):
                with torch.device('cpu'):
                    return original_frontend(*args, **kwargs)

            def unsafe_load(*_args, **_kwargs):
                raise RuntimeError('upstream deserialization is disabled')

            random_state = random.getstate()
            try:
                with torch.random.fork_rng(devices=[]), torch.device('meta'), \
                        patch.object(Wav2Vec2ConformerConfig, 'from_pretrained', config_from_local), \
                        patch.object(module, 'open', open_stats, create=True), \
                        patch.object(module, 'MelSTFT', cpu_frontend), patch.object(torch, 'load', unsafe_load):
                    model = module.MusicFM25Hz(is_flash=False, model_path=None, stat_path=str(stats_path))
            finally:
                random.setstate(random_state)
            require(calls == [CONFIG_NAME], 'local config was not consumed exactly once')
            require(len(model.conformer.layers) == 12 and model.conformer.config.hidden_size == 1024,
                    'effective architecture mismatch')
            require(all(p.device.type == 'meta' and p.dtype == torch.float32 for p in model.parameters()),
                    'unexpected real/non-float32 parameter allocation')
            return model.eval()


def tensor_schema(model):
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype))
            for key, value in model.state_dict().items()}


def checked_state(checkpoint, expected, torch):
    require(type(checkpoint) in (dict, OrderedDict) and 'state_dict' in checkpoint,
            'missing checkpoint state_dict')
    state = checkpoint['state_dict']
    require(type(state) in (dict, OrderedDict) and len(state) == len(expected), 'state key count mismatch')
    mapped = {}
    for key, value in state.items():
        require(type(key) is str and key.startswith('model.'), 'unexpected checkpoint key prefix')
        name = key[6:]
        require(name in expected and name not in mapped, 'unknown/duplicate state key')
        require(type(value) is torch.Tensor and value.device.type == 'cpu' and
                value.layout == torch.strided and not value.is_quantized,
                'expected ordinary CPU state tensor')
        require(list(value.shape) == expected[name]['shape'] and
                str(value.dtype) == expected[name]['dtype'], 'state tensor shape/dtype mismatch')
        require(bool(torch.isfinite(value).all()), 'nonfinite checkpoint tensor')
        mapped[name] = value
    require(set(mapped) == set(expected), 'missing state keys')
    return mapped


def restricted_load(path, identity, load):
    """Injectable deserializer for rejection tests; no unsafe fallback exists."""
    with verified_file(path, identity['size_bytes'], identity['sha256']) as stream:
        return load(stream, map_location='cpu', weights_only=True)


def load_fma(model, checkpoint_path):
    import torch
    expected = tensor_schema(model)
    identity = next(row for row in screen()['artifacts'] if row['path'] == 'pretrained_fma.pt')
    with no_network():
        checkpoint = restricted_load(checkpoint_path, identity, torch.load)
        mapped = checked_state(checkpoint, expected, torch)
        # assign=True replaces meta tensors directly, avoiding a second full
        # random model allocation. Exact key/shape/dtype checks precede this.
        model.load_state_dict(mapped, strict=True, assign=True)
        require(all(v.device.type == 'cpu' for v in model.state_dict().values()),
                'unmaterialized state after strict load')
        return model.eval()
