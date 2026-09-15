"""Frozen Beat This capture only: no tempo model, labels in forwards or optimizer."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from evaluation.parity import prehead_capture as geometry
from .prepare import (ROOT, CHECKPOINT_SHA, array_manifest, fresh_output,
                      load_inputs, require, sha, write_json)


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def capture_chunk(model, mel):
    """Untapped baseline + two tapped passes; replay hidden bytes, not just heads."""
    geometry.finite_f32(mel, (len(mel), 128))
    require(not any(m.training for m in model.modules()), 'encoder must be eval')
    parameters = list(model.parameters())
    require(parameters and all(not p.requires_grad and p.dtype == torch.float32 for p in parameters),
            'encoder must be frozen float32')
    require(not model.task_heads._forward_pre_hooks, 'unexpected existing tap')
    x = torch.from_numpy(mel.copy()).to(parameters[0].device)[None]
    before = x.clone()
    outputs, errors = [], []
    with torch.inference_mode():
        baseline = model(x)
        for _ in range(2):
            hidden = []
            handle = model.task_heads.register_forward_pre_hook(
                lambda _module, inputs: hidden.append(inputs[0].detach().clone()))
            try:
                heads = model(x)
            finally:
                handle.remove()
            require(len(hidden) == 1 and hidden[0].shape == (1, len(mel), 512), 'wrong/nonunique pre-head tap')
            recreated = model.task_heads(hidden[0])
            result = dict(hidden=hidden[0][0].cpu().numpy().copy())
            for key in ('beat', 'downbeat'):
                require(torch.equal(heads[key], baseline[key]), 'tap changed model heads')
                require(torch.allclose(heads[key], recreated[key], atol=2e-5, rtol=1e-6), 'head reconstruction failed')
                errors.append(float((heads[key]-recreated[key]).abs().max()))
                result[key] = heads[key][0].cpu().numpy().copy()
            outputs.append(result)
        require(torch.equal(before, x), 'encoder mutated input')
    for key in geometry.KEYS:
        geometry.difference(outputs[0][key], outputs[1][key], exact=True)
    return outputs[0], max(errors)


def upstream_identity(assets, dependencies):
    require(sha(assets/'final0.complete.ckpt') == CHECKPOINT_SHA, 'checkpoint identity differs')
    pinned = json.loads((ROOT/'evaluation/parity/prehead-capture-v1.json').read_bytes())
    require(pinned['checkpoint_sha256'] == CHECKPOINT_SHA, 'old encoder pin differs')
    require(all(sha(assets/'upstream'/p) == h for p, h in pinned['upstream_source_sha256'].items()),
            'pinned Beat This source changed')
    sources = {}
    for prefix, directory in [('upstream', assets/'upstream'), ('dependencies', dependencies)]:
        for path in sorted(directory.rglob('*.py')):
            sources[f'{prefix}/{path.relative_to(directory).as_posix()}'] = sha(path)
    require(any(k.startswith('dependencies/rotary_embedding_torch/') for k in sources),
            'explicit rotary source dependency required')
    return sources


def run(args, output):
    started = time.monotonic()
    inputs = load_inputs(args.inputs, args.input_report_sha256)
    sources = upstream_identity(args.assets, args.dependencies)
    require(torch.__version__ == '2.10.0+cu129' and np.__version__ == '2.2.6'
            and torch.cuda.is_available(), 'requires registered CUDA capture runtime')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.4)
    torch.manual_seed(42)
    registration = dict(schema='rhythm-map.brid-capture-registration.v1',
        input_report_sha256=args.input_report_sha256, source_sha256=inputs['source_sha256'],
        checkpoint_sha256=CHECKPOINT_SHA, asset_source_sha256=sources,
        torch=torch.__version__, numpy=np.__version__, device=torch.cuda.get_device_name(),
        threads=2, tf32=False, deterministic=True, gpu_memory_fraction=.4,
        max_wall_seconds=1800, passes_per_chunk=3, feature_width=512, frame_rate=50,
        chunk_frames=1500, border_frames=6, stride_frames=1488, ownership='keep_first',
        optimizer_steps=0, recording_ids=[r['recording_id'] for r in inputs['records']])
    write_json(output/'registration.json', registration)
    sys.path.insert(0, str(args.dependencies.resolve()))
    sys.path.insert(0, str((args.assets/'upstream').resolve()))
    from beat_this.model.beat_tracker import BeatThis
    from beat_this.utils import replace_state_dict_key
    checkpoint = torch.load(args.assets/'final0.complete.ckpt', map_location='cpu', weights_only=True)
    hparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in inspect.signature(BeatThis).parameters}
    encoder = BeatThis(**hparams).eval().to('cuda')
    encoder.load_state_dict(replace_state_dict_key(checkpoint['state_dict'], 'model.', ''))
    del checkpoint
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    before = state_hash(encoder)
    rows = []
    with (output/'progress.jsonl').open('x', encoding='utf-8') as progress:
        for row in inputs['records']:
            identity = row['recording_id']
            path = args.inputs/f'{identity}.input.npz'
            require(sha(path) == row['input_sha256'], 'input changed after registration')
            with np.load(path, allow_pickle=False) as archive:
                mel = archive['mel'].copy()  # No label array is passed to the encoder.
            chunks, errors = [], []
            for piece in geometry.split(mel):
                require(time.monotonic()-started < 1800, 'capture wall cap exceeded')
                chunk, error = capture_chunk(encoder, piece)
                chunks.append(chunk)
                errors.append(error)
            arrays = geometry.stitch(len(mel), chunks)
            require(len(arrays['hidden']) == row['frames'], 'feature/reference extent differs')
            archive = geometry.save_capture(output/f'{identity}.features.npz', arrays)
            receipt = dict(recording_id=identity, frames=row['frames'], input_sha256=row['input_sha256'],
                feature_sha256=archive['sha256'], arrays=array_manifest(arrays), chunks=len(chunks),
                forward_calls=3*len(chunks), untapped_heads_bit_exact=True,
                hidden_replay_bit_exact=True, roundtrip_bit_exact=archive['roundtrip_bit_exact'],
                max_head_reconstruction_error=max(errors))
            rows.append(receipt)
            print(json.dumps(receipt), file=progress, flush=True)
            print(json.dumps(dict(event='captured', recording_id=identity, frames=row['frames'])), flush=True)
    require(state_hash(encoder) == before, 'frozen encoder mutated')
    require(upstream_identity(args.assets, args.dependencies) == sources, 'encoder assets changed')
    load_inputs(args.inputs, args.input_report_sha256)
    elapsed = time.monotonic()-started
    require(elapsed < 1800, 'capture wall cap exceeded')
    report = dict(schema='rhythm-map.brid-features.v1', registration_sha256=sha(output/'registration.json'),
        input_report_sha256=args.input_report_sha256, records=rows, recording_count=len(rows),
        frames=sum(r['frames'] for r in rows), encoder_state_sha256=before, encoder_unchanged=True,
        encoder_forward_calls=sum(r['forward_calls'] for r in rows), elapsed_s=elapsed,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        optimizer_steps=0, training_ready=False, holdout_access=False, production_change=False)
    write_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'assets', 'dependencies', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--input-report-sha256', required=True)
    args = parser.parse_args()
    out = fresh_output(args.output)
    try:
        value = run(args, out)
    except Exception as error:
        # Leave partial packets, registration and receipts; never retry a subset.
        write_json(out/'failure.json', dict(error_type=type(error).__name__,
                                           message=str(error), optimizer_steps=0))
        raise
    print(json.dumps({k: v for k, v in value.items() if k != 'records'}), flush=True)
