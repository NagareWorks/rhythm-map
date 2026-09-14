"""Fresh schedule encoding is forbidden until both arms' exports are frozen."""
import gc
import inspect
import sys
import time

import numpy as np
import torch

from experiments.coupled_clock.run import state_hash
from experiments.tempo_alignment.run import load_pcm
from evaluation.parity import prehead_capture as geometry
from . import data
from .data import require, sha, write_json


def verify_freeze(output, freeze):
    require(set(freeze) == {a+s for a in ('local-only', 'local-retained') for s in ('-selected', '-final')},
            'both selected and final exports must be frozen')
    require(all(sha(output/(k+'.weights.npz')) == v for k, v in freeze.items()), 'frozen exports changed')


def capture(prepared, preparation, assets, output, device, freeze, check):
    verify_freeze(output, freeze)
    started = time.monotonic()
    checkpoint_path = assets/'final0.complete.ckpt'
    require(sha(checkpoint_path) == data.old.CHECKPOINT_SHA, 'encoder checkpoint changed')
    sys.path.insert(0, str(assets/'upstream'))
    from beat_this.model.beat_tracker import BeatThis
    from beat_this.preprocessing import LogMelSpect
    from beat_this.utils import replace_state_dict_key
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    hparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in inspect.signature(BeatThis).parameters}
    encoder = BeatThis(**hparams).eval().to(device)
    encoder.load_state_dict(replace_state_dict_key(checkpoint['state_dict'], 'model.', ''))
    del checkpoint
    encoder.requires_grad_(False)
    before = state_hash(encoder)
    frontend = LogMelSpect(device='cpu').eval()
    records, reports = {}, []
    for item in preparation['pairs']:
        check()
        identity, profile, digest = item['source_id'], item['profile'], item['wav_sha256']
        pcm = load_pcm(prepared/f'{identity}.{profile}.wav', digest).astype(np.float32)
        with torch.inference_mode():
            mel = frontend(torch.from_numpy(pcm)).numpy().copy()
        chunks, reconstruction = [], []
        for piece in geometry.split(mel):
            hidden = []
            handle = encoder.task_heads.register_forward_pre_hook(lambda _module, inp: hidden.append(inp[0].detach().clone()))
            try:
                with torch.inference_mode():
                    heads = encoder(torch.from_numpy(piece.copy()).to(device)[None])
                    require(len(hidden) == 1, 'nonunique feature tap')
                    recreated = encoder.task_heads(hidden[0])
                    for key in ('beat', 'downbeat'):
                        require(torch.allclose(heads[key], recreated[key], atol=2e-5, rtol=1e-6), 'head reconstruction failed')
                        reconstruction.append(float((heads[key]-recreated[key]).abs().max()))
                    chunks.append(dict(hidden=hidden[0][0].cpu().numpy(), **{k: heads[k][0].cpu().numpy() for k in ('beat', 'downbeat')}))
            finally:
                handle.remove()
        arrays = geometry.stitch(len(mel), chunks)
        archive = geometry.save_capture(output/f'{identity}.{profile}.features.npz', arrays)
        records[identity, profile] = torch.from_numpy(arrays['hidden']).to(device)
        reports.append(dict(source_id=identity, profile=profile, wav_sha256=digest, frames=len(mel),
                            pcm_samples=len(pcm), feature_sha256=archive['sha256'],
                            max_head_reconstruction_error=max(reconstruction), encoder_chunks=len(chunks)))
        print(f'captured {identity} {profile}', flush=True)
    require(len(records) == 6 and before == state_hash(encoder), 'incomplete capture or encoder changed')
    verify_freeze(output, freeze)
    del encoder, frontend
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    report = dict(cases=reports, encoder_state_sha256=before, encoder_unchanged=True,
                  checkpoint_sha256=data.old.CHECKPOINT_SHA, frozen_exports=freeze, elapsed_s=time.monotonic()-started)
    write_json(output/'capture.json', report)
    return records, report
