"""Fresh neural features from retained PCM, not time-warped cached tensors."""
import gc
import inspect
import json
import sys
import time

import numpy as np
import torch

from experiments.coupled_clock.run import state_hash
from experiments.tempo_alignment.run import load_pcm, OLD_SHA
from evaluation.parity import prehead_capture as geometry
from .data import ROOT, IDS, CHECKPOINT_SHA, require, sha, write_json


def capture(pairs, assets, output, device, check):
    started = time.monotonic()
    checkpoint_path = assets/'final0.complete.ckpt'
    require(sha(checkpoint_path) == CHECKPOINT_SHA, 'encoder checkpoint changed')
    sys.path.insert(0, str(assets/'upstream'))
    from beat_this.model.beat_tracker import BeatThis
    from beat_this.preprocessing import LogMelSpect
    from beat_this.utils import replace_state_dict_key
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    hparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in inspect.signature(BeatThis).parameters}
    encoder = BeatThis(**hparams).eval().to(device)
    encoder.load_state_dict(replace_state_dict_key(checkpoint['state_dict'], 'model.', ''))
    del checkpoint
    for p in encoder.parameters():
        p.requires_grad_(False)
    before = state_hash(encoder)
    frontend = LogMelSpect(device='cpu').eval()
    report_path = ROOT/'experiments/tempo_pairs/results-v1.json'
    require(sha(report_path) == OLD_SHA, 'renderer report changed')
    previous = json.loads(report_path.read_bytes())
    items, records, reports = [], {}, []
    for identity in IDS:
        previews = [r for r in previous['previews'] if r['source_id'] == identity and r['backend'] == 'rubberband']
        require(len(previews) == 5, 'preview population changed')
        items.append((identity, 'source', identity+'.source.wav', previews[0]['source_preview_sha256']))
        items.extend((identity, r['profile'], f"{identity}.rubberband.{r['profile']}.wav", r['wav_sha256']) for r in previews)
    for identity, profile, name, digest in items:
        check()
        pcm = load_pcm(pairs/name, digest).astype(np.float32)
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
        path = output/f'{identity}.{profile}.features.npz'
        archive = geometry.save_capture(path, arrays)
        records[identity, profile] = torch.from_numpy(arrays['hidden']).to(device)
        reports.append(dict(source_id=identity, profile=profile, wav_sha256=digest,
                            frames=len(mel), pcm_samples=len(pcm), feature_sha256=archive['sha256'],
                            max_head_reconstruction_error=max(reconstruction), encoder_chunks=len(chunks)))
        print(json.dumps(dict(event='captured', source_id=identity, profile=profile, frames=len(mel))), flush=True)
    require(state_hash(encoder) == before, 'frozen encoder changed')
    del encoder, frontend
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    report = dict(cases=reports, encoder_state_sha256=before, encoder_unchanged=True,
                  checkpoint_sha256=CHECKPOINT_SHA, elapsed_s=time.monotonic()-started)
    write_json(output/'capture.json', report)
    return records, report
