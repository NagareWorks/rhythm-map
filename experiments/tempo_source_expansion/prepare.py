"""Join complete Rust frontend outputs to immutable BRID references, offline."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from .design import ROOT, HERE, plan, require, sha

sys.path.insert(0, str(ROOT/'evaluation/parity'))
from brid_reference import array_manifest
from prehead_capture import finite_f32

REFERENCE = 'evaluation/datasets/brid-reference-v1.json'
REFERENCE_SHA = 'ad70e2ed3a005c1d349368e25be405abb5f6a779f1efcf22d2f495291aa34a6a'
CHECKPOINT_SHA = '8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331'
MEL_SHA = 'fdd59e65c515331308e4c8841edf99972deca646bdf6197744c2a5b7755e3de9'
IDS = [f'{i:04d}' for i in range(1, 94)]


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def fresh_output(path):
    path = path.resolve()
    require(not path.exists() and not path.is_relative_to(ROOT), 'fresh private output required')
    if os.name == 'nt':
        require(path.drive.lower() != os.environ.get('SystemDrive', 'C:').lower(), 'use data drive')
    path.mkdir(parents=True)
    return path


def closure():
    frozen = plan()
    require(frozen == json.loads((HERE/'plan-v1.json').read_bytes()), 'design changed')
    paths = [REFERENCE, 'experiments/tempo_source_expansion/plan-v1.json',
             'experiments/tempo_source_expansion/ALIGNMENT-v1.md',
             'experiments/tempo_source_expansion/prepare.py',
             'experiments/tempo_source_expansion/capture.py',
             'crates/rhythm-map-eval/examples/brid_frontend.rs',
             'crates/rhythm-map-beat-this/src/audio.rs', 'Cargo.lock',
             'evaluation/parity/prehead-capture-v1.json']
    refs = json.loads((ROOT/REFERENCE).read_bytes())
    require(sha(ROOT/REFERENCE) == REFERENCE_SHA, 'reference report changed')
    result = dict(frozen['source_sha256'], **refs['source_sha256'])
    require(all(sha(ROOT/p) == h for p, h in result.items()), 'frozen dependency changed')
    result.update({p: sha(ROOT/p) for p in paths})
    return result


def verify_join(front, row, pcm, mel, arrays):
    require(front['recording_id'] == row['recording_id']
            and front['audio_sha256'] == row['audio_sha256'], 'audio/record join differs')
    samples = round(row['duration_s']*22050)
    require(front['samples'] == samples and front['frames'] == row['frames']
            and row['frames'] == 1+samples//441, 'full-record geometry differs')
    finite_f32(pcm, (samples,))
    finite_f32(mel, (row['frames'], 128))
    require(array_manifest(arrays) == row['arrays'], 'reference arrays changed')


def build(frontend, references, output):
    sources = closure()
    refs = json.loads((ROOT/REFERENCE).read_bytes())
    require([r['recording_id'] for r in refs['records']] == IDS, 'reference population changed')
    require(json.loads((references/'report.json').read_bytes()) == refs, 'private reference report differs')
    front = json.loads((frontend/'report.json').read_bytes())
    require(front['schema'] == 'rhythm-map.brid-frontend.v1'
            and front['reference_sha256'] == REFERENCE_SHA and front['mel_model_sha256'] == MEL_SHA
            and front['exporter_sha256'] == sources['crates/rhythm-map-eval/examples/brid_frontend.rs']
            and front['audio_preprocessing_sha256'] == sources['crates/rhythm-map-beat-this/src/audio.rs']
            and front['sample_rate'] == 22050 and front['frame_rate'] == 50
            and front['full_recordings'] is True and front['encoder_calls'] == front['optimizer_steps'] == 0,
            'frontend contract drift')
    require([r['recording_id'] for r in front['records']] == IDS, 'frontend subset/reorder rejected')
    rows = []
    for f, row in zip(front['records'], refs['records']):
        identity = row['recording_id']
        pcm_path, mel_path = (frontend/f'{identity}.{k}.f32' for k in ('pcm', 'mel'))
        require(sha(pcm_path) == f['pcm_sha256'] and sha(mel_path) == f['mel_sha256'], 'frontend bytes changed')
        pcm = np.fromfile(pcm_path, dtype='<f4')
        mel = np.fromfile(mel_path, dtype='<f4').reshape(row['frames'], 128)
        with np.load(references/f'{identity}.reference.npz', allow_pickle=False) as packet:
            arrays = {k: packet[k] for k in packet.files}
        verify_join(f, row, pcm, mel, arrays)
        joined = dict(mel=mel, **arrays)
        packet_path = output/f'{identity}.input.npz'
        with packet_path.open('xb') as stream:
            np.savez(stream, **joined)
        with np.load(packet_path, allow_pickle=False) as replay:
            require(array_manifest({k: replay[k] for k in replay.files}) == array_manifest(joined), 'join roundtrip differs')
        rows.append(dict(recording_id=identity, frames=row['frames'], samples=f['samples'],
                         audio_sha256=f['audio_sha256'], pcm_sha256=f['pcm_sha256'],
                         input_sha256=sha(packet_path), arrays=array_manifest(joined)))
    return dict(schema='rhythm-map.brid-feature-inputs.v1', source_sha256=sources,
                frontend_report_sha256=sha(frontend/'report.json'), records=rows,
                recording_count=93, frames=sum(r['frames'] for r in rows),
                alignment_disposition='preserve_published_labels_with_residual_uncertainty',
                label_shifts=0, source_groups=1, encoder_calls=0, optimizer_steps=0,
                training_ready=False, holdout_access=False, production_change=False)


def load_inputs(directory, expected_report_sha):
    require(sha(directory/'report.json') == expected_report_sha, 'input report identity differs')
    report = json.loads((directory/'report.json').read_bytes())
    require(report['schema'] == 'rhythm-map.brid-feature-inputs.v1'
            and report['source_sha256'] == closure(), 'input source closure differs')
    require([r['recording_id'] for r in report['records']] == IDS, 'all 93 ordered inputs required')
    for row in report['records']:
        path = directory/f"{row['recording_id']}.input.npz"
        require(sha(path) == row['input_sha256'], 'input archive changed')
        with np.load(path, allow_pickle=False) as packet:
            require(array_manifest({k: packet[k] for k in packet.files}) == row['arrays'], 'input array changed')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frontend', 'references', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    out = fresh_output(args.output)
    value = build(args.frontend, args.references, out)
    write_json(out/'report.json', value)
    load_inputs(out, sha(out/'report.json'))
    print(json.dumps({k: v for k, v in value.items() if k not in ('records', 'source_sha256')}))
