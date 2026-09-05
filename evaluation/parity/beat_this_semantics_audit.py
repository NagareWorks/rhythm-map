"""Pinned backend semantics and frozen-head audit; no calibration or decoder.

The optional upstream check needs torch, but CI controls and capture statistics
use only the standard library and existing audit helpers. No network access,
model inference, parameter fitting, holdout access or probability clipping.
"""
import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import subprocess

import dense_clock_evidence as dense

ROOT = dense.ROOT
SOURCE_FILES = (
    'beat_this/model/loss.py', 'beat_this/model/pl_module.py',
    'beat_this/model/beat_tracker.py', 'beat_this/dataset/dataset.py',
    'beat_this/inference.py', 'launch_scripts/train.py',
)
WEIGHTS = {'beat': 19, 'downbeat': 86}


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def softplus(x):
    return max(x, 0) + math.log1p(math.exp(-abs(x)))


def shift_loss(logits, targets, mask=None, weight=19, tolerance=3):
    """Independent scalar form for binary-target shift-tolerant BCE controls.

    Return every retained center, including ignored neighbors with zero loss.
    Like upstream, mean reduction counts these zeros in its denominator.
    """
    dense.require(type(tolerance) is int and tolerance >= 0 and
                  len(logits) == len(targets) and len(logits) > 4 * tolerance,
                  'invalid loss dimensions')
    dense.require(all(math.isfinite(x) for x in logits) and
                  all(y in (0, 1) for y in targets) and
                  math.isfinite(weight) and weight > 0, 'invalid loss values')
    mask = [1] * len(logits) if mask is None else mask
    dense.require(len(mask) == len(logits) and all(m in (0, 1) for m in mask), 'invalid mask')
    rows = []
    for t in range(2 * tolerance, len(logits) - 2 * tolerance):
        peak = max(logits[t - tolerance:t + tolerance + 1])
        active = mask[t] * (targets[t] + 1 - max(targets[t - 2 * tolerance:t + 2 * tolerance + 1]))
        loss = active * (weight * softplus(-peak) if targets[t] else softplus(peak))
        rows.append(dict(center=t, active=active, target=targets[t], peak=peak, loss=loss))
    return sum(r['loss'] for r in rows) / len(rows), rows


def controls():
    targets = [int(t == 20) for t in range(41)]
    shifts = []
    for offset in range(-3, 4):
        values = [4. if t == 20 + offset else -4. for t in range(41)]
        loss, rows = shift_loss(values, targets)
        shifts.append(dict(offset_frames=offset, exact_annotation_logit=values[20],
                           loss=loss, positive_pooled_logit=rows[14]['peak']))
    # SumHead emits b=u+v and d=v. This is not a nested categorical head.
    b, d = -2., 2.
    return dict(shifted_pulse=shifts, sum_head_counterexample=dict(
        internal_u=-4., internal_v=2., beat=b, downbeat=d,
        naive_plain_mass=sigmoid(b) - sigmoid(d),
        weight_offset_plain_mass=sigmoid(b - math.log(19)) - sigmoid(d - math.log(86))),
        ordinary_weighted_bce_only=dict(
            zero_logit_implied_posterior={k: 1 / (1 + w) for k, w in WEIGHTS.items()},
            applicable_to_shift_tolerant_final0=False))


def default_arg(source, cls, arg):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    pairs = zip(init.args.args[-len(init.args.defaults):], init.args.defaults)
    return next(ast.literal_eval(value) for name, value in pairs if name.arg == arg)


def verify_upstream(checkout, checkpoint):
    """Verify exact Git objects/checkpoint before importing the audited loss.

    Only allowlisted checkpoint metadata is exported. weights_only=True must
    succeed; no unrestricted pickle fallback and no model is instantiated.
    """
    lock = json.loads((ROOT / 'evaluation/parity/reference-lock.json').read_text())
    revision = lock['reference_revision']
    def git(*args):
        return subprocess.check_output(['git', '-C', str(checkout), *args])
    dense.require(git('rev-parse', 'HEAD').decode().strip() == revision and
                  not git('status', '--porcelain'), 'upstream must be clean and pinned')
    sources = {p: git('show', f'{revision}:{p}') for p in SOURCE_FILES}
    # Checkpoint bytes are verified before deserialization.
    data = Path(checkpoint).read_bytes()
    dense.require(len(data) == lock['checkpoint']['size_bytes'] and
                  dense.sha(data) == lock['checkpoint']['sha256'], 'checkpoint identity changed')
    del data
    import torch
    h = torch.load(checkpoint, map_location='cpu', weights_only=True)['hyper_parameters']
    expected = dict(fps=50, loss_type='shift_tolerant_weighted_bce', pos_weights=WEIGHTS)
    dense.require(all(h.get(k) == v for k, v in expected.items()) and 'sum_head' not in h,
                  'checkpoint semantics changed')
    effective_sum = default_arg(sources['beat_this/model/beat_tracker.py'], 'BeatThis', 'sum_head')
    tolerance = default_arg(sources['beat_this/model/loss.py'], 'ShiftTolerantBCELoss', 'tolerance')
    dense.require(effective_sum is True and tolerance == 3, 'pinned defaults changed')
    spec = importlib.util.spec_from_file_location('audited_beat_this_loss', Path(checkout) / SOURCE_FILES[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Check the independent scalar equation against the pinned implementation,
    # including zero-gradient exact annotation when the peak is displaced.
    max_error = 0.
    target = torch.tensor([[int(t == 20) for t in range(41)]], dtype=torch.float64)
    central_gradient = None
    for offset in range(-3, 4):
        values = [4. if t == 20 + offset else -4. for t in range(41)]
        x = torch.tensor([values], dtype=torch.float64, requires_grad=True)
        actual = module.ShiftTolerantBCELoss(pos_weight=19)(x, target)
        max_error = max(max_error, abs(actual.item() - shift_loss(values, target[0].tolist())[0]))
        if offset == 1:
            actual.backward()
            central_gradient = x.grad[0, 20].item()
    dense.require(max_error < 1e-12 and central_gradient == 0., 'loss parity failed')
    return dict(reference_revision=revision, checkpoint_sha256=lock['checkpoint']['sha256'],
                source_sha256={p: dense.sha(v) for p, v in sources.items()},
                checkpoint_hyperparameters=expected, checkpoint_sum_head_present=False,
                effective_sum_head_from_pinned_loader_default=effective_sum,
                loss_tolerance_frames=tolerance, torch_weights_only=True,
                scalar_loss_parity_passed=True, displaced_annotation_gradient=central_gradient)


def head_counts(beats, downbeats):
    beats, downbeats = [float(v) for v in beats], [float(v) for v in downbeats]
    dense.require(len(beats) == len(downbeats) > 0 and
                  all(math.isfinite(v) for v in (*beats, *downbeats)), 'invalid heads')
    delta = math.log(WEIGHTS['downbeat']) - math.log(WEIGHTS['beat'])
    # Work in logits to avoid sigmoid underflow/saturation hiding violations.
    return dict(frames=len(beats), downbeat_gt_beat=sum(d > b for b, d in zip(beats, downbeats)),
                after_weight_offset_downbeat_gt_beat=sum(d - b > delta for b, d in zip(beats, downbeats)),
                beat_nonpositive=sum(b <= 0 for b in beats), downbeat_nonpositive=sum(d <= 0 for d in downbeats))


def audit_cohort(cohort, evidence_path, capture_dir):
    count, identity = dense.INPUTS[cohort]
    evidence, _ = dense.read_json(evidence_path, identity)
    old, _ = dense.read_json(ROOT / 'evaluation/parity/dense-clock-evidence-v1.json',
                             'e4826ec6996b58e404a9773f49fe21c46126b960ee1ca83aa719cce6fe18fd12')
    prior = next(c for c in old['cohorts'] if c['cohort'] == cohort)
    summary, summary_hash = dense.read_json(Path(capture_dir) / 'summary.json', prior['capture_summary_sha256'])
    records = dense.validate_summary(summary, evidence['cases'], count)
    sources = {k: dense.sha((ROOT / p).read_bytes()) for k, p in dense.SOURCES.items()}
    tracks = []
    for record, case in zip(records, evidence['cases']):
        payload, _ = dense.read_json(Path(capture_dir) / f"{case['id']}.json", record['capture_sha256'])
        beats = dense.validate_capture(payload, record, case, cohort, sources)
        tracks.append(dict(id=case['id'], capture_sha256=record['capture_sha256'],
                           counts=head_counts(beats, payload['downbeat_logits'])))
    pooled = {k: sum(t['counts'][k] for t in tracks) for k in tracks[0]['counts']}
    return dict(cohort=cohort, complete=True, frozen_evidence_sha256=identity,
                capture_summary_sha256=summary_hash, source_hashes=sources,
                pooled=pooled, cases=tracks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    for cohort in dense.INPUTS:
        parser.add_argument(f'--{cohort}-evidence', type=Path, required=True)
        parser.add_argument(f'--{cohort}-captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dense.require((args.upstream is None) == (args.checkpoint is None), 'supply both upstream and checkpoint')
    source_record = ROOT / 'evaluation/parity/beat-this-semantics-source-v1.json'
    if args.upstream is not None:
        source = verify_upstream(args.upstream, args.checkpoint)
        frozen, _ = dense.read_json(source_record)
        dense.require(source == frozen, 'upstream source record mismatch')
    else:
        source, _ = dense.read_json(source_record)
    cohorts = [audit_cohort(c, getattr(args, f'{c}_evidence'), getattr(args, f'{c}_captures'))
               for c in dense.INPUTS]
    report = dict(schema_version=1, purpose='backend_score_semantics_not_likelihood_calibration',
                  script_sha256=dense.sha(Path(__file__).read_bytes()),
                  source_record_sha256=dense.sha(source_record.read_bytes()),
                  dense_report_sha256='e4826ec6996b58e404a9773f49fe21c46126b960ee1ca83aa719cce6fe18fd12',
                  helper_sha256={p: dense.sha(Path(__file__).with_name(p).read_bytes()) for p in
                                 ('dense_clock_evidence.py', 'clock_phase_evidence.py', 'resampler_event_audit.py')},
                  upstream=source, controls=controls(), cohorts=cohorts,
                  holdout_opened=False, training_run=False, decoder_replayed=False,
                  fitted_parameters=False, production_observations_changed=False,
                  accuracy_improvement_claimed=False, adapter_accepted=False)
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(serialized)


if __name__ == '__main__':
    main()
