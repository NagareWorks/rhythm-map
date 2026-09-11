"""Bounded selected-checkpoint audit; no optimizer, parameter step or resume."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.run import state_hash
from experiments.phase_sync.model import PhaseSyncReadout
from experiments.scaled_adjoint.scaled import accumulate
from experiments.gradient_audit.audit import TERMS, GROUPS, audit_record, comparisons, group
from experiments.gradient_audit.register import ROOT, REPORT, PINS, make_plan, sha, verify, write_json


def flatten(value, prefix=''):
    out = {}
    for name, v in value.items():
        key = prefix + name
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            require(v is None or (isinstance(v, (int, float)) and np.isfinite(v)), 'nonfinite summary value')
            out[key] = v
    return out


def summarize(rows):
    require(rows, 'empty cohort')
    keys = rows[0]['metrics'].keys()
    require(all(r['metrics'].keys() == keys for r in rows), 'summary metric geometry differs')
    result = {}
    for key in keys:
        work = {}
        for row in rows:
            work.setdefault(row['work'], []).append(row['metrics'][key])
        missing = sum(v is None for values in work.values() for v in values)
        result[key] = dict(undefined_recordings=missing,
                           work_macro=None if missing else float(np.mean([np.mean(v) for v in work.values()])))
    return result


def load_model(path, expected):
    verify(path, expected)
    model = PhaseSyncReadout().to('cuda').eval()
    template = model.state_dict()
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(template), 'selected state keys differ')
        values = {k: torch.from_numpy(saved[k]).to('cuda') for k in saved.files}
    require(all(values[k].shape == template[k].shape and values[k].dtype == template[k].dtype
                and torch.isfinite(values[k]).all().item() for k in template), 'selected state geometry differs')
    model.load_state_dict(values, strict=True)
    require(all(p.grad is None for p in model.parameters()), 'loaded parameter grads')
    return model


def persist_case(output, row, kind, result, packet):
    """Commit complete diagnostic metrics before the next case or final report."""
    stem = row['id'] + '.' + kind
    packet_path = output / (stem + '.gradients.pt')
    with packet_path.open('xb') as stream:
        torch.save(dict(schema='gradient-audit-packet-v1', id=row['id'], checkpoint=kind,
                        gradients=packet), stream)
    result_row = dict(**row, checkpoint=kind, metrics=flatten(result), gradient_packet_sha256=sha(packet_path))
    write_json(output / (stem + '.case.json'), result_row)
    return result_row


def execute(inputs, predictions, plan_path, expected_plan, output):
    started = time.monotonic()
    require(not output.exists() and not output.resolve().is_relative_to(ROOT), 'new private output required')
    verify(plan_path, expected_plan)
    plan = json.loads(plan_path.read_bytes())
    require(plan == make_plan(), 'registered audit differs')
    torch.cuda.set_per_process_memory_fraction(.4)
    torch.cuda.reset_peak_memory_stats()
    output.mkdir()
    context, current_transport = {}, None
    def deadline():
        require(time.monotonic() - started <= plan['audit_seconds'], 'audit wall budget exceeded')
    def observe(term, transport):
        nonlocal current_transport
        context['term'], current_transport = term, transport
        deadline()
    def journal(item):
        with (output / 'journal.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(item, allow_nan=False) + '\n')
    try:
        old = json.loads(REPORT.read_bytes())
        verify(inputs / 'inputs.json', old['inputs_sha256'])
        artifacts = [(inputs / 'inputs.json', old['inputs_sha256'])]
        for row in plan['population']:
            artifacts.extend(((inputs / (row['id'] + '.npz'), row['packet_sha256']),
                              (predictions / (row['id'] + '.prediction.npz'), row['prediction_sha256'])))
        for key, filename in (('audio', 'audio'), ('zero', 'zero-audio')):
            artifacts.append((predictions / (filename + '.weights.npz'), plan['weights'][key]))
        for path, digest in artifacts:
            verify(path, digest)
            deadline()
        models = {k: load_model(predictions / (f + '.weights.npz'), plan['weights'][k])
                  for k, f in (('audio', 'audio'), ('zero', 'zero-audio'))}
        before = {k: state_hash(m) for k, m in models.items()}
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        fit_counts = Counter(r['work'] for r in plan['population'] if r['role'] == 'fit')
        require(sum(fit_counts.values()) == 20 and len(fit_counts) == 9, 'fit population differs')
        fit_gradients = {k: {term: [] for term in TERMS} for k in models}
        rows = []
        for row in plan['population']:
            context = dict(id=row['id'], checkpoint=None, completed_cases=len(rows), term='loading_inputs')
            current_transport = None
            deadline()
            with np.load(inputs / (row['id'] + '.npz'), allow_pickle=False) as packet:
                hidden, reference, valid = (packet[k] for k in ('hidden', 'reference', 'valid'))
            require(hidden.shape == (row['frames'], 512) and hidden.dtype == np.float32
                    and reference.shape == valid.shape == (row['frames'],) and valid.dtype == bool,
                    'input geometry differs')
            context['term'] = 'loading_predictions'
            with np.load(predictions / (row['id'] + '.prediction.npz'), allow_pickle=False) as saved:
                for kind, model in models.items():
                    context = dict(id=row['id'], checkpoint=kind, completed_cases=len(rows), term='loading')
                    current_transport = None
                    deadline()
                    x = hidden if kind == 'audio' else np.zeros_like(hidden)
                    payload = dict(features=torch.from_numpy(x).to('cuda')[None],
                                   reference=torch.from_numpy(reference).to('cuda')[None],
                                   valid=torch.from_numpy(valid).to('cuda')[None])
                    result, packet, parameters = audit_record(model, payload,
                        saved['fields_' + kind], saved[kind], observer=observe)
                    deadline()
                    require(state_hash(model) == before[kind], 'audit changed selected state')
                    if row['role'] == 'fit':
                        weight = 1. / (len(fit_counts) * fit_counts[row['work']])
                        for term in TERMS:
                            fit_gradients[kind][term].append(parameters[term].multiply(weight))
                    context['term'] = 'persisting_case'
                    result_row = persist_case(output, row, kind, result, packet)
                    rows.append(result_row)
                    journal(dict(event='case_completed', id=row['id'], checkpoint=kind, completed_cases=len(rows),
                                 gradient_packet_sha256=result_row['gradient_packet_sha256']))
                    deadline()
                    del payload, packet, parameters
        context = dict(id=None, checkpoint=None, completed_cases=len(rows), term='summary')
        current_transport = None
        summary = {role: {kind: summarize([r for r in rows if r['role'] == role and r['checkpoint'] == kind])
                         for kind in models} for role in ('fit', 'development', 'diagnostic')}
        attribution = json.loads((ROOT / 'experiments/phase_attribution/results-v1.json').read_bytes())
        for role in summary:
            for kind in models:
                for term, previous in (('phase', 'circular_loss'), ('count', 'count_loss'), ('total', 'total_loss')):
                    require(np.isclose(summary[role][kind]['loss.' + term]['work_macro'],
                        attribution['summary'][role]['native'][kind]['detail'][previous], atol=1e-10, rtol=1e-12),
                        'original fixed-clock native loss differs')
        aggregate = {}
        for kind in models:
            gradients = {term: accumulate(parts) for term, parts in fit_gradients[kind].items()}
            require(all(g.alignment_underflows == 0 for g in gradients.values()), 'fit aggregation lost alignment')
            aggregate[kind] = {n: comparisons({k: group(g, n) for k, g in gradients.items()}) for n in GROUPS}
        after = {k: state_hash(m) for k, m in models.items()}
        require(before == after and torch.equal(rng, torch.get_rng_state())
                and torch.equal(cuda_rng, torch.cuda.get_rng_state()), 'state/RNG changed during audit')
        require(all(p.grad is None for m in models.values() for p in m.parameters()), 'grad buffers changed')
        for path, digest in artifacts:
            verify(path, digest)
        require(plan == make_plan(), 'registered identity changed during audit')
        deadline()
        report = dict(schema='rhythm-map.gradient-audit.v1', plan_sha256=expected_plan,
                      source_sha256=plan['source_sha256'], prerequisite_sha256=plan['prerequisite_sha256'],
                      model_state_before=before, model_state_after=after, cases=rows, summary=summary,
                      fit_aggregate=aggregate, fit_work_weights={w: 1. / (len(fit_counts) * n) for w, n in fit_counts.items()},
                      actual_cnn_field_forwards=len(rows) * 4, actual_clock_forwards=len(rows) * 3,
                      training=False, optimizer_constructed=False, parameter_updates=0,
                      encoder_inference=False, holdout_access=False, production_change=False,
                      independent_acceptance=False, elapsed_before_report_s=time.monotonic() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      decision='fixed_checkpoint_gradient_audit_only_no_automatic_fit')
        write_json(output / 'report.json', report)
        report_hash = sha(output / 'report.json')
        deadline()
        journal(dict(event='audit_complete', cases=len(rows), report_sha256=report_hash,
                     elapsed_s=time.monotonic() - started, parameter_updates=0))
        deadline()
        print(json.dumps(dict(cases=len(rows), report_sha256=report_hash,
                              elapsed_s=time.monotonic() - started, parameter_updates=0)), flush=True)
    except Exception as error:
        # Checkpoints are immutable; persist diagnostics but never resume a fit.
        transport = current_transport.snapshot() if current_transport is not None else None
        torch.save(dict(case=context, transport=transport), output / 'failure-state.pt')
        write_json(output / 'failure.json', dict(case=context, exception_type=type(error).__name__,
                   message=str(error), automatic_retry=False, parameter_updates=0))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('inputs', 'predictions', 'plan', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    execute(args.inputs, args.predictions, args.plan, args.expected_plan_sha256, args.output)


if __name__ == '__main__':
    main()
