"""Model-free design arithmetic and conservative candidate-vote semantics.

No model, data acquisition, annotation collection, fit, or readiness decision.
Distinct listener IDs and matching view hashes do not prove real independence.
"""
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def proposal():
    return json.loads((HERE / 'clock-readout-proposal-v1.json').read_bytes())


def query_lengths(frames, window=200, hop=100):
    if any(type(v) is not int or v <= 0 for v in (frames, window, hop)) or hop > window:
        raise ValueError('invalid query geometry')
    starts = [0]
    while starts[-1] + window < frames:
        starts.append(starts[-1] + hop)
    return [min(window, frames - start) for start in starts]


def budget():
    """Exact symbolic counts for the one proposed architecture, not benchmarks."""
    spec = proposal()
    a, q = spec['architecture'], spec['query']
    width, incoming = a['shared_width'], a['encoder_width']
    blocks, kernel = len(a['dilations']), a['kernel_size']
    fused = width + a['clock_fields'] + a['coverage_fields']
    projection = incoming * width + width
    temporal = blocks * (width * kernel + width + width * width + width)
    fusion = fused * width + width
    output = width * a['output_logits'] + a['output_logits']
    radius = (kernel - 1) * sum(a['dilations']) // 2
    lengths = query_lengths(q['example_owned_frames'], q['frames'], q['hop_frames'])
    shared_mac_per_frame = incoming * width + blocks * (width * kernel + width * width)
    fusion_mac = sum(lengths) * q['candidate_cap'] * fused * width
    output_mac = len(lengths) * q['candidate_cap'] * width * a['output_logits']
    parameters = projection + temporal + fusion + output
    return dict(projection_parameters=projection, temporal_parameters=temporal,
        fusion_parameters=fusion, output_parameters=output, parameters=parameters,
        weight_bytes=parameters * a['bytes_per_parameter'], radius_frames=radius,
        receptive_field_frames=2 * radius + 1, center_span_seconds=2 * radius / a['frame_rate_hz'],
        example_queries=len(lengths), example_query_frames=sum(lengths),
        example_multiply_accumulates=q['example_owned_frames'] * shared_mac_per_frame + fusion_mac + output_mac,
        feature_chunk_bytes=q['feature_chunk_frames'] * incoming * a['bytes_per_parameter'],
        feature_chunk_with_halo_bytes=(q['feature_chunk_frames'] + 2 * radius) * incoming * a['bytes_per_parameter'],
        runtime_measured=False)


def consensus(candidate_ids, votes):
    """Convert two complete same-view votes to per-clock targets, not one class.

    Unknown/disagreement have no target. Multiple positives and all negatives
    are valid; neither denotes global tempo, rhythm presence or training consent.
    """
    spec = proposal()
    if not isinstance(candidate_ids, (tuple, list)) or not 1 <= len(candidate_ids) <= spec['query']['candidate_cap']:
        raise ValueError('invalid candidate population')
    if any(type(k) is not str or not k.strip() for k in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('candidate identifiers must be distinct')
    if not isinstance(votes, (tuple, list)) or len(votes) != 2 or any(type(v) is not dict for v in votes):
        raise ValueError('two complete independent vote records required')
    listeners, views = [], []
    for vote in votes:
        if set(vote) != {'listener_id', 'view_sha256', 'labels'}:
            raise ValueError('invalid vote fields')
        listener, view, labels = vote['listener_id'], vote['view_sha256'], vote['labels']
        if type(listener) is not str or not listener.strip() or type(view) is not str or not re.fullmatch(r'[0-9a-f]{64}', view):
            raise ValueError('invalid listener/view identity')
        if type(labels) is not dict or set(labels) != set(candidate_ids) or any(v not in spec['labels'] for v in labels.values()):
            raise ValueError('incomplete or invalid labels')
        listeners.append(listener)
        views.append(view)
    if listeners[0] == listeners[1] or views[0] != views[1]:
        raise ValueError('duplicate listener or mismatched blinded view')
    targets, reasons = {}, {}
    for key in candidate_ids:
        a, b = (v['labels'][key] for v in votes)
        targets[key] = 1 if a == b == 'supported' else 0 if a == b == 'contradicted' else None
        reasons[key] = 'agreement' if targets[key] is not None else 'both_uncertain' if a == b else 'no_consensus'
    return dict(targets=targets, reasons=reasons, known=sum(v is not None for v in targets.values()),
                unknown=sum(v is None for v in targets.values()))


if __name__ == '__main__':
    print(json.dumps(budget(), indent=2))
