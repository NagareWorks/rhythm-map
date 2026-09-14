"""Pure vector accounting; descriptive directions, never optimizer routing."""
import numpy as np

from .data import require


def vector(value):
    value = np.asarray(value, dtype=np.float64)
    require(value.ndim == 1 and len(value) > 0 and np.isfinite(value).all(), 'invalid gradient vector')
    return value


def relation(left, right):
    left, right = vector(left), vector(right)
    require(left.shape == right.shape, 'parameter vector geometry differs')
    a, b = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    dot = float(np.dot(left, right))
    return dict(left_norm=a, right_norm=b, dot=dot, cosine=dot/a/b if a and b else None)


def descent(development, training):
    result = relation(development, training)
    result['negative_gradient_derivative'] = -result['dot']
    result['unit_negative_gradient_derivative'] = -result['dot']/result['right_norm'] if result['right_norm'] else None
    return result


def observed(start_gradient, end_gradient, delta, before, after):
    require(np.isfinite([before, after]).all(), 'invalid endpoint loss')
    left, right = relation(start_gradient, delta), relation(end_gradient, delta)
    actual = float(after-before)
    estimate = .5*(left['dot']+right['dot'])
    return dict(delta_norm=left['right_norm'], actual_mse_change=actual,
                start_linear=left['dot'], end_linear=right['dot'], endpoint_trapezoid=estimate,
                trapezoid_residual=actual-estimate)


def aggregates(vectors, development):
    primitives = ('native', 'pair_identity', 'pair_global', 'pair_local', 'retention')
    training = {k: vector(vectors[k]) for k in primitives}
    training['pair'] = training['pair_identity']+training['pair_global']+training['pair_local']
    training['native_pair'] = training['native']+training['pair']
    training['native_pair_retention'] = training['native_pair']+training['retention']
    targets, losses = {}, {}
    for i, row in enumerate(development):
        targets['record:'+row['id']] = vector(vectors['dev_'+str(i)])
        losses['record:'+row['id']] = row['mse']
    works = sorted({r['work'] for r in development})
    require(len(development) > 0, 'missing development population')
    for work in works:
        names = ['record:'+r['id'] for r in development if r['work'] == work]
        targets['work:'+work] = np.mean([targets[n] for n in names], axis=0)
        losses['work:'+work] = float(np.mean([losses[n] for n in names]))
    targets['macro'] = np.mean([targets['work:'+w] for w in works], axis=0)
    losses['macro'] = float(np.mean([losses['work:'+w] for w in works]))
    return training, targets, losses


def summarize(vectors, development):
    training, targets, _ = aggregates(vectors, development)
    return dict(training_norms={k: float(np.linalg.norm(v)) for k, v in training.items()},
                native_pair=relation(training['native'], training['pair']),
                retention_fit_direction=relation(training['retention'], training['native_pair']),
                development={k: {term: descent(v, g) for term, g in training.items()} for k, v in targets.items()})
