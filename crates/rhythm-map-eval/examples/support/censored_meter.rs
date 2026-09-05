//! Conditional meter inference with censored first/last bars; evaluation only.
use anyhow::{Result, ensure};
use serde::Serialize;

const NEG: f64 = f64::NEG_INFINITY;
type Edges = Vec<Vec<(usize, f64)>>;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct State {
    pub meter: usize,
    pub beat_in_bar: usize,
}

#[derive(Debug, Serialize)]
pub struct Position {
    pub map_state: State,
    pub meter_probabilities: Vec<f64>,
    pub downbeat_probability: f64,
}

#[derive(Debug, Serialize)]
pub struct Inference {
    pub states: Vec<State>,
    pub positions: Vec<Position>,
    pub log_ratio_to_reference: f64,
    pub map_log_probability: f64,
    pub expected_bar_starts: f64,
}

fn add(a: f64, b: f64) -> f64 {
    if a == NEG {
        return b;
    }
    if b == NEG {
        return a;
    }
    a.max(b) + (-(a - b).abs()).exp().ln_1p()
}

#[allow(clippy::cast_precision_loss)]
fn graph(minimum: usize, maximum: usize, change: f64) -> (Vec<State>, Vec<f64>, Edges) {
    let states: Vec<State> = (minimum..=maximum)
        .flat_map(|meter| (0..meter).map(move |beat_in_bar| State { meter, beat_in_bar }))
        .collect();
    let choices = (maximum - minimum + 1) as f64;
    let initial = states
        .iter()
        .map(|s| -(choices * s.meter as f64).ln())
        .collect();
    let edges = states
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if s.beat_in_bar + 1 < s.meter {
                vec![(i + 1, 0.)]
            } else {
                states
                    .iter()
                    .enumerate()
                    .filter(|(_, next)| next.beat_in_bar == 0)
                    .map(|(j, next)| {
                        (
                            j,
                            if maximum == minimum {
                                0.
                            } else if next.meter == s.meter {
                                (-change).ln_1p()
                            } else {
                                (change / (choices - 1.)).ln()
                            },
                        )
                    })
                    .collect()
            }
        })
        .collect();
    (states, initial, edges)
}

/// Each input is a likelihood ratio for a downbeat mark on an already inferred
/// beat cell. A non-downbeat uses the same cell reference (ratio one). This is
/// a different observation model from whole-bar cyclic scoring. Initial phase
/// is uniform conditional on meter. All terminal states are accepted with mass
/// one: an unfinished final bar is not forced to become a shorter complete bar.
#[allow(clippy::too_many_lines)]
fn infer_at_rate(
    log_marks: &[f64],
    minimum: usize,
    maximum: usize,
    change: f64,
) -> Result<Inference> {
    ensure!(
        !log_marks.is_empty() && log_marks.iter().all(|v| v.is_finite()),
        "invalid mark evidence"
    );
    ensure!(
        minimum >= 2 && maximum >= minimum && maximum <= 7,
        "invalid meter domain"
    );
    let (states, initial, edges) = graph(minimum, maximum, change);
    let count = states.len();
    let n = log_marks.len();
    let emission = |t: usize, s: usize| {
        if states[s].beat_in_bar == 0 {
            log_marks[t]
        } else {
            0.
        }
    };
    let mut forward = vec![vec![NEG; count]; n];
    let mut best = initial.clone();
    let mut previous = vec![vec![0; count]; n];
    for s in 0..count {
        forward[0][s] = initial[s] + emission(0, s);
        best[s] = forward[0][s];
    }
    for t in 1..n {
        let mut next_best = vec![NEG; count];
        for s in 0..count {
            for &(next, prior) in &edges[s] {
                forward[t][next] = add(forward[t][next], forward[t - 1][s] + prior);
                let candidate = best[s] + prior;
                if candidate > next_best[next] {
                    next_best[next] = candidate;
                    previous[t][next] = s;
                }
            }
        }
        for s in 0..count {
            forward[t][s] += emission(t, s);
            next_best[s] += emission(t, s);
        }
        best = next_best;
    }
    let partition = forward[n - 1].iter().copied().fold(NEG, add);
    let mut state = (0..count)
        .max_by(|&a, &b| best[a].total_cmp(&best[b]))
        .unwrap();
    let map_log_probability = best[state] - partition;
    let mut map = vec![0; n];
    for t in (0..n).rev() {
        map[t] = state;
        state = previous[t][state];
    }
    let mut backward = vec![vec![0.; count]; n];
    for t in (0..n - 1).rev() {
        for (s, outgoing) in edges.iter().enumerate() {
            backward[t][s] = outgoing
                .iter()
                .map(|&(next, prior)| prior + emission(t + 1, next) + backward[t + 1][next])
                .fold(NEG, add);
        }
    }
    let mut positions = Vec::new();
    let mut expected_bar_starts = 0.;
    for t in 0..n {
        let mut meter_probabilities = vec![0.; maximum - minimum + 1];
        let mut downbeat_probability = 0.;
        for (s, label) in states.iter().enumerate() {
            let probability = (forward[t][s] + backward[t][s] - partition).exp();
            meter_probabilities[label.meter - minimum] += probability;
            if label.beat_in_bar == 0 {
                downbeat_probability += probability;
            }
        }
        expected_bar_starts += downbeat_probability;
        positions.push(Position {
            map_state: states[map[t]],
            meter_probabilities,
            downbeat_probability,
        });
    }
    Ok(Inference {
        states,
        positions,
        log_ratio_to_reference: partition,
        map_log_probability,
        expected_bar_starts,
    })
}

#[derive(Debug, Serialize)]
pub struct MarginalPosition {
    pub meter_probabilities: Vec<f64>,
    pub downbeat_probability: f64,
}

#[derive(Debug, Serialize)]
pub struct Marginalized {
    pub positions: Vec<MarginalPosition>,
    pub log_ratio_to_reference: f64,
    pub mean_change_probability_per_bar: f64,
    pub quadrature_points: usize,
}

/// Gauss-Legendre quadrature on [0,1]. At Q points it integrates polynomials
/// through degree 2Q-1 exactly, up to floating-point arithmetic.
#[allow(clippy::cast_precision_loss)]
fn quadrature(points: usize) -> Vec<(f64, f64)> {
    let polynomial = |x: f64| {
        let (mut previous, mut current) = (1., x);
        for k in 2..=points {
            let next = ((2 * k - 1) as f64 * x * current - (k - 1) as f64 * previous) / k as f64;
            previous = current;
            current = next;
        }
        (
            current,
            points as f64 * (x * current - previous) / (x * x - 1.),
        )
    };
    (0..points)
        .map(|i| {
            let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (points as f64 + 0.5)).cos();
            for _ in 0..64 {
                let (value, derivative) = polynomial(x);
                let step = value / derivative;
                x -= step;
                if step.abs() < 1e-14 {
                    break;
                }
            }
            let (_, derivative) = polynomial(x);
            (
                f64::midpoint(x, 1.),
                1. / ((1. - x * x) * derivative * derivative),
            )
        })
        .collect()
}

/// Integrate one unknown, run-wide meter-change probability with a uniform
/// Beta(1,1) prior. The integrand is polynomial: at most ceil((N-1)/2) bar
/// boundaries are visible. The quadrature order also integrates its first moment.
/// Returns marginals; independently best positions need not form a legal path.
pub fn infer(log_marks: &[f64], minimum: usize, maximum: usize) -> Result<Marginalized> {
    ensure!(!log_marks.is_empty(), "empty mark evidence");
    let points = usize::midpoint(log_marks.len() / 2, 3);
    let mut components = Vec::new();
    let mut partition = NEG;
    for (change, weight) in quadrature(points) {
        let component = infer_at_rate(log_marks, minimum, maximum, change)?;
        let log_mass = component.log_ratio_to_reference + weight.ln();
        partition = add(partition, log_mass);
        components.push((change, log_mass, component));
    }
    let mut positions: Vec<MarginalPosition> = (0..log_marks.len())
        .map(|_| MarginalPosition {
            meter_probabilities: vec![0.; maximum - minimum + 1],
            downbeat_probability: 0.,
        })
        .collect();
    let mut mean_change = 0.;
    for (change, log_mass, component) in components {
        let mass = (log_mass - partition).exp();
        mean_change += mass * change;
        for (out, input) in positions.iter_mut().zip(component.positions) {
            out.downbeat_probability += mass * input.downbeat_probability;
            for (a, b) in out
                .meter_probabilities
                .iter_mut()
                .zip(input.meter_probabilities)
            {
                *a += mass * b;
            }
        }
    }
    Ok(Marginalized {
        positions,
        log_ratio_to_reference: partition,
        mean_change_probability_per_bar: mean_change,
        quadrature_points: points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_evidence_has_unit_mass_for_every_crop_length() {
        for length in 1..=48 {
            let r = infer(&vec![0.; length], 2, 7).unwrap();
            assert!(r.log_ratio_to_reference.abs() < 1e-12);
            for p in r.positions {
                assert!((p.meter_probabilities.iter().sum::<f64>() - 1.).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn neutral_future_does_not_change_past_marginals() {
        let marks = [3., -1., -1., -1., 3., -1.];
        let prefix = infer(&marks, 2, 7).unwrap();
        let mut extended = marks.to_vec();
        extended.extend([0.; 20]);
        let longer = infer(&extended, 2, 7).unwrap();
        assert!((prefix.log_ratio_to_reference - longer.log_ratio_to_reference).abs() < 1e-12);
        for (a, b) in prefix.positions.iter().zip(&longer.positions) {
            for (x, y) in a.meter_probabilities.iter().zip(&b.meter_probabilities) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn forward_backward_and_map_match_independent_path_enumeration() {
        fn visit(
            t: usize,
            path: &mut Vec<usize>,
            weight: f64,
            marks: &[f64],
            totals: &mut (f64, f64, Vec<Vec<f64>>),
        ) {
            // Explicit states: 2/0, 2/1, 3/0, 3/1, 3/2.
            let s = *path.last().unwrap();
            let weighted = weight * if s == 0 || s == 2 { marks[t].exp() } else { 1. };
            if t + 1 == marks.len() {
                totals.0 += weighted;
                totals.1 = totals.1.max(weighted);
                for (i, &j) in path.iter().enumerate() {
                    totals.2[i][j] += weighted;
                }
                return;
            }
            let next: &[(usize, f64)] = match s {
                0 => &[(1, 1.)],
                2 => &[(3, 1.)],
                3 => &[(4, 1.)],
                _ => &[(0, 0.5), (2, 0.5)],
            };
            for &(j, prior) in next {
                path.push(j);
                visit(t + 1, path, weighted * prior, marks, totals);
                path.pop();
            }
        }
        let marks = [1.2, -0.7, 0.3, 2., -1.];
        let mut totals = (0., 0_f64, vec![vec![0.; 5]; marks.len()]);
        for (s, probability) in [0.25, 0.25, 1. / 6., 1. / 6., 1. / 6.]
            .into_iter()
            .enumerate()
        {
            visit(0, &mut vec![s], probability, &marks, &mut totals);
        }
        let r = infer_at_rate(&marks, 2, 3, 0.5).unwrap();
        assert!((r.log_ratio_to_reference - totals.0.ln()).abs() < 1e-12);
        assert!((r.map_log_probability - (totals.1 / totals.0).ln()).abs() < 1e-12);
        for (t, p) in r.positions.iter().enumerate() {
            assert!(
                (p.meter_probabilities[0] - (totals.2[t][0] + totals.2[t][1]) / totals.0).abs()
                    < 1e-12
            );
            assert!(
                (p.downbeat_probability - (totals.2[t][0] + totals.2[t][2]) / totals.0).abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert!(infer(&[], 2, 7).is_err());
        assert!(infer(&[f64::NAN], 2, 7).is_err());
        assert!(infer(&[0.], 1, 7).is_err());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn quadrature_integrates_every_required_moment() {
        for points in 1..=20 {
            let nodes = quadrature(points);
            for degree in 0..2 * points {
                let actual: f64 = nodes
                    .iter()
                    .map(|(x, w)| w * x.powi(i32::try_from(degree).unwrap()))
                    .sum();
                assert!((actual - 1. / (degree + 1) as f64).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn uniform_hyperprior_has_mean_half_without_evidence() {
        for length in [1, 4, 17, 48] {
            let r = infer(&vec![0.; length], 2, 7).unwrap();
            assert!((r.mean_change_probability_per_bar - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn integrated_marginals_match_beta_weighted_path_enumeration() {
        fn beta(changes: usize, stays: usize) -> f64 {
            let factorial = |n: usize| (1..=n).map(|x| x as f64).product::<f64>();
            factorial(changes) * factorial(stays) / factorial(changes + stays + 1)
        }
        // Deliberately independent of graph(), forward/backward and quadrature.
        let labels = [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)];
        let marks = [1.2_f64, -0.7, 0.3, 2., -1., 0.8, -0.4];
        let mut paths: Vec<_> = (0..5)
            .map(|s| (vec![s], 1. / (2 * labels[s].0) as f64, 0, 0))
            .collect();
        for (t, mark) in marks.into_iter().enumerate() {
            for (path, weight, _, _) in &mut paths {
                if labels[*path.last().unwrap()].1 == 0 {
                    *weight *= mark.exp();
                }
            }
            if t + 1 == marks.len() {
                break;
            }
            let mut next_paths = Vec::new();
            for (path, weight, changes, stays) in paths {
                let s = *path.last().unwrap();
                let (meter, phase) = labels[s];
                let next: &[usize] = match s {
                    0 => &[1],
                    2 => &[3],
                    3 => &[4],
                    _ => &[0, 2],
                };
                for &j in next {
                    let boundary = phase + 1 == meter;
                    let changed = boundary && labels[j].0 != meter;
                    let mut extended = path.clone();
                    extended.push(j);
                    next_paths.push((
                        extended,
                        weight,
                        changes + usize::from(changed),
                        stays + usize::from(boundary && !changed),
                    ));
                }
            }
            paths = next_paths;
        }
        let mut partition = 0.;
        let mut first_moment = 0.;
        let mut occupancy = [[0.; 5]; 7];
        for (path, weight, changes, stays) in paths {
            let mass = weight * beta(changes, stays);
            partition += mass;
            first_moment += weight * beta(changes + 1, stays);
            for (t, s) in path.into_iter().enumerate() {
                occupancy[t][s] += mass;
            }
        }
        let actual = infer(&marks, 2, 3).unwrap();
        assert!((actual.log_ratio_to_reference - partition.ln()).abs() < 1e-12);
        assert!((actual.mean_change_probability_per_bar - first_moment / partition).abs() < 1e-12);
        for (p, counts) in actual.positions.iter().zip(occupancy) {
            assert!((p.meter_probabilities[0] - (counts[0] + counts[1]) / partition).abs() < 1e-12);
            assert!((p.downbeat_probability - (counts[0] + counts[2]) / partition).abs() < 1e-12);
        }
    }

    #[test]
    fn single_meter_and_invalid_domains() {
        let r = infer(&[1., 0., 0., 0., 1., 0.], 4, 4).unwrap();
        for p in r.positions {
            assert!((p.meter_probabilities[0] - 1.).abs() < 1e-12);
        }
        assert!((r.mean_change_probability_per_bar - 0.5).abs() < 1e-12);
        assert!(infer(&[0.], 7, 2).is_err());
        assert!(infer(&[0.], 2, 8).is_err());
    }
}
