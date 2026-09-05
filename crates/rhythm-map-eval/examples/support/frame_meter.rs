//! Exact meter/count marginalization with externally shared-frame normalizers.
use anyhow::{Result, ensure};
use serde::Serialize;

const NEG: f64 = f64::NEG_INFINITY;
type Edges = Vec<Vec<(usize, f64)>>;

#[derive(Clone, Copy)]
struct State {
    meter: usize,
    phase: usize,
}

#[derive(Debug, Serialize)]
pub struct Position {
    pub meter_probabilities: Vec<f64>,
    pub downbeat_probability: f64,
}

#[derive(Debug, Serialize)]
pub struct Inference {
    pub positions: Vec<Position>,
    pub count_probabilities: Vec<f64>,
    pub log_ratio_to_reference: f64,
    pub mean_change_probability_per_bar: f64,
    pub quadrature_points: usize,
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

// Same initial phase/meter and change-rate prior as the frozen censored audit.
#[allow(clippy::cast_precision_loss)]
fn graph(minimum: usize, maximum: usize, change: f64) -> (Vec<State>, Vec<f64>, Edges) {
    let states: Vec<_> = (minimum..=maximum)
        .flat_map(|meter| (0..meter).map(move |phase| State { meter, phase }))
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
            if s.phase + 1 < s.meter {
                vec![(i + 1, 0.)]
            } else {
                states
                    .iter()
                    .enumerate()
                    .filter(|(_, next)| next.phase == 0)
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

#[allow(clippy::too_many_lines)]
fn at_rate(
    scores: &[Option<f64>],
    norm: &[f64],
    minimum: usize,
    maximum: usize,
    change: f64,
) -> Inference {
    let (states, initial, edges) = graph(minimum, maximum, change);
    let n = scores.len();
    let width = states.len();
    let counts = norm.len();
    let index = |k: usize, s: usize| k * width + s;
    let mut forward = vec![vec![NEG; counts * width]; n];
    for (s, label) in states.iter().enumerate() {
        let k = usize::from(label.phase == 0 && scores[0].is_some());
        forward[0][index(k, s)] = initial[s] + if k == 1 { scores[0].unwrap() } else { 0. };
    }
    for t in 1..n {
        for k in 0..counts {
            for (s, outgoing) in edges.iter().enumerate() {
                let mass = forward[t - 1][index(k, s)];
                if mass == NEG {
                    continue;
                }
                for &(next, prior) in outgoing {
                    let mark = usize::from(states[next].phase == 0 && scores[t].is_some());
                    if k + mark < counts {
                        let destination = index(k + mark, next);
                        let value = mass + prior + if mark == 1 { scores[t].unwrap() } else { 0. };
                        forward[t][destination] = add(forward[t][destination], value);
                    }
                }
            }
        }
    }
    // All ending phases accepted. K-dependent observation normalization is
    // charged once per path, not once per bar or once per frame.
    let mut backward = vec![vec![NEG; counts * width]; n];
    for (k, normalizer) in norm.iter().enumerate().take(counts) {
        for s in 0..width {
            backward[n - 1][index(k, s)] = -normalizer;
        }
    }
    for t in (0..n - 1).rev() {
        for k in 0..counts {
            for (s, outgoing) in edges.iter().enumerate() {
                for &(next, prior) in outgoing {
                    let mark = usize::from(states[next].phase == 0 && scores[t + 1].is_some());
                    if k + mark < counts {
                        let value = prior
                            + if mark == 1 {
                                scores[t + 1].unwrap()
                            } else {
                                0.
                            }
                            + backward[t + 1][index(k + mark, next)];
                        let target = index(k, s);
                        backward[t][target] = add(backward[t][target], value);
                    }
                }
            }
        }
    }
    let mut count_logs = vec![NEG; counts];
    for k in 0..counts {
        count_logs[k] = (0..width)
            .map(|s| forward[n - 1][index(k, s)] - norm[k])
            .fold(NEG, add);
    }
    let partition = count_logs.iter().copied().fold(NEG, add);
    let positions = (0..n)
        .map(|t| {
            let mut meter_probabilities = vec![0.; maximum - minimum + 1];
            let mut downbeat_probability = 0.;
            for (s, label) in states.iter().enumerate() {
                let probability = (0..counts)
                    .map(|k| (forward[t][index(k, s)] + backward[t][index(k, s)] - partition).exp())
                    .sum::<f64>();
                meter_probabilities[label.meter - minimum] += probability;
                if label.phase == 0 {
                    downbeat_probability += probability;
                }
            }
            Position {
                meter_probabilities,
                downbeat_probability,
            }
        })
        .collect();
    Inference {
        positions,
        count_probabilities: count_logs
            .iter()
            .map(|mass| (mass - partition).exp())
            .collect(),
        log_ratio_to_reference: partition,
        mean_change_probability_per_bar: change,
        quadrature_points: 1,
    }
}

/// Meter phase advances through missing ticks, but only observed marks enter K.
/// `norm[K]` must come from the same whole-frame domain as the caller numerator.
pub fn infer(
    values: &[Option<f64>],
    norm: &[f64],
    minimum: usize,
    maximum: usize,
) -> Result<Inference> {
    ensure!(
        !values.is_empty() && values.iter().flatten().all(|x| x.is_finite()),
        "invalid score evidence"
    );
    ensure!(
        minimum >= 2 && maximum >= minimum && maximum <= 7,
        "invalid meter domain"
    );
    ensure!(
        !norm.is_empty() && norm.iter().all(|x| x.is_finite()),
        "invalid shared normalizers"
    );
    ensure!(
        norm.len()
            == values
                .iter()
                .flatten()
                .count()
                .min(values.len().div_ceil(minimum))
                + 1,
        "incomplete count domain"
    );
    let points = usize::midpoint(values.len() / 2, 3);
    let mut components = Vec::new();
    let mut partition = NEG;
    for (change, weight) in quadrature(points) {
        let component = at_rate(values, norm, minimum, maximum, change);
        let mass = component.log_ratio_to_reference + weight.ln();
        partition = add(partition, mass);
        components.push((mass, component));
    }
    let mut result = Inference {
        positions: (0..values.len())
            .map(|_| Position {
                meter_probabilities: vec![0.; maximum - minimum + 1],
                downbeat_probability: 0.,
            })
            .collect(),
        count_probabilities: vec![0.; norm.len()],
        log_ratio_to_reference: partition,
        mean_change_probability_per_bar: 0.,
        quadrature_points: points,
    };
    for (log_mass, component) in components {
        let mass = (log_mass - partition).exp();
        result.mean_change_probability_per_bar += mass * component.mean_change_probability_per_bar;
        for (out, input) in result.positions.iter_mut().zip(component.positions) {
            out.downbeat_probability += mass * input.downbeat_probability;
            for (a, b) in out
                .meter_probabilities
                .iter_mut()
                .zip(input.meter_probabilities)
            {
                *a += mass * b;
            }
        }
        for (out, input) in result
            .count_probabilities
            .iter_mut()
            .zip(component.count_probabilities)
        {
            *out += mass * input;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn missing_mark_count_and_shared_terminal_factor_match_enumeration() {
        let labels = [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)];
        let marks = [Some(0.3_f64), None, Some(-0.4), Some(0.7), Some(-0.1)];
        let norm = [0., 0.2, -0.3, 0.7];
        let mut paths: Vec<_> = (0..5)
            .map(|s| (vec![s], 1. / (2 * labels[s].0) as f64, 0, 0))
            .collect();
        for t in 0..marks.len() - 1 {
            let mut next = Vec::new();
            for (path, weight, changes, stays) in paths {
                let s = *path.last().unwrap();
                let options: &[usize] = match s {
                    0 => &[1],
                    2 => &[3],
                    3 => &[4],
                    _ => &[0, 2],
                };
                for &j in options {
                    let boundary = labels[s].1 + 1 == labels[s].0;
                    let changed = boundary && labels[s].0 != labels[j].0;
                    let mut extended = path.clone();
                    extended.push(j);
                    next.push((
                        extended,
                        weight,
                        changes + usize::from(changed),
                        stays + usize::from(boundary && !changed),
                    ));
                }
            }
            paths = next;
            assert!(paths.iter().all(|(p, _, _, _)| p.len() == t + 2));
        }
        let factorial = |n: usize| (1..=n).map(|i| i as f64).product::<f64>();
        let mut mass = 0.;
        let mut moment = 0.;
        let mut count_mass = [0.; 4];
        let mut occupancy = [[0.; 5]; 5];
        for (path, prior, changes, stays) in paths {
            let selected: Vec<_> = path
                .iter()
                .zip(marks)
                .filter_map(|(&s, m)| if labels[s].1 == 0 { m } else { None })
                .collect();
            let k = selected.len();
            let weight = prior
                * (selected.iter().sum::<f64>() - norm[k]).exp()
                * factorial(changes)
                * factorial(stays)
                / factorial(changes + stays + 1);
            mass += weight;
            moment += weight * (changes + 1) as f64 / (changes + stays + 2) as f64;
            count_mass[k] += weight;
            for (t, s) in path.into_iter().enumerate() {
                occupancy[t][s] += weight;
            }
        }
        let actual = infer(&marks, &norm, 2, 3).unwrap();
        assert!((actual.log_ratio_to_reference - mass.ln()).abs() < 1e-12);
        assert!((actual.mean_change_probability_per_bar - moment / mass).abs() < 1e-12);
        for (p, counts) in actual.positions.iter().zip(occupancy) {
            assert!((p.downbeat_probability - (counts[0] + counts[2]) / mass).abs() < 1e-12);
            assert!((p.meter_probabilities[0] - (counts[0] + counts[1]) / mass).abs() < 1e-12);
        }
        for (a, b) in actual.count_probabilities.iter().zip(count_mass) {
            assert!((a - b / mass).abs() < 1e-12);
        }
        assert!(infer(&marks, &[0.], 2, 3).is_err());
        let missing = infer(&[None; 5], &[0.], 2, 7).unwrap();
        assert!(missing.log_ratio_to_reference.abs() < 1e-12);
        assert!((missing.mean_change_probability_per_bar - 0.5).abs() < 1e-12);
    }
}
