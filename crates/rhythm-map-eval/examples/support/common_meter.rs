//! Conditional meter inference against a common observed-score permutation null.
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

/// log mean exp(sum scores on a K-subset), for each K, on the SAME observed bag.
/// A whole-path numerator minus this quantity averages to ratio one over all
/// permutations under the exchangeable-score null. No independent-cell product
/// or observed-silence imputation is used. Scores should be offset-centered.
#[allow(clippy::cast_precision_loss)]
pub fn normalizers(scores: &[f64]) -> Vec<f64> {
    let n = scores.len();
    let mut elementary = vec![NEG; n + 1];
    elementary[0] = 0.;
    for (t, score) in scores.iter().enumerate() {
        for k in (1..=t + 1).rev() {
            elementary[k] = add(elementary[k], elementary[k - 1] + score);
        }
    }
    let mut log_choose = 0.;
    for (k, value) in elementary.iter_mut().enumerate().skip(1) {
        log_choose += ((n + 1 - k) as f64).ln() - (k as f64).ln();
        *value -= log_choose;
    }
    elementary
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
fn at_rate(scores: &[f64], norm: &[f64], minimum: usize, maximum: usize, change: f64) -> Inference {
    let (states, initial, edges) = graph(minimum, maximum, change);
    let n = scores.len();
    let width = states.len();
    let counts = n.div_ceil(minimum) + 1;
    let index = |k: usize, s: usize| k * width + s;
    let mut forward = vec![vec![NEG; counts * width]; n];
    for (s, label) in states.iter().enumerate() {
        let k = usize::from(label.phase == 0);
        forward[0][index(k, s)] = initial[s] + if k == 1 { scores[0] } else { 0. };
    }
    for t in 1..n {
        for k in 0..counts {
            for (s, outgoing) in edges.iter().enumerate() {
                let mass = forward[t - 1][index(k, s)];
                if mass == NEG {
                    continue;
                }
                for &(next, prior) in outgoing {
                    let mark = usize::from(states[next].phase == 0);
                    if k + mark < counts {
                        let destination = index(k + mark, next);
                        let value = mass + prior + if mark == 1 { scores[t] } else { 0. };
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
                    let mark = usize::from(states[next].phase == 0);
                    if k + mark < counts {
                        let value = prior
                            + if mark == 1 { scores[t + 1] } else { 0. }
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

/// Finite observed scores only. Unobserved edge beats are absent, not zeroes.
/// Conditional on a supplied clock, not calibrated confidence or clock search.
pub fn infer(values: &[f64], minimum: usize, maximum: usize) -> Result<Inference> {
    ensure!(
        !values.is_empty() && values.iter().all(|x| x.is_finite()),
        "invalid score evidence"
    );
    ensure!(
        minimum >= 2 && maximum >= minimum && maximum <= 7,
        "invalid meter domain"
    );
    let maximum_score = values.iter().copied().fold(NEG, f64::max);
    let scores: Vec<f64> = values.iter().map(|s| s - maximum_score).collect();
    ensure!(scores.iter().all(|x| x.is_finite()), "score range overflow");
    let norm = normalizers(&scores);
    ensure!(norm.iter().all(|x| x.is_finite()), "normalizer overflow");
    let points = usize::midpoint(values.len() / 2, 3);
    let mut components = Vec::new();
    let mut partition = NEG;
    for (change, weight) in quadrature(points) {
        let component = at_rate(&scores, &norm, minimum, maximum, change);
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
        count_probabilities: vec![0.; values.len().div_ceil(minimum) + 1],
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
    fn every_subset_count_has_exact_unit_reference_mass() {
        let scores = [-2_f64, -1., -4., -3., -0.5];
        let norm = normalizers(&scores);
        for (k, normalizer) in norm.iter().enumerate() {
            let subsets: Vec<_> = (0_u32..32)
                .filter(|bits| bits.count_ones() as usize == k)
                .map(|bits| {
                    scores
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| bits & (1 << i) != 0)
                        .map(|(_, x)| x)
                        .sum::<f64>()
                })
                .collect();
            let mean = subsets
                .iter()
                .map(|sum| (sum - normalizer).exp())
                .sum::<f64>()
                / subsets.len() as f64;
            assert!((mean - 1.).abs() < 1e-12);
        }
    }

    #[test]
    fn extra_unsupported_bar_starts_lose_common_context_evidence() {
        let scores: Vec<_> = (0..48)
            .map(|i| if i % 4 == 0 { -2. } else { -8. })
            .collect();
        let norm = normalizers(&scores);
        let four = scores.iter().step_by(4).sum::<f64>() - norm[12];
        let two = scores.iter().step_by(2).sum::<f64>() - norm[24];
        assert!(four > two + 5.);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn flat_inputs_remain_unit_evidence_and_counts_match_marginals() {
        for n in [1, 2, 7, 17, 48] {
            let r = infer(&vec![-8.; n], 2, 7).unwrap();
            assert!(r.log_ratio_to_reference.abs() < 1e-11);
            assert!((r.mean_change_probability_per_bar - 0.5).abs() < 1e-11);
            assert!((r.count_probabilities.iter().sum::<f64>() - 1.).abs() < 1e-11);
            let expected = r
                .count_probabilities
                .iter()
                .enumerate()
                .map(|(k, p)| k as f64 * p)
                .sum::<f64>();
            let sum = r
                .positions
                .iter()
                .map(|p| p.downbeat_probability)
                .sum::<f64>();
            assert!((expected - sum).abs() < 1e-10);
        }
    }

    #[test]
    fn offsets_do_not_change_inference_and_invalid_inputs_fail() {
        let x = [-2., -8., -8., -8., -2., -8., -8.];
        let a = infer(&x, 2, 7).unwrap();
        let b = infer(&x.map(|v| v + 100.), 2, 7).unwrap();
        assert!((a.log_ratio_to_reference - b.log_ratio_to_reference).abs() < 1e-12);
        for (a, b) in a.positions.iter().zip(b.positions) {
            assert!((a.downbeat_probability - b.downbeat_probability).abs() < 1e-12);
        }
        assert!(infer(&[], 2, 7).is_err());
        assert!(infer(&[f64::NAN], 2, 7).is_err());
        assert!(infer(&[0.], 7, 2).is_err());
        assert!(infer(&[f64::MAX, -f64::MAX], 2, 7).is_err());
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    fn integrated_counted_graph_matches_independent_path_and_subset_enumeration() {
        fn beta(changes: usize, stays: usize) -> f64 {
            let factorial = |n: usize| (1..=n).map(|x| x as f64).product::<f64>();
            factorial(changes) * factorial(stays) / factorial(changes + stays + 1)
        }
        // Explicit states and subset enumeration: no graph, DP, quadrature or
        // elementary-symmetric normalizer is shared with the implementation.
        let labels = [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)];
        let marks = [1.2_f64, -0.7, 0.3, 2., -1., 0.8, -0.4];
        let mut means = [0.; 8];
        let mut subsets = [0.; 8];
        for mask in 0_u32..128 {
            let k = mask.count_ones() as usize;
            means[k] += marks
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, x)| x)
                .sum::<f64>()
                .exp();
            subsets[k] += 1.;
        }
        for (value, count) in means.iter_mut().zip(subsets) {
            *value /= count;
        }
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
        let (mut partition, mut first_moment) = (0., 0.);
        let mut occupancy = [[0.; 5]; 7];
        let mut counts = [0.; 5];
        for (path, weight, changes, stays) in paths {
            let k = path.iter().filter(|&&s| labels[s].1 == 0).count();
            let mass = weight * beta(changes, stays) / means[k];
            partition += mass;
            first_moment += weight * beta(changes + 1, stays) / means[k];
            counts[k] += mass;
            for (t, s) in path.into_iter().enumerate() {
                occupancy[t][s] += mass;
            }
        }
        let actual = infer(&marks, 2, 3).unwrap();
        assert!((actual.log_ratio_to_reference - partition.ln()).abs() < 1e-12);
        assert!((actual.mean_change_probability_per_bar - first_moment / partition).abs() < 1e-12);
        for (a, b) in actual.count_probabilities.iter().zip(counts) {
            assert!((a - b / partition).abs() < 1e-12);
        }
        for (p, counts) in actual.positions.iter().zip(occupancy) {
            assert!((p.meter_probabilities[0] - (counts[0] + counts[1]) / partition).abs() < 1e-12);
            assert!((p.downbeat_probability - (counts[0] + counts[2]) / partition).abs() < 1e-12);
        }
    }

    #[test]
    fn missing_edges_are_not_imputed_as_observed_background() {
        let prefix = [-2., -8., -8., -8., -2., -8., -8.];
        let a = infer(&prefix, 2, 7).unwrap();
        let mut observed_background = prefix.to_vec();
        observed_background.extend([-8.; 20]);
        let b = infer(&observed_background, 2, 7).unwrap();
        assert_eq!(a.positions.len(), prefix.len());
        assert_eq!(b.positions.len(), observed_background.len());
        // Padding changes the comparison bag, even if mixtures can happen to
        // have similar aggregate evidence. It is not an allowed missing-data operation.
        assert!((normalizers(&prefix)[2] - normalizers(&observed_background)[2]).abs() > 1e-8);
        let single = infer(&prefix, 4, 4).unwrap();
        for p in single.positions {
            assert!((p.meter_probabilities[0] - 1.).abs() < 1e-12);
        }
        assert!((single.mean_change_probability_per_bar - 0.5).abs() < 1e-12);
    }
}
