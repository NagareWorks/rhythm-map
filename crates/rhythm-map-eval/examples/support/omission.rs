//! Exact omission semantics conditional on supplied clocks and constant meters.
use super::frames::{Table, add};
use anyhow::{Result, ensure};
use serde::Serialize;

const NEG: f64 = f64::NEG_INFINITY;

#[allow(clippy::cast_precision_loss)]
fn log_beta(success: usize, failure: usize) -> f64 {
    let factorial = |n: usize| (1..=n).map(|i| (i as f64).ln()).sum::<f64>();
    factorial(success) + factorial(failure) - factorial(success + failure + 1)
}

/// A polynomial over retained ticks K and emitted accents D. At an ordinary
/// tick only labels 0/1 exist; a latent bar start also allows label 2.
struct Factor {
    values: Vec<[f64; 2]>,
    stride: usize,
    sum: Vec<Vec<f64>>,
    best: Vec<Vec<f64>>,
    choice: Vec<Vec<u8>>,
}

impl Factor {
    fn new(values: Vec<[f64; 2]>, accents: bool) -> Self {
        let n = values.len();
        let stride = if accents { n + 1 } else { 1 };
        let width = (n + 1) * stride;
        let mut sum = vec![vec![NEG; width]; n + 1];
        let mut best = sum.clone();
        let mut choice = vec![vec![0; width]; n + 1];
        sum[0][0] = 0.;
        best[0][0] = 0.;
        for (t, pair) in values.iter().enumerate() {
            for k in 0..=t {
                for d in 0..=k.min(stride - 1) {
                    let from = k * stride + d;
                    for label in 0..=if accents { 2 } else { 1 } {
                        let next =
                            (k + usize::from(label > 0)) * stride + d + usize::from(label == 2);
                        let score = if label == 0 {
                            0.
                        } else {
                            pair[0] + if label == 2 { pair[1] } else { 0. }
                        };
                        sum[t + 1][next] = add(sum[t + 1][next], sum[t][from] + score);
                        if best[t][from] + score > best[t + 1][next] {
                            best[t + 1][next] = best[t][from] + score;
                            choice[t + 1][next] = label;
                        }
                    }
                }
            }
        }
        Self {
            values,
            stride,
            sum,
            best,
            choice,
        }
    }

    fn trace(&self, mut k: usize, mut d: usize) -> Vec<u8> {
        let mut labels = vec![0; self.values.len()];
        for t in (1..=self.values.len()).rev() {
            let label = self.choice[t][k * self.stride + d];
            labels[t - 1] = label;
            k -= usize::from(label > 0);
            d -= usize::from(label == 2);
        }
        labels
    }

    fn marginals(&self, terminal: &[f64], partition: f64) -> Vec<[f64; 3]> {
        let mut backward = terminal.to_vec();
        let mut out = vec![[0.; 3]; self.values.len()];
        for t in (0..self.values.len()).rev() {
            let pair = self.values[t];
            let mut previous = vec![NEG; backward.len()];
            for k in 0..=t {
                for d in 0..=k.min(self.stride - 1) {
                    let from = k * self.stride + d;
                    for (label, probability) in out[t]
                        .iter_mut()
                        .enumerate()
                        .take(if self.stride > 1 { 3 } else { 2 })
                    {
                        let next = (k + usize::from(label > 0)) * self.stride
                            + d
                            + usize::from(label == 2);
                        let score = if label == 0 {
                            0.
                        } else {
                            pair[0] + if label == 2 { pair[1] } else { 0. }
                        };
                        previous[from] = add(previous[from], score + backward[next]);
                        *probability +=
                            (self.sum[t][from] + score + backward[next] - partition).exp();
                    }
                }
            }
            backward = previous;
        }
        out
    }
}

#[derive(Clone, Serialize)]
pub struct Map {
    pub meter: usize,
    pub phase: usize,
    /// Null means unavailable, not an observed omission. Labels 0/1/2 are
    /// model-inferred omitted/plain/accented pulses, never detected beat events.
    pub inferred_labels: Vec<Option<u8>>,
    pub log_weight: f64,
    pub feature_log_ratio: f64,
    pub omission_log_prior: f64,
}

#[derive(Serialize)]
pub struct Component {
    pub meter: usize,
    pub phase: usize,
    pub log_ratio: f64,
    pub probability: f64,
}

#[derive(Serialize)]
pub struct Inference {
    pub log_ratio: f64,
    pub mean_pulse_retention: f64,
    pub mean_accent_retention: f64,
    pub count_probabilities: Vec<Vec<f64>>,
    pub label_probabilities: Vec<Option<[f64; 3]>>,
    pub components: Vec<Component>,
    pub joint_map: Map,
}

#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]
fn at_meter(table: &Table, ticks: &[usize], meter: usize, phase: usize) -> Result<Inference> {
    let mut plain = Vec::new();
    let mut bars = Vec::new();
    for (i, &t) in ticks.iter().enumerate() {
        if let Some(pair) = table.centered[t] {
            if (i + phase).is_multiple_of(meter) {
                bars.push((i, pair));
            } else {
                plain.push((i, pair));
            }
        }
    }
    let ordinary = Factor::new(plain.iter().map(|p| p.1).collect(), false);
    let accented = Factor::new(bars.iter().map(|p| p.1).collect(), true);
    let (o, s) = (plain.len(), bars.len());
    let n = o + s;
    let a = &ordinary.sum[o];
    let b = &accented.sum[s];
    let mut ta = vec![NEG; a.len()];
    let mut tb = vec![NEG; b.len()];
    let mut count = vec![vec![NEG; s + 1]; n + 1];
    let mut partition = NEG;
    let (mut best, mut best_counts) = (NEG, (0, 0, 0));
    let mut rate_b = NEG;
    let mut rate_d = NEG;
    let beat_prior: Vec<_> = (0..=n).map(|k| log_beta(k, n - k)).collect();
    let accent_prior: Vec<Vec<_>> = (0..=s)
        .map(|z| (0..=z).map(|d| log_beta(d, z - d)).collect())
        .collect();
    for k in 0..=o {
        for z in 0..=s {
            for d in 0..=z {
                let index = z * (s + 1) + d;
                let prior = beat_prior[k + z] + accent_prior[z][d];
                let terminal = prior - table.normalizer(k + z - d, d)?;
                ta[k] = add(ta[k], b[index] + terminal);
                tb[index] = add(tb[index], a[k] + terminal);
                let mass = a[k] + b[index] + terminal;
                partition = add(partition, mass);
                count[k + z][d] = add(count[k + z][d], mass);
                rate_b = add(rate_b, mass + ((k + z + 1) as f64 / (n + 2) as f64).ln());
                rate_d = add(rate_d, mass + ((d + 1) as f64 / (z + 2) as f64).ln());
                let value = ordinary.best[o][k] + accented.best[s][index] + terminal;
                if value > best {
                    best = value;
                    best_counts = (k, z, d);
                }
            }
        }
    }
    let (k, z, d) = best_counts;
    let prior = beat_prior[k + z] + log_beta(d, z - d);
    let mut labels = vec![None; ticks.len()];
    let mut probabilities = labels.iter().map(|_| None).collect::<Vec<_>>();
    for (entries, factor, terminal, retained, accents) in [
        (&plain, &ordinary, &ta, k, 0),
        (&bars, &accented, &tb, z, d),
    ] {
        let trace = factor.trace(retained, accents);
        let marginal = factor.marginals(terminal, partition);
        for ((&(i, _), label), p) in entries.iter().zip(trace).zip(marginal) {
            labels[i] = Some(label);
            probabilities[i] = Some(p);
        }
    }
    Ok(Inference {
        log_ratio: partition,
        mean_pulse_retention: (rate_b - partition).exp(),
        mean_accent_retention: (rate_d - partition).exp(),
        count_probabilities: count
            .into_iter()
            .map(|row| row.into_iter().map(|v| (v - partition).exp()).collect())
            .collect(),
        label_probabilities: probabilities,
        components: vec![],
        joint_map: Map {
            meter,
            phase,
            inferred_labels: labels,
            log_weight: best,
            feature_log_ratio: best - prior,
            omission_log_prior: prior,
        },
    })
}

/// Marginalizes a SINGLE constant meter/phase per supplied clock, plus both
/// omission processes. It does NOT carry forward the previous changing-meter
/// graph. This deliberate bounded diagnostic boundary is exported in reports.
#[allow(clippy::cast_precision_loss)]
pub fn infer(table: &Table, ticks: &[usize], minimum: usize, maximum: usize) -> Result<Inference> {
    validate(table, ticks, minimum, maximum)?;
    let mut components = Vec::new();
    let mut total = NEG;
    for meter in minimum..=maximum {
        for phase in 0..meter {
            let result = at_meter(table, ticks, meter, phase)?;
            let prior = -(((maximum - minimum + 1) * meter) as f64).ln();
            total = add(total, result.log_ratio + prior);
            components.push((prior, result));
        }
    }
    let n = ticks
        .iter()
        .filter(|&&t| table.centered[t].is_some())
        .count();
    let mut output = Inference {
        log_ratio: total,
        mean_pulse_retention: 0.,
        mean_accent_retention: 0.,
        count_probabilities: vec![vec![0.; ticks.len().div_ceil(minimum) + 1]; n + 1],
        label_probabilities: ticks
            .iter()
            .map(|&t| table.centered[t].map(|_| [0.; 3]))
            .collect(),
        components: vec![],
        joint_map: components[0].1.joint_map.clone(),
    };
    output.joint_map.log_weight = NEG;
    for (prior, result) in components {
        let p = (prior + result.log_ratio - total).exp();
        output.mean_pulse_retention += p * result.mean_pulse_retention;
        output.mean_accent_retention += p * result.mean_accent_retention;
        for (row, source) in output
            .count_probabilities
            .iter_mut()
            .zip(result.count_probabilities)
        {
            for (a, b) in row.iter_mut().zip(source) {
                *a += p * b;
            }
        }
        for (a, b) in output
            .label_probabilities
            .iter_mut()
            .zip(result.label_probabilities)
        {
            if let (Some(a), Some(b)) = (a, b) {
                for (a, b) in a.iter_mut().zip(b) {
                    *a += p * b;
                }
            }
        }
        output.components.push(Component {
            meter: result.joint_map.meter,
            phase: result.joint_map.phase,
            log_ratio: result.log_ratio,
            probability: p,
        });
        if result.joint_map.log_weight + prior > output.joint_map.log_weight {
            output.joint_map = result.joint_map;
            output.joint_map.log_weight += prior;
        }
    }
    Ok(output)
}

/// All constant-meter explanations of ONE inferred full-frame label assignment.
/// The emission/reference score is identical across compatible latent clocks;
/// returned weights contain priors only, not extra musical evidence.
#[allow(clippy::cast_precision_loss)]
pub fn assignment_prior(
    table: &Table,
    ticks: &[usize],
    labels: &[(usize, u8)],
    minimum: usize,
    maximum: usize,
) -> Result<Option<f64>> {
    validate(table, ticks, minimum, maximum)?;
    ensure!(
        labels.windows(2).all(|w| w[0].0 < w[1].0)
            && labels.iter().all(|&(t, l)| t < table.centered.len()
                && table.centered[t].is_some()
                && (1..=2).contains(&l)),
        "invalid emitted assignment"
    );
    let n = ticks
        .iter()
        .filter(|&&t| table.centered[t].is_some())
        .count();
    let indexed: Option<Vec<_>> = labels
        .iter()
        .map(|&(t, label)| ticks.binary_search(&t).ok().map(|i| (i, label)))
        .collect();
    let Some(indexed) = indexed else {
        return Ok(None);
    };
    let d = labels.iter().filter(|p| p.1 == 2).count();
    let mut total = NEG;
    for meter in minimum..=maximum {
        for phase in 0..meter {
            if indexed
                .iter()
                .any(|&(i, label)| label == 2 && !(i + phase).is_multiple_of(meter))
            {
                continue;
            }
            let z = indexed
                .iter()
                .filter(|&&(i, _)| (i + phase).is_multiple_of(meter))
                .count();
            let weight = -(((maximum - minimum + 1) * meter) as f64).ln()
                + log_beta(labels.len(), n - labels.len())
                + log_beta(d, z - d);
            total = add(total, weight);
        }
    }
    Ok(if total == NEG { None } else { Some(total) })
}

fn validate(table: &Table, ticks: &[usize], minimum: usize, maximum: usize) -> Result<()> {
    ensure!(
        !ticks.is_empty()
            && ticks.len() <= 64
            && ticks.windows(2).all(|w| w[0] < w[1])
            && ticks.iter().all(|&t| t < table.centered.len()),
        "invalid supplied clock"
    );
    ensure!(
        minimum >= 2 && maximum >= minimum && maximum <= 7,
        "invalid constant meter family"
    );
    Ok(())
}

/// Matched constant-meter baseline: every available latent tick emits, and
/// every available latent bar start emits an accent. No omission-rate prior.
#[allow(clippy::cast_precision_loss)]
pub fn intact_reference(
    table: &Table,
    ticks: &[usize],
    minimum: usize,
    maximum: usize,
) -> Result<f64> {
    validate(table, ticks, minimum, maximum)?;
    let mut total = NEG;
    for meter in minimum..=maximum {
        for phase in 0..meter {
            let (mut b, mut d, mut score) = (0, 0, 0.);
            for (i, &t) in ticks.iter().enumerate() {
                if let Some(p) = table.centered[t] {
                    b += 1;
                    score += p[0];
                    if (i + phase).is_multiple_of(meter) {
                        d += 1;
                        score += p[1];
                    }
                }
            }
            total = add(
                total,
                score
                    - table.normalizer(b - d, d)?
                    - (((maximum - minimum + 1) * meter) as f64).ln(),
            );
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn omissions_and_both_beta_integrals_match_exhaustive_paths() {
        let values = [
            Some([1., 0.5]),
            Some([-1., 2.]),
            None,
            Some([0.5, -0.2]),
            Some([2., 1.]),
        ];
        let table = Table::new(&values, 4, 4).unwrap();
        let ticks = [0, 1, 2, 3, 4];
        let actual = infer(&table, &ticks, 2, 3).unwrap();
        let factorial = |n: usize| (1..=n).map(|i| i as f64).product::<f64>();
        let (mut mass, mut rb, mut rd, mut best) = (0., 0., 0., 0_f64);
        let mut occupancy = [[0.; 3]; 5];
        let mut counts = [[0.; 4]; 5];
        for meter in 2..=3 {
            for phase in 0..meter {
                for encoding in 0..243 {
                    let mut code = encoding;
                    let (mut b, mut z, mut d, mut score, mut valid) = (0, 0, 0, 0., true);
                    let mut labels = [0; 5];
                    for i in 0..5 {
                        let label = code % 3;
                        code /= 3;
                        labels[i] = label;
                        let bar = (i + phase).is_multiple_of(meter);
                        if let Some(p) = table.centered[i] {
                            if label > 0 {
                                b += 1;
                                score += p[0];
                                z += usize::from(bar);
                            }
                            if label == 2 {
                                d += 1;
                                score += p[1];
                                valid &= bar;
                            }
                        } else {
                            valid &= label == 0;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    let weight = (score - table.normalizer(b - d, d).unwrap()).exp()
                        * factorial(b)
                        * factorial(4 - b)
                        / factorial(5)
                        * factorial(d)
                        * factorial(z - d)
                        / factorial(z + 1)
                        / (2 * meter) as f64;
                    mass += weight;
                    best = best.max(weight);
                    rb += weight * (b + 1) as f64 / 6.;
                    rd += weight * (d + 1) as f64 / (z + 2) as f64;
                    counts[b][d] += weight;
                    for i in 0..5 {
                        if values[i].is_some() {
                            occupancy[i][labels[i]] += weight;
                        }
                    }
                }
            }
        }
        assert!((actual.log_ratio - mass.ln()).abs() < 1e-12);
        assert!((actual.joint_map.log_weight - best.ln()).abs() < 1e-12);
        assert!((actual.mean_pulse_retention - rb / mass).abs() < 1e-12);
        assert!((actual.mean_accent_retention - rd / mass).abs() < 1e-12);
        assert!(actual.label_probabilities[2].is_none());
        assert!(actual.joint_map.inferred_labels[2].is_none());
        for (i, p) in actual.label_probabilities.iter().enumerate() {
            if let Some(p) = p {
                for j in 0..3 {
                    assert!((p[j] - occupancy[i][j] / mass).abs() < 1e-12);
                }
            }
        }
        for (a, b) in actual.count_probabilities.iter().zip(counts) {
            for (a, b) in a.iter().zip(b) {
                assert!((a - b / mass).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn flat_and_unavailable_are_not_observed_omissions() {
        let flat = Table::new(&[Some([0., 0.]); 5], 5, 5).unwrap();
        let f = infer(&flat, &[0, 1, 2, 3, 4], 2, 7).unwrap();
        assert!(f.log_ratio.abs() < 1e-12);
        assert!((f.mean_pulse_retention - 0.5).abs() < 1e-12);
        assert!((f.mean_accent_retention - 0.5).abs() < 1e-12);
        assert!(
            f.label_probabilities
                .iter()
                .flatten()
                .all(|p| (p[0] - 0.5).abs() < 1e-12)
        );
        let unknown = Table::new(&[None; 5], 0, 0).unwrap();
        let u = infer(&unknown, &[0, 1, 2, 3, 4], 2, 7).unwrap();
        assert!(u.log_ratio.abs() < 1e-12);
        assert!(u.label_probabilities.iter().all(Option::is_none));
        assert!(u.joint_map.inferred_labels.iter().all(Option::is_none));
        assert!(infer(&flat, &[0, 0], 2, 7).is_err());
        assert!(infer(&flat, &[9], 2, 7).is_err());
        assert!(infer(&flat, &[0], 1, 7).is_err());
    }

    #[test]
    fn equivalent_assignments_require_actual_support_and_valid_labels() {
        let table = Table::new(&[Some([0., 0.]); 9], 9, 9).unwrap();
        let dense = [0, 1, 2, 3, 4, 5, 6, 7];
        let sparse = [0, 2, 4, 6];
        let labels = [(0, 2), (2, 1), (4, 2), (6, 1)];
        assert!(
            assignment_prior(&table, &dense, &labels, 2, 7)
                .unwrap()
                .is_some()
        );
        assert!(
            assignment_prior(&table, &sparse, &labels, 2, 7)
                .unwrap()
                .is_some()
        );
        assert!(
            assignment_prior(&table, &sparse, &[(1, 1)], 2, 7)
                .unwrap()
                .is_none()
        );
        assert!(assignment_prior(&table, &sparse, &[(0, 0)], 2, 7).is_err());
        assert!(assignment_prior(&table, &sparse, &[(0, 1), (0, 1)], 2, 7).is_err());
        assert!(assignment_prior(&table, &sparse, &[(9, 1)], 2, 7).is_err());
    }
}
