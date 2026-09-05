//! Shared-frame paired-head evidence, not a calibrated audio likelihood.
use anyhow::{Result, ensure};

const NEG: f64 = f64::NEG_INFINITY;
pub fn add(a: f64, b: f64) -> f64 {
    if a == NEG {
        return b;
    }
    if b == NEG {
        return a;
    }
    a.max(b) + (-(a - b).abs()).exp().ln_1p()
}

/// Candidate-independent [1,2,1]/4 smoothing, optionally contrasted against
/// the largest odd window shorter than the minimum allowed beat period.
/// Both kernels are clipped/renormalized within each observed run. Unknown
/// samples never become observed silence, and no candidate defines a window.
#[allow(clippy::cast_precision_loss)]
pub fn features(
    beat: &[f32],
    bar: &[f32],
    available: &[bool],
    minimum_period: usize,
    contextual: bool,
) -> Result<Vec<Option<[f64; 2]>>> {
    ensure!(
        !beat.is_empty() && beat.len() == bar.len() && beat.len() == available.len(),
        "head shape mismatch"
    );
    ensure!(
        (3..=75).contains(&minimum_period),
        "invalid feature period domain"
    );
    ensure!(
        beat.iter()
            .zip(bar)
            .zip(available)
            .all(|((&b, &d), &a)| !a || (b.is_finite() && d.is_finite())),
        "non-finite observed head"
    );
    let mut output = vec![None; beat.len()];
    let mut start = 0;
    while start < beat.len() {
        if !available[start] {
            start += 1;
            continue;
        }
        let end = (start..beat.len())
            .find(|&i| !available[i])
            .unwrap_or(beat.len());
        let mut smooth = vec![[0.; 2]; end - start];
        for (offset, pair) in smooth.iter_mut().enumerate() {
            let t = start + offset;
            let left = t.saturating_sub(1).max(start);
            let right = (t + 2).min(end);
            let weight_sum: f64 = (left..right).map(|i| if i == t { 2. } else { 1. }).sum();
            for (head, out) in [beat, bar].into_iter().zip(pair) {
                *out = (left..right)
                    .map(|i| f64::from(head[i]) * if i == t { 2. } else { 1. })
                    .sum::<f64>()
                    / weight_sum;
            }
        }
        for (offset, &pair) in smooth.iter().enumerate() {
            let mut result = pair;
            if contextual {
                let radius = (minimum_period - 1) / 2;
                let from = offset.saturating_sub(radius);
                let to = (offset + radius + 1).min(smooth.len());
                for h in 0..2 {
                    let log_mean = smooth[from..to].iter().map(|p| p[h]).fold(NEG, add)
                        - ((to - from) as f64).ln();
                    result[h] -= log_mean;
                }
            }
            output[start + offset] = Some(result);
        }
        start = end;
    }
    Ok(output)
}

pub struct Table {
    pub centered: Vec<Option<[f64; 2]>>,
    pub available_frames: usize,
    normalizers: Vec<f64>,
    stride: usize,
    max_plain: usize,
    max_bars: usize,
}

impl Table {
    /// Coefficient of u^a v^d in product(1+exp(b)*u+exp(b+d)*v),
    /// divided by N!/(a! d! (N-a-d)!). Paired heads are permuted together;
    /// a downbeat gets one joint label, not two independent normalizations.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(values: &[Option<[f64; 2]>], max_plain: usize, max_bars: usize) -> Result<Self> {
        ensure!(
            !values.is_empty() && values.iter().flatten().flatten().all(|x| x.is_finite()),
            "invalid features"
        );
        let available_frames = values.iter().flatten().count();
        ensure!(
            max_plain <= available_frames && max_bars <= available_frames,
            "count bound exceeds observed domain"
        );
        let maximum = values.iter().flatten().fold([NEG; 2], |mut a, b| {
            for h in 0..2 {
                a[h] = a[h].max(b[h]);
            }
            a
        });
        let centered: Vec<_> = values
            .iter()
            .map(|pair| pair.map(|p| [p[0] - maximum[0], p[1] - maximum[1]]))
            .collect();
        ensure!(
            centered.iter().flatten().flatten().all(|x| x.is_finite()),
            "feature range overflow"
        );
        let stride = max_bars + 1;
        let mut coefficients = vec![NEG; (max_plain + 1) * stride];
        coefficients[0] = 0.;
        for (t, p) in centered.iter().flatten().enumerate() {
            for a in (0..=max_plain.min(t + 1)).rev() {
                for d in (0..=max_bars.min(t + 1 - a)).rev() {
                    let i = a * stride + d;
                    if a > 0 {
                        coefficients[i] =
                            add(coefficients[i], coefficients[(a - 1) * stride + d] + p[0]);
                    }
                    if d > 0 {
                        coefficients[i] = add(
                            coefficients[i],
                            coefficients[a * stride + d - 1] + p[0] + p[1],
                        );
                    }
                }
            }
        }
        let mut factorials = vec![0.; available_frames + 1];
        for n in 1..factorials.len() {
            factorials[n] = factorials[n - 1] + (n as f64).ln();
        }
        for a in 0..=max_plain {
            for d in 0..=max_bars.min(available_frames - a) {
                coefficients[a * stride + d] -= factorials[available_frames]
                    - factorials[a]
                    - factorials[d]
                    - factorials[available_frames - a - d];
                ensure!(
                    coefficients[a * stride + d].is_finite(),
                    "normalizer overflow"
                );
            }
        }
        Ok(Self {
            centered,
            available_frames,
            normalizers: coefficients,
            stride,
            max_plain,
            max_bars,
        })
    }

    pub fn normalizer(&self, plain: usize, bars: usize) -> Result<f64> {
        ensure!(
            plain <= self.max_plain
                && bars <= self.max_bars
                && plain + bars <= self.available_frames,
            "unsupported count"
        );
        Ok(self.normalizers[plain * self.stride + bars])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn paired_reference_matches_exhaustive_three_label_assignments() {
        let values = [
            Some([0.2, 1.]),
            Some([-1., 0.7]),
            Some([1.4, -0.3]),
            Some([0.5, 1.5]),
        ];
        let table = Table::new(&values, 4, 4).unwrap();
        let mut sums = [[0.; 5]; 5];
        let mut counts = [[0; 5]; 5];
        for mut encoding in 0..81 {
            let (mut a, mut d, mut weight) = (0, 0, 0.);
            for p in table.centered.iter().flatten() {
                match encoding % 3 {
                    1 => {
                        a += 1;
                        weight += p[0];
                    }
                    2 => {
                        d += 1;
                        weight += p[0] + p[1];
                    }
                    _ => {}
                }
                encoding /= 3;
            }
            sums[a][d] += weight.exp();
            counts[a][d] += 1;
        }
        for a in 0..=4 {
            for d in 0..=4 - a {
                assert!(
                    (table.normalizer(a, d).unwrap() - (sums[a][d] / counts[a][d] as f64).ln())
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn missing_frames_and_kernel_edges_do_not_borrow_unobserved_values() {
        let beat = [-2., -2., f32::NAN, -8., -8.];
        let available = [true, true, false, true, true];
        let scores = features(&beat, &beat, &available, 10, true).unwrap();
        assert_eq!(scores[2], None);
        assert!(scores.iter().flatten().flatten().all(|x| x.abs() < 1e-12));
        let table = Table::new(&scores, 4, 4).unwrap();
        assert_eq!(table.available_frames, 4);
        assert!(table.normalizer(2, 1).unwrap().abs() < 1e-12);
        assert!(features(&beat, &beat, &[true; 5], 10, true).is_err());
        let missing = Table::new(&[None; 5], 0, 0).unwrap();
        assert_eq!(missing.normalizer(0, 0).unwrap(), 0.);
    }

    #[test]
    fn paired_heads_are_not_independent_repeated_evidence() {
        let pairs = [Some([0., 0.]), Some([1., 1.]), Some([2., 2.])];
        let table = Table::new(&pairs, 1, 1).unwrap();
        let first = table.normalizer(1, 0).unwrap();
        let joint = table.normalizer(0, 1).unwrap();
        assert!((joint - 2. * first).abs() > 0.1);
        let shifted: Vec<_> = pairs
            .iter()
            .map(|p| p.map(|x| [x[0] + 100., x[1] - 40.]))
            .collect();
        let other = Table::new(&shifted, 1, 1).unwrap();
        assert!((joint - other.normalizer(0, 1).unwrap()).abs() < 1e-12);
    }
}
