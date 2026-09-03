//! One frozen evaluation-only approximation, not a product resampler choice.
//!
//! Standard windowed-sinc interpolation with a rational sample clock. A squared
//! Blackman-Harris window spans 256 samples at the lower rate. The 0.95685
//! normalized sinc cutoff approximates the reference HQ impulse's -6 dB point
//! on generated signals; it was fixed before any musical candidate inference.
//! No libsoxr implementation, filter table, or runtime is included.

use anyhow::{Result, ensure};

pub const ID: &str = "phase-exact-bh2-256-v1";
/// Coefficient storage only; excludes the caller's PCM and returned PCM.
pub const COEFFICIENT_BUDGET_BYTES: usize = 8 * 1024 * 1024;
const TARGET: u32 = 22050;
const CUTOFF: f64 = 0.95685;

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn resample(samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    resample_with_budget(samples, rate, COEFFICIENT_BUDGET_BYTES)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn resample_with_budget(samples: &[f32], rate: u32, budget: usize) -> Result<Vec<f32>> {
    ensure!(
        (8000..=192_000).contains(&rate),
        "candidate is only characterized for 8..192 kHz inputs"
    );
    ensure!(
        !samples.is_empty() && samples.iter().all(|s| s.is_finite()),
        "candidate requires finite nonempty PCM"
    );
    if rate == TARGET {
        return Ok(samples.to_vec());
    }
    let wanted = usize::try_from(
        (samples.len() as u128 * u128::from(TARGET) + u128::from(rate) / 2) / u128::from(rate),
    )?;
    ensure!(wanted > 0, "input is shorter than one output frame");
    let scale = (f64::from(TARGET) / f64::from(rate)).min(1.0);
    let radius = (128.0 / scale).ceil() as usize;
    let phases = (TARGET / gcd(rate, TARGET)) as usize;
    let taps = radius * 2;
    let batch_phases = (budget / (taps * size_of::<f64>())).min(phases).min(wanted);
    ensure!(
        batch_phases > 0,
        "coefficient budget cannot hold one kernel"
    );
    // Tile over rational phases, not time. Each coefficient is generated once,
    // and every output keeps the exact same dot-product order as the full table.
    // Common rates fit in one tile; coprime rates cannot allocate an unbounded
    // phase table. This bound excludes input/output audio storage.
    let mut coefficients = vec![0.0; batch_phases * taps];
    let mut output = vec![0.0; wanted];
    for first_phase in (0..phases.min(wanted)).step_by(batch_phases) {
        let last_phase = (first_phase + batch_phases).min(phases).min(wanted);
        for phase in first_phase..last_phase {
            let fraction =
                ((phase as u64 * u64::from(rate)) % u64::from(TARGET)) as f64 / f64::from(TARGET);
            let offset = (phase - first_phase) * taps;
            let kernel = &mut coefficients[offset..offset + taps];
            for (i, coefficient) in kernel.iter_mut().enumerate() {
                let distance = i as f64 + 1.0 - radius as f64 - fraction;
                let x = distance / radius as f64;
                if x.abs() >= 1.0 {
                    *coefficient = 0.0;
                    continue;
                }
                let angle = std::f64::consts::PI * x;
                let window = 0.35875
                    + 0.48829 * angle.cos()
                    + 0.14128 * (2.0 * angle).cos()
                    + 0.01168 * (3.0 * angle).cos();
                let argument = std::f64::consts::PI * CUTOFF * scale * distance;
                let sinc = if argument.abs() < 1e-12 {
                    1.0
                } else {
                    argument.sin() / argument
                };
                *coefficient = window * window * sinc;
            }
            let sum = kernel.iter().sum::<f64>();
            for weight in kernel {
                *weight /= sum;
            }
        }
        for cycle in (0..wanted).step_by(phases) {
            for (frame, output_sample) in output
                .iter_mut()
                .enumerate()
                .take((cycle + last_phase).min(wanted))
                .skip(cycle + first_phase)
            {
                let source =
                    usize::try_from(frame as u128 * u128::from(rate) / u128::from(TARGET))?;
                let offset = (frame % phases - first_phase) * taps;
                let kernel = &coefficients[offset..offset + taps];
                let left = source.saturating_sub(radius - 1);
                let skip = (radius - 1).saturating_sub(source);
                let right = source.saturating_add(radius + 1).min(samples.len());
                let value = samples[left..right]
                    .iter()
                    .zip(&kernel[skip..])
                    .map(|(&sample, &weight)| f64::from(sample) * weight)
                    .sum::<f64>();
                *output_sample = value as f32;
            }
        }
    }
    ensure!(
        output.iter().all(|s| s.is_finite()),
        "candidate output overflowed"
    );
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_coefficients_preserve_every_output_bit() {
        for rate in [8000, 16000, 44100, 48000, 96000, 192_000, 44101, 191_999] {
            let len = if rate == 44101 || rate == 191_999 {
                257
            } else {
                4097
            };
            let samples = (0..len)
                .map(|i| if i % 17 == 0 { 0.75 } else { -0.125 })
                .collect::<Vec<_>>();
            let expected = resample(&samples, rate).unwrap();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let radius = (128.0 / (f64::from(TARGET) / f64::from(rate)).min(1.0)).ceil() as usize;
            for kernels in [1, 7] {
                let actual =
                    resample_with_budget(&samples, rate, kernels * radius * 2 * size_of::<f64>())
                        .unwrap();
                assert!(
                    expected
                        .iter()
                        .map(|s| s.to_bits())
                        .eq(actual.iter().map(|s| s.to_bits())),
                    "rate={rate} kernels={kernels}"
                );
            }
        }
    }

    #[test]
    fn worst_coprime_phase_table_is_tiled_below_eight_mib() {
        let rate = 191_999_u32;
        let phases = TARGET / gcd(rate, TARGET);
        let taps = 2230_usize;
        assert_eq!(phases, 22050);
        assert!(phases as usize * taps * size_of::<f64>() > 390_000_000);
        let batch = COEFFICIENT_BUDGET_BYTES / (taps * size_of::<f64>());
        assert!(batch > 0 && batch < phases as usize);
        assert!(batch * taps * size_of::<f64>() <= COEFFICIENT_BUDGET_BYTES);
        assert!(resample_with_budget(&[0.25; 257], rate, 1).is_err());
    }

    #[test]
    fn native_pcm_is_bit_exact_and_invalid_inputs_fail() {
        let input = [0.0, -0.0, 0.25, -0.5];
        assert_eq!(
            resample(&input, 22050)
                .unwrap()
                .iter()
                .map(|s| s.to_bits())
                .collect::<Vec<_>>(),
            input.map(f32::to_bits)
        );
        for (samples, rate) in [
            (&[][..], 22050),
            (&[f32::NAN], 44100),
            (&[f32::INFINITY], 44100),
            (&[1.0], 0),
            (&[1.0], 400_000),
        ] {
            assert!(resample(samples, rate).is_err());
        }
    }

    #[test]
    fn durations_and_partial_inputs_are_preserved() {
        for rate in [8000, 16000, 22050, 44100, 48000, 96000, 192_000] {
            for len in [1, 2, 17, 127, 256, 4095, 4096, 4097] {
                let wanted = (len * TARGET as usize + rate as usize / 2) / rate as usize;
                let result = resample(&vec![0.25; len], rate);
                if wanted == 0 {
                    assert!(result.is_err());
                } else {
                    assert_eq!(result.unwrap().len(), wanted);
                }
            }
        }
    }

    #[test]
    fn impulses_at_shared_grid_points_have_no_integer_delay() {
        for rate in [8000, 16000, 44100, 48000, 96000, 192_000] {
            let mut input = vec![0.0; rate as usize / 4];
            let period = (rate / gcd(rate, TARGET)) as usize;
            let position = (input.len() / 2 / period) * period;
            input[position] = 1.0;
            let output = resample(&input, rate).unwrap();
            let peak = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap()
                .0;
            assert_eq!(peak, position * TARGET as usize / rate as usize);
        }
    }
}
