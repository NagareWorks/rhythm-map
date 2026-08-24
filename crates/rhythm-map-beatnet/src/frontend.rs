//! `BeatNet`'s training-time log-frequency spectrogram contract.

use std::{f32::consts::PI, sync::Arc};

use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex32};
use rhythm_map_core::BackendError;

pub(crate) const SAMPLE_RATE_HZ: u32 = 22_050;
pub(crate) const FRAME_SIZE: usize = 1_411;
pub(crate) const HOP_SIZE: usize = 441;
pub(crate) const FREQUENCY_BANDS: usize = 136;
pub(crate) const FEATURE_DIMENSION: usize = FREQUENCY_BANDS * 2;

const FFT_BINS: usize = FRAME_SIZE / 2;
const FILTERS_PER_OCTAVE: usize = 24;
const MINIMUM_FREQUENCY_HZ: f64 = 30.0;
const MAXIMUM_FREQUENCY_HZ: f64 = 17_000.0;
const REFERENCE_FREQUENCY_HZ: f64 = 440.0;

#[derive(Debug)]
struct TriangularFilter {
    start_bin: usize,
    weights: Vec<f32>,
}

/// Stateful FFT frontend matching `BeatNet`'s madmom feature pipeline.
pub(crate) struct BeatNetFrontend {
    fft: Arc<dyn RealToComplex<f32>>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    window: Vec<f32>,
    filters: Vec<TriangularFilter>,
}

impl BeatNetFrontend {
    pub(crate) fn new() -> Result<Self, BackendError> {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let filters = logarithmic_filterbank();
        if filters.len() != FREQUENCY_BANDS {
            return Err(BackendError::new(format!(
                "BeatNet frontend produced {} bands, expected {FREQUENCY_BANDS}",
                filters.len()
            )));
        }
        Ok(Self {
            fft_input: fft.make_input_vec(),
            fft_output: fft.make_output_vec(),
            fft_scratch: fft.make_scratch_vec(),
            fft,
            window: hanning_window(FRAME_SIZE),
            filters,
        })
    }

    pub(crate) fn extract(&mut self, samples: &[f32]) -> Result<Vec<f32>, BackendError> {
        let frame_count = samples.len().div_ceil(HOP_SIZE);
        let mut features = Vec::with_capacity(frame_count * FEATURE_DIMENSION);
        let mut previous = vec![0.0_f32; FREQUENCY_BANDS];
        let mut magnitude = vec![0.0_f32; FFT_BINS];

        for frame_index in 0..frame_count {
            fill_centered_frame(
                &mut self.fft_input,
                samples,
                frame_index * HOP_SIZE,
                &self.window,
            );
            self.fft
                .process_with_scratch(
                    &mut self.fft_input,
                    &mut self.fft_output,
                    &mut self.fft_scratch,
                )
                .map_err(|error| BackendError::new(format!("BeatNet STFT failed: {error}")))?;
            for (value, bin) in magnitude.iter_mut().zip(&self.fft_output[..FFT_BINS]) {
                *value = bin.norm();
            }

            let frame_start = features.len();
            for filter in &self.filters {
                let filtered = magnitude[filter.start_bin..]
                    .iter()
                    .zip(&filter.weights)
                    .map(|(bin, weight)| bin * weight)
                    .sum::<f32>();
                features.push((1.0 + filtered).log10());
            }
            for band in 0..FREQUENCY_BANDS {
                let current = features[frame_start + band];
                features.push((current - previous[band]).max(0.0));
                previous[band] = current;
            }
        }
        Ok(features)
    }
}

fn fill_centered_frame(output: &mut [f32], samples: &[f32], reference: usize, window: &[f32]) {
    let half = FRAME_SIZE / 2;
    for (index, (sample, weight)) in output.iter_mut().zip(window).enumerate() {
        let source = reference
            .checked_add(index)
            .and_then(|value| value.checked_sub(half));
        *sample = source
            .and_then(|source| samples.get(source))
            .copied()
            .unwrap_or(0.0)
            * weight;
    }
}

#[allow(clippy::cast_precision_loss)]
fn hanning_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / (size - 1) as f32).cos())
        .collect()
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn logarithmic_filterbank() -> Vec<TriangularFilter> {
    let left = (MINIMUM_FREQUENCY_HZ / REFERENCE_FREQUENCY_HZ)
        .log2()
        .mul_add(FILTERS_PER_OCTAVE as f64, 0.0)
        .floor() as i32;
    let right = (MAXIMUM_FREQUENCY_HZ / REFERENCE_FREQUENCY_HZ)
        .log2()
        .mul_add(FILTERS_PER_OCTAVE as f64, 0.0)
        .ceil() as i32;
    let bin_width = f64::from(SAMPLE_RATE_HZ) / (FFT_BINS * 2) as f64;
    let mut bins = Vec::new();
    for step in left..right {
        let frequency =
            REFERENCE_FREQUENCY_HZ * 2.0_f64.powf(f64::from(step) / FILTERS_PER_OCTAVE as f64);
        if !(MINIMUM_FREQUENCY_HZ..=MAXIMUM_FREQUENCY_HZ).contains(&frequency) {
            continue;
        }
        let bin = ((frequency / bin_width).round() as usize).min(FFT_BINS - 1);
        if bins.last().copied() != Some(bin) {
            bins.push(bin);
        }
    }

    bins.windows(3)
        .map(|points| triangular_filter(points[0], points[1], points[2]))
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn triangular_filter(start: usize, mut center: usize, mut stop: usize) -> TriangularFilter {
    if stop - start < 2 {
        center = start;
        stop = start + 1;
    }
    let center = center - start;
    let length = stop - start;
    let mut weights = vec![0.0_f32; length];
    if center > 0 {
        for (index, weight) in weights[..center].iter_mut().enumerate() {
            *weight = index as f32 / center as f32;
        }
    }
    let falling = length - center;
    for (index, weight) in weights[center..].iter_mut().enumerate() {
        *weight = 1.0 - index as f32 / falling as f32;
    }
    let sum = weights.iter().sum::<f32>();
    for weight in &mut weights {
        *weight /= sum;
    }
    TriangularFilter {
        start_bin: start,
        weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_contract_has_expected_shape() {
        let mut frontend = BeatNetFrontend::new().unwrap();
        let features = frontend
            .extract(&vec![0.0; SAMPLE_RATE_HZ as usize])
            .unwrap();
        assert_eq!(features.len(), 50 * FEATURE_DIMENSION);
        assert!(features.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn impulse_uses_centered_zero_padded_first_frame() {
        let mut frontend = BeatNetFrontend::new().unwrap();
        let mut samples = vec![0.0; HOP_SIZE];
        samples[0] = 1.0;
        let features = frontend.extract(&samples).unwrap();
        assert_eq!(features.len(), FEATURE_DIMENSION);
        assert!(features[..FREQUENCY_BANDS].iter().all(|value| *value > 0.0));
        assert_eq!(&features[..FREQUENCY_BANDS], &features[FREQUENCY_BANDS..]);
    }
}
