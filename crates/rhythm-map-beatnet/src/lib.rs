//! Experimental pure-Rust adapter for `BeatNet`'s ONNX beat/downbeat activations.
//!
//! This crate is a calibration backend, not a shipping policy. It retains all
//! model-supported pulse maxima so Rhythm Map can measure whether `BeatNet` adds
//! evidence that the default Beat This backend misses.

use std::path::Path;

use rhythm_map_core::{
    BackendError, BeatCandidate, ModelInfo, ObservedBeat, RhythmObservationBackend,
    RhythmObservations,
};
use rten::{Model, NodeId, ValueView};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

use crate::frontend::{BeatNetFrontend, FEATURE_DIMENSION, SAMPLE_RATE_HZ};

mod frontend;

const FRAME_RATE_HZ: f64 = 50.0;
const MINIMUM_BPM: f64 = 40.0;
const MAXIMUM_BPM: f64 = 320.0;
const TEMPO_CHANGE_PENALTY: f64 = 100.0;
const BEAT_STATE_BIAS: f64 = 2.0;
const MAXIMUM_PEAK_CORRECTION_FRAMES: usize = 3;

/// Neural activations retained before discrete event selection.
#[derive(Debug, Clone)]
pub struct BeatNetInference {
    duration_s: f64,
    beat_probabilities: Vec<f32>,
    downbeat_probabilities: Vec<f32>,
    nonbeat_probabilities: Vec<f32>,
}

impl BeatNetInference {
    /// Per-frame probability of a non-downbeat beat.
    #[must_use]
    pub fn beat_probabilities(&self) -> &[f32] {
        &self.beat_probabilities
    }

    /// Per-frame downbeat probability.
    #[must_use]
    pub fn downbeat_probabilities(&self) -> &[f32] {
        &self.downbeat_probabilities
    }

    /// Per-frame non-beat probability.
    #[must_use]
    pub fn nonbeat_probabilities(&self) -> &[f32] {
        &self.nonbeat_probabilities
    }
}

/// `BeatNet` ONNX inference backed by pure-Rust `RTen`.
pub struct BeatNetBackend {
    model: Model,
    input_id: NodeId,
    output_id: NodeId,
    frontend: BeatNetFrontend,
    model_name: String,
}

impl BeatNetBackend {
    /// Load a `BeatNet` ONNX graph.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the graph or its named I/O cannot load.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, BackendError> {
        let path = model_path.as_ref();
        let model = Model::load_file(path)
            .map_err(|error| BackendError::new(format!("failed to load BeatNet model: {error}")))?;
        let input_id = model
            .node_id("input")
            .map_err(|error| BackendError::new(format!("BeatNet input is missing: {error}")))?;
        let output_id = model
            .node_id("output")
            .map_err(|error| BackendError::new(format!("BeatNet output is missing: {error}")))?;
        let model_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("beatnet_bda.onnx")
            .to_string();
        Ok(Self {
            model,
            input_id,
            output_id,
            frontend: BeatNetFrontend::new()?,
            model_name,
        })
    }

    /// Resample, extract the 272-value `BeatNet` feature frames, and run ONNX.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for invalid PCM, resampling, feature extraction,
    /// model inference, or an incompatible model output.
    pub fn infer_mono(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<BeatNetInference, BackendError> {
        if sample_rate == 0 {
            return Err(BackendError::new("sample rate must be greater than zero"));
        }
        if samples.is_empty() {
            return Err(BackendError::new("audio buffer must not be empty"));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(BackendError::new(
                "audio buffer contains a non-finite sample",
            ));
        }
        let duration_s = usize_to_f64(samples.len()) / f64::from(sample_rate);
        let resampled = resample_mono(samples, sample_rate)?;
        let features = self.frontend.extract(&resampled)?;
        let frame_count = features.len() / FEATURE_DIMENSION;
        let input = ValueView::from_shape([1, frame_count, FEATURE_DIMENSION], &features)
            .map_err(|error| BackendError::new(format!("invalid BeatNet input shape: {error}")))?;
        let [output] = self
            .model
            .run_n(vec![(self.input_id, input.into())], [self.output_id], None)
            .map_err(|error| BackendError::new(format!("BeatNet inference failed: {error}")))?;
        let (shape, probabilities) = output.into_shape_vec::<f32, 3>().map_err(|error| {
            BackendError::new(format!("invalid BeatNet output tensor: {error}"))
        })?;
        if shape != [1, 3, frame_count] {
            return Err(BackendError::new(format!(
                "BeatNet output shape {shape:?} does not match [1, 3, {frame_count}]"
            )));
        }
        Ok(BeatNetInference {
            duration_s,
            beat_probabilities: probabilities[..frame_count].to_vec(),
            downbeat_probabilities: probabilities[frame_count..2 * frame_count].to_vec(),
            nonbeat_probabilities: probabilities[2 * frame_count..].to_vec(),
        })
    }

    fn decode_activations(&self, inference: &BeatNetInference) -> RhythmObservations {
        let pulse = inference
            .beat_probabilities
            .iter()
            .zip(&inference.downbeat_probabilities)
            .map(|(beat, downbeat)| beat + downbeat)
            .collect::<Vec<_>>();
        let candidate_frames = local_maxima(&pulse);
        let beat_candidates = candidate_frames
            .iter()
            .filter_map(|&frame| observation_at(inference, frame, pulse[frame]))
            .map(|event| BeatCandidate {
                time_s: event.time_s,
                confidence: event.confidence,
                downbeat_confidence: event.downbeat_confidence,
            })
            .collect();
        let beats = decode_activation_path(&pulse, &candidate_frames)
            .into_iter()
            .filter_map(|frame| observation_at(inference, frame, pulse[frame]))
            .collect();
        RhythmObservations {
            duration_s: inference.duration_s,
            beats,
            beat_candidates,
            activity: Vec::new(),
            onsets: Vec::new(),
            source: ModelInfo {
                backend: "beatnet-rten-viterbi-v1-experimental".to_string(),
                model: self.model_name.clone(),
                version: None,
                frame_rate_hz: Some(FRAME_RATE_HZ),
            },
        }
    }
}

impl RhythmObservationBackend for BeatNetBackend {
    fn observe_mono(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<RhythmObservations, BackendError> {
        let inference = self.infer_mono(samples, sample_rate)?;
        Ok(self.decode_activations(&inference))
    }
}

fn resample_mono(samples: &[f32], source_sample_rate: u32) -> Result<Vec<f32>, BackendError> {
    if source_sample_rate == SAMPLE_RATE_HZ {
        return Ok(samples.to_vec());
    }
    let parameters = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = Async::<f32>::new_sinc(
        f64::from(SAMPLE_RATE_HZ) / f64::from(source_sample_rate),
        2.0,
        &parameters,
        samples.len(),
        1,
        FixedAsync::Input,
    )
    .map_err(|error| BackendError::new(format!("failed to configure resampler: {error}")))?;
    let input = InterleavedSlice::new(samples, 1, samples.len())
        .map_err(|error| BackendError::new(format!("invalid resampler input: {error:?}")))?;
    let output = resampler
        .process(&input, 0, None)
        .map_err(|error| BackendError::new(format!("BeatNet resampling failed: {error}")))?;
    Ok(output.take_data())
}

fn observation_at(
    inference: &BeatNetInference,
    frame: usize,
    pulse_probability: f32,
) -> Option<ObservedBeat> {
    let time_s = usize_to_f64(frame) / FRAME_RATE_HZ;
    (time_s <= inference.duration_s).then(|| {
        let downbeat = inference.downbeat_probabilities[frame];
        ObservedBeat {
            time_s,
            confidence: f64::from(pulse_probability.clamp(0.0, 1.0)),
            downbeat_confidence: f64::from(downbeat.clamp(0.0, 1.0)),
        }
    })
}

#[allow(clippy::float_cmp)]
fn local_maxima(values: &[f32]) -> Vec<usize> {
    let mut maxima = Vec::new();
    let mut index = 0;
    while index < values.len() {
        let plateau_start = index;
        while index + 1 < values.len() && values[index + 1] == values[plateau_start] {
            index += 1;
        }
        let plateau_end = index;
        let value = values[plateau_start];
        let left = plateau_start
            .checked_sub(1)
            .map_or(f32::NEG_INFINITY, |i| values[i]);
        let right = values
            .get(plateau_end + 1)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        if value > left && value > right {
            maxima.push(usize::midpoint(plateau_start, plateau_end));
        }
        index += 1;
    }
    maxima
}

fn decode_activation_path(probabilities: &[f32], candidates: &[usize]) -> Vec<usize> {
    if probabilities.is_empty() || candidates.is_empty() {
        return Vec::new();
    }
    let path = viterbi_beat_path(probabilities);
    let mut snapped = Vec::new();
    let mut last = None;
    for frame in path {
        let candidate = candidates
            .iter()
            .copied()
            .filter(|candidate| last.is_none_or(|last| *candidate > last))
            .filter(|candidate| candidate.abs_diff(frame) <= MAXIMUM_PEAK_CORRECTION_FRAMES)
            .min_by_key(|candidate| candidate.abs_diff(frame));
        if let Some(candidate) = candidate {
            snapped.push(candidate);
            last = Some(candidate);
        }
    }
    snapped
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn viterbi_beat_path(probabilities: &[f32]) -> Vec<usize> {
    let minimum_period = (60.0 * FRAME_RATE_HZ / MAXIMUM_BPM).ceil() as usize;
    let maximum_period = (60.0 * FRAME_RATE_HZ / MINIMUM_BPM).floor() as usize;
    let periods = (minimum_period..=maximum_period).collect::<Vec<_>>();
    let mut offsets = Vec::with_capacity(periods.len());
    let mut total_states = 0;
    for &period in &periods {
        offsets.push(total_states);
        total_states += period;
    }

    let (beat_emission, nonbeat_emission) = activation_emissions(probabilities[0]);
    let mut scores = vec![f64::NEG_INFINITY; total_states];
    for (&period, &offset) in periods.iter().zip(&offsets) {
        scores[offset] = beat_emission;
        scores[offset + 1..offset + period].fill(nonbeat_emission);
    }
    let mut back_periods = vec![u16::MAX; probabilities.len() * periods.len()];

    for (frame, &probability) in probabilities.iter().enumerate().skip(1) {
        let (beat_emission, nonbeat_emission) = activation_emissions(probability);
        let mut next = vec![f64::NEG_INFINITY; total_states];
        for (period_index, (&period, &offset)) in periods.iter().zip(&offsets).enumerate() {
            for phase in 1..period {
                next[offset + phase] = scores[offset + phase - 1] + nonbeat_emission;
            }
            let mut best_transition = f64::NEG_INFINITY;
            let mut best_period_index = 0;
            for (source_index, (&source_period, &source_offset)) in
                periods.iter().zip(&offsets).enumerate()
            {
                let log_ratio = (usize_to_f64(period) / usize_to_f64(source_period)).ln();
                let penalty = TEMPO_CHANGE_PENALTY * log_ratio * log_ratio;
                let score = scores[source_offset + source_period - 1] - penalty;
                if score > best_transition {
                    best_transition = score;
                    best_period_index = source_index;
                }
            }
            next[offset] = best_transition + beat_emission;
            back_periods[frame * periods.len() + period_index] =
                u16::try_from(best_period_index).expect("period state count fits u16");
        }
        scores = next;
    }

    let mut period_index = 0;
    let mut phase = 0;
    let mut terminal_best = f64::NEG_INFINITY;
    for (candidate_period, (&period, &offset)) in periods.iter().zip(&offsets).enumerate() {
        for candidate_phase in 0..period {
            let score = scores[offset + candidate_phase];
            if score > terminal_best {
                period_index = candidate_period;
                phase = candidate_phase;
                terminal_best = score;
            }
        }
    }
    let mut beats = Vec::new();
    for frame in (0..probabilities.len()).rev() {
        if phase == 0 {
            beats.push(frame);
        }
        if frame == 0 {
            break;
        }
        if phase == 0 {
            period_index = usize::from(back_periods[frame * periods.len() + period_index]);
            phase = periods[period_index] - 1;
        } else {
            phase -= 1;
        }
    }
    beats.reverse();
    beats
}

fn activation_emissions(probability: f32) -> (f64, f64) {
    let probability = f64::from(probability).clamp(1e-7, 1.0 - 1e-7);
    (probability.ln() + BEAT_STATE_BIAS, (-probability).ln_1p())
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_maxima_collapse_plateaus() {
        assert_eq!(local_maxima(&[0.0, 0.2, 0.2, 0.1, 0.3, 0.0]), [1, 4]);
    }

    #[test]
    fn activation_path_snaps_only_to_real_peaks() {
        let mut probabilities = vec![0.01; 100];
        for frame in [10, 30, 50, 70, 90] {
            probabilities[frame] = 0.95;
        }
        let candidates = local_maxima(&probabilities);
        let decoded = decode_activation_path(&probabilities, &candidates);
        assert_eq!(decoded, [10, 30, 50, 70, 90]);
    }

    #[test]
    fn frame_rate_matches_upstream_contract() {
        assert!((FRAME_RATE_HZ - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn external_model_smoke_test_when_configured() {
        let Some(path) = std::env::var_os("RHYTHM_MAP_BEATNET_MODEL") else {
            return;
        };
        let mut backend = BeatNetBackend::load(path).unwrap();
        let samples = vec![0.0_f32; SAMPLE_RATE_HZ as usize];
        let inference = backend.infer_mono(&samples, SAMPLE_RATE_HZ).unwrap();
        assert_eq!(inference.beat_probabilities.len(), 50);
        for frame in 0..50 {
            let sum = inference.beat_probabilities[frame]
                + inference.downbeat_probabilities[frame]
                + inference.nonbeat_probabilities[frame];
            assert!((sum - 1.0).abs() < 1e-4);
        }
    }
}
