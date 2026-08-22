//! Adapter from the MIT-licensed Beat This Rust port to backend-neutral events.

use std::path::Path;

use beat_this::{BeatThis, RtenRuntime, Runtime};
use rhythm_map_core::{
    BackendError, ModelInfo, ObservedBeat, RhythmObservationBackend, RhythmObservations,
};

type DefaultModel = <RtenRuntime as Runtime>::Model;
const FRAME_RATE_HZ: f64 = 50.0;

/// Configurable peak-picking policy applied to Beat This frame logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakPickingOptions {
    /// Strict lower logit bound for a peak. Zero equals probability 0.5.
    pub logit_threshold: f32,
    /// Number of frames inspected on either side of a candidate maximum.
    pub local_max_radius_frames: usize,
    /// Adjacent peak indices at or below this distance are averaged together.
    pub deduplicate_width_frames: usize,
}

impl Default for PeakPickingOptions {
    fn default() -> Self {
        Self {
            logit_threshold: 0.0,
            local_max_radius_frames: 3,
            deduplicate_width_frames: 1,
        }
    }
}

/// Conservative sequence decoder for weak peaks between strong Beat This events.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SupportedMidpointOptions {
    /// Strict lower logit bound for weak candidate peaks.
    pub candidate_logit_threshold: f32,
    /// Maximum distance from an interval midpoint as a fraction of that interval.
    pub maximum_midpoint_offset_ratio: f64,
    /// Number of strong-beat gaps inspected on either side for run support.
    pub support_radius_gaps: usize,
    /// Minimum supported gaps required inside the local support window.
    pub minimum_supported_gaps: usize,
}

/// Deployable decoding policy used by [`BeatThisBackend`].
///
/// The upstream policy remains the default. Alternative policies are explicit
/// so calibration can exercise the complete product path without changing the
/// behavior of CLI, FFI, or WASM consumers.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum BeatThisDecoderPolicy {
    /// Match the decoder shipped by the upstream Rust port.
    #[default]
    Upstream,
    /// Apply an explicit peak-picking configuration.
    PeakPicking(PeakPickingOptions),
    /// Recover repeated weak model peaks between strong events.
    SupportedMidpoints(SupportedMidpointOptions),
}

impl Default for SupportedMidpointOptions {
    fn default() -> Self {
        Self {
            candidate_logit_threshold: -3.0,
            maximum_midpoint_offset_ratio: 0.15,
            support_radius_gaps: 2,
            minimum_supported_gaps: 3,
        }
    }
}

/// One model inference retained before discrete peak decoding.
#[derive(Debug, Clone)]
pub struct BeatThisInference {
    duration_s: f64,
    beat_logits: Vec<f32>,
    downbeat_logits: Vec<f32>,
}

impl BeatThisInference {
    /// Per-frame beat logits at 50 frames per second.
    #[must_use]
    pub fn beat_logits(&self) -> &[f32] {
        &self.beat_logits
    }

    /// Per-frame downbeat logits at 50 frames per second.
    #[must_use]
    pub fn downbeat_logits(&self) -> &[f32] {
        &self.downbeat_logits
    }
}

/// Decoded mono audio returned by the convenience file adapter.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Mono PCM samples.
    pub samples: Vec<f32>,
    /// PCM sample rate.
    pub sample_rate: u32,
}

/// Beat This implementation of the observation boundary.
pub struct BeatThisBackend {
    tracker: BeatThis<DefaultModel>,
    model_name: String,
    decoder_policy: BeatThisDecoderPolicy,
}

impl BeatThisBackend {
    /// Load the frontend and beat/downbeat ONNX graphs using pure-Rust `rten`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when either model cannot be loaded.
    pub fn load(
        mel_model_path: impl AsRef<Path>,
        beat_model_path: impl AsRef<Path>,
    ) -> Result<Self, BackendError> {
        let beat_model_path = beat_model_path.as_ref();
        let tracker = BeatThis::new(&RtenRuntime, mel_model_path.as_ref(), beat_model_path)
            .map_err(|error| {
                BackendError::new(format!("failed to load Beat This models: {error}"))
            })?;
        let model_name = beat_model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("beat_this")
            .to_string();
        Ok(Self {
            tracker,
            model_name,
            decoder_policy: BeatThisDecoderPolicy::default(),
        })
    }

    /// Select an explicit deployable decoder policy.
    #[must_use]
    pub const fn with_decoder_policy(mut self, policy: BeatThisDecoderPolicy) -> Self {
        self.decoder_policy = policy;
        self
    }

    /// Run the neural frontend and model while retaining undecoded frame logits.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when model inference fails.
    pub fn infer_mono(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<BeatThisInference, BackendError> {
        if sample_rate == 0 {
            return Err(BackendError::new("sample rate must be greater than zero"));
        }
        let result = self
            .tracker
            .analyze_audio(samples, sample_rate)
            .map_err(|error| BackendError::new(format!("Beat This inference failed: {error}")))?;
        Ok(BeatThisInference {
            duration_s: usize_to_f64(samples.len()) / f64::from(sample_rate),
            beat_logits: result.beat_logits,
            downbeat_logits: result.downbeat_logits,
        })
    }

    /// Decode one retained inference with an explicit peak-picking policy.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for a non-finite threshold or mismatched logits.
    pub fn decode_inference(
        &self,
        inference: &BeatThisInference,
        options: PeakPickingOptions,
    ) -> Result<RhythmObservations, BackendError> {
        if !options.logit_threshold.is_finite() {
            return Err(BackendError::new("peak logit threshold must be finite"));
        }
        if inference.beat_logits.len() != inference.downbeat_logits.len() {
            return Err(BackendError::new(
                "Beat This beat and downbeat logits have different lengths",
            ));
        }
        let beat_frames = find_peaks(&inference.beat_logits, options);
        let downbeat_frames = find_peaks(&inference.downbeat_logits, options);
        Ok(self.observations_from_frames(inference, &beat_frames, &downbeat_frames))
    }

    /// Decode upstream peaks and recover locally supported weak midpoint peaks.
    ///
    /// This candidate never invents a grid timestamp: every added event must be
    /// a radius-three local maximum above `candidate_logit_threshold`, close to
    /// the midpoint of two upstream beats, and part of a locally repeated run.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for invalid options or mismatched logits.
    pub fn decode_inference_with_supported_midpoints(
        &self,
        inference: &BeatThisInference,
        options: SupportedMidpointOptions,
    ) -> Result<RhythmObservations, BackendError> {
        validate_midpoint_options(options)?;
        if inference.beat_logits.len() != inference.downbeat_logits.len() {
            return Err(BackendError::new(
                "Beat This beat and downbeat logits have different lengths",
            ));
        }
        let upstream_options = PeakPickingOptions::default();
        let upstream_beats = find_peaks(&inference.beat_logits, upstream_options);
        let candidate_options = PeakPickingOptions {
            logit_threshold: options.candidate_logit_threshold,
            ..upstream_options
        };
        let candidates = find_peaks(&inference.beat_logits, candidate_options);
        let beat_frames = recover_supported_midpoints(&upstream_beats, &candidates, options);
        let downbeat_frames = find_peaks(&inference.downbeat_logits, upstream_options);
        Ok(self.observations_from_frames(inference, &beat_frames, &downbeat_frames))
    }

    fn observations_from_frames(
        &self,
        inference: &BeatThisInference,
        beat_frames: &[f64],
        downbeat_frames: &[f64],
    ) -> RhythmObservations {
        let beats = beat_frames
            .iter()
            .map(|&frame| frame_to_time(frame))
            .collect::<Vec<_>>();
        let mut downbeats = downbeat_frames
            .iter()
            .map(|&frame| frame_to_time(frame))
            .collect::<Vec<_>>();
        snap_to_beats(&beats, &mut downbeats);

        RhythmObservations {
            duration_s: inference.duration_s,
            beats: observations_from_events(
                &beats,
                &downbeats,
                &inference.beat_logits,
                &inference.downbeat_logits,
            ),
            activity: Vec::new(),
            source: ModelInfo {
                backend: "beat-this-rten".to_string(),
                model: self.model_name.clone(),
                version: None,
                frame_rate_hz: Some(FRAME_RATE_HZ),
            },
        }
    }
}

impl RhythmObservationBackend for BeatThisBackend {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn observe_mono(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<RhythmObservations, BackendError> {
        let inference = self.infer_mono(samples, sample_rate)?;
        match self.decoder_policy {
            BeatThisDecoderPolicy::Upstream => {
                self.decode_inference(&inference, PeakPickingOptions::default())
            }
            BeatThisDecoderPolicy::PeakPicking(options) => {
                self.decode_inference(&inference, options)
            }
            BeatThisDecoderPolicy::SupportedMidpoints(options) => {
                self.decode_inference_with_supported_midpoints(&inference, options)
            }
        }
    }
}

/// Decode a supported audio file to mono PCM for CLI and GUI adapters.
///
/// # Errors
///
/// Returns [`BackendError`] when the file cannot be decoded or resampled.
pub fn decode_audio(path: impl AsRef<Path>) -> Result<DecodedAudio, BackendError> {
    let audio = beat_this::load_audio(path.as_ref(), 22_050)
        .map_err(|error| BackendError::new(format!("failed to decode audio: {error}")))?;
    Ok(DecodedAudio {
        samples: audio.samples,
        sample_rate: audio.sample_rate,
    })
}

fn sigmoid(value: f32) -> f64 {
    1.0 / (1.0 + (-f64::from(value)).exp())
}

#[allow(clippy::cast_possible_truncation)]
fn frame_to_time(frame: f64) -> f64 {
    f64::from((frame / FRAME_RATE_HZ) as f32)
}

fn find_peaks(logits: &[f32], options: PeakPickingOptions) -> Vec<f64> {
    let candidates = logits
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| {
            if value <= options.logit_threshold {
                return None;
            }
            let start = index.saturating_sub(options.local_max_radius_frames);
            let end = index
                .saturating_add(options.local_max_radius_frames)
                .saturating_add(1)
                .min(logits.len());
            logits[start..end]
                .iter()
                .all(|&neighbor| neighbor <= value)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    deduplicate_peaks(&candidates, options.deduplicate_width_frames)
}

fn validate_midpoint_options(options: SupportedMidpointOptions) -> Result<(), BackendError> {
    if !options.candidate_logit_threshold.is_finite() {
        return Err(BackendError::new(
            "midpoint candidate logit threshold must be finite",
        ));
    }
    if !options.maximum_midpoint_offset_ratio.is_finite()
        || !(0.0..=0.5).contains(&options.maximum_midpoint_offset_ratio)
    {
        return Err(BackendError::new(
            "maximum midpoint offset ratio must be finite and between zero and 0.5",
        ));
    }
    if options.minimum_supported_gaps == 0 {
        return Err(BackendError::new(
            "minimum supported midpoint gaps must be greater than zero",
        ));
    }
    Ok(())
}

fn recover_supported_midpoints(
    upstream_beats: &[f64],
    candidates: &[f64],
    options: SupportedMidpointOptions,
) -> Vec<f64> {
    let supported = upstream_beats
        .windows(2)
        .map(|pair| midpoint_candidate(pair[0], pair[1], candidates, options))
        .collect::<Vec<_>>();
    let mut recovered = upstream_beats.to_vec();
    for (gap_index, candidate) in supported.iter().enumerate() {
        let Some(candidate) = candidate else {
            continue;
        };
        let start = gap_index.saturating_sub(options.support_radius_gaps);
        let end = gap_index
            .saturating_add(options.support_radius_gaps)
            .saturating_add(1)
            .min(supported.len());
        let support_count = supported[start..end]
            .iter()
            .filter(|candidate| candidate.is_some())
            .count();
        if support_count >= options.minimum_supported_gaps {
            recovered.push(*candidate);
        }
    }
    recovered.sort_by(f64::total_cmp);
    recovered.dedup();
    recovered
}

fn midpoint_candidate(
    left: f64,
    right: f64,
    candidates: &[f64],
    options: SupportedMidpointOptions,
) -> Option<f64> {
    let gap = right - left;
    if gap <= 2.0 {
        return None;
    }
    let midpoint = left.midpoint(right);
    let maximum_offset = gap * options.maximum_midpoint_offset_ratio;
    candidates
        .iter()
        .copied()
        .filter(|&candidate| candidate > left + 1.0 && candidate < right - 1.0)
        .filter(|&candidate| (candidate - midpoint).abs() <= maximum_offset)
        .min_by(|left, right| {
            (*left - midpoint)
                .abs()
                .total_cmp(&(*right - midpoint).abs())
        })
}

fn deduplicate_peaks(peaks: &[usize], width: usize) -> Vec<f64> {
    let Some(&first) = peaks.first() else {
        return Vec::new();
    };
    let mut result = Vec::new();
    let mut running_mean = usize_to_f64(first);
    let mut count = 1.0;
    for &candidate in &peaks[1..] {
        let candidate = usize_to_f64(candidate);
        if candidate - running_mean <= usize_to_f64(width) {
            count += 1.0;
            running_mean += (candidate - running_mean) / count;
        } else {
            result.push(running_mean);
            running_mean = candidate;
            count = 1.0;
        }
    }
    result.push(running_mean);
    result
}

fn snap_to_beats(beats: &[f64], downbeats: &mut Vec<f64>) {
    for downbeat in downbeats.iter_mut() {
        if let Some(beat) = beats.iter().min_by(|left, right| {
            (*left - *downbeat)
                .abs()
                .total_cmp(&(*right - *downbeat).abs())
        }) {
            *downbeat = *beat;
        }
    }
    downbeats.sort_by(f64::total_cmp);
    downbeats.dedup();
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn observations_from_events(
    beats: &[f64],
    downbeats: &[f64],
    beat_logits: &[f32],
    downbeat_logits: &[f32],
) -> Vec<ObservedBeat> {
    beats
        .iter()
        .map(|&time_s| {
            let frame = ((time_s * FRAME_RATE_HZ).round() as usize)
                .min(beat_logits.len().saturating_sub(1));
            let downbeat = downbeats
                .iter()
                .any(|&candidate| (candidate - time_s).abs() <= 0.07);
            ObservedBeat {
                time_s,
                confidence: beat_logits.get(frame).copied().map_or(0.5, sigmoid),
                downbeat_confidence: if downbeat {
                    downbeat_logits
                        .get(frame)
                        .copied()
                        .map_or(0.75, sigmoid)
                        .max(0.5)
                } else {
                    0.0
                },
            }
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_peak_picker_matches_upstream_rules() {
        let mut logits = vec![-1.0; 40];
        logits[5] = 2.0;
        logits[20] = 1.0;
        logits[21] = 1.0;
        logits[30] = -0.1;

        assert_eq!(
            find_peaks(&logits, PeakPickingOptions::default()),
            vec![5.0, 20.5]
        );
    }

    #[test]
    fn lower_threshold_recovers_subzero_local_peak() {
        let logits = [-2.0, -0.5, -2.0];
        let options = PeakPickingOptions {
            logit_threshold: -1.0,
            ..PeakPickingOptions::default()
        };

        assert_eq!(find_peaks(&logits, options), vec![1.0]);
    }

    #[test]
    fn narrower_local_max_window_retains_nearby_peaks() {
        let logits = [-2.0, 1.0, -1.0, -1.0, 0.5, -2.0];
        let narrow = PeakPickingOptions {
            local_max_radius_frames: 1,
            ..PeakPickingOptions::default()
        };

        assert_eq!(find_peaks(&logits, narrow), vec![1.0, 4.0]);
        assert_eq!(
            find_peaks(&logits, PeakPickingOptions::default()),
            vec![1.0]
        );
    }

    #[test]
    fn extreme_radius_does_not_overflow() {
        let logits = [-1.0, 1.0, -1.0];
        let options = PeakPickingOptions {
            local_max_radius_frames: usize::MAX,
            ..PeakPickingOptions::default()
        };

        assert_eq!(find_peaks(&logits, options), vec![1.0]);
    }

    #[test]
    fn supported_midpoint_run_recovers_real_candidate_peaks() {
        let upstream = [0.0, 20.0, 40.0, 60.0, 80.0];
        let candidates = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        assert_eq!(
            recover_supported_midpoints(
                &upstream,
                &candidates,
                SupportedMidpointOptions::default()
            ),
            candidates
        );
    }

    #[test]
    fn isolated_midpoint_candidate_is_not_recovered() {
        let upstream = [0.0, 20.0, 40.0, 60.0, 80.0];
        let candidates = [0.0, 20.0, 30.0, 40.0, 60.0, 80.0];

        assert_eq!(
            recover_supported_midpoints(
                &upstream,
                &candidates,
                SupportedMidpointOptions::default()
            ),
            upstream
        );
    }
}
