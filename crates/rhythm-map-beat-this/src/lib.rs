//! Adapter from the MIT-licensed Beat This Rust port to backend-neutral events.

use std::path::Path;

use beat_this::{BeatThis, RtenRuntime, Runtime};
use rhythm_map_core::{
    BackendError, ModelInfo, ObservedBeat, RhythmObservationBackend, RhythmObservations,
};

type DefaultModel = <RtenRuntime as Runtime>::Model;

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
        })
    }
}

impl RhythmObservationBackend for BeatThisBackend {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn observe_mono(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<RhythmObservations, BackendError> {
        let result = self
            .tracker
            .analyze_audio(samples, sample_rate)
            .map_err(|error| BackendError::new(format!("Beat This inference failed: {error}")))?;

        let beats = result
            .beats
            .iter()
            .map(|&time| {
                let frame = ((f64::from(time) * 50.0).round() as usize)
                    .min(result.beat_logits.len().saturating_sub(1));
                let confidence = result.beat_logits.get(frame).copied().map_or(0.5, sigmoid);
                let nearest_downbeat = result
                    .downbeats
                    .iter()
                    .map(|&downbeat| (downbeat, (downbeat - time).abs()))
                    .min_by(|left, right| left.1.total_cmp(&right.1));
                let downbeat_confidence = nearest_downbeat.map_or(0.0, |(_, distance)| {
                    if distance <= 0.07 {
                        result
                            .downbeat_logits
                            .get(frame)
                            .copied()
                            .map_or(0.75, sigmoid)
                            .max(0.5)
                    } else {
                        0.0
                    }
                });
                ObservedBeat {
                    time_s: f64::from(time),
                    confidence,
                    downbeat_confidence,
                }
            })
            .collect();

        Ok(RhythmObservations {
            duration_s: usize_to_f64(samples.len()) / f64::from(sample_rate),
            beats,
            activity: Vec::new(),
            source: ModelInfo {
                backend: "beat-this-rten".to_string(),
                model: self.model_name.clone(),
                version: None,
                frame_rate_hz: Some(50.0),
            },
        })
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

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}
