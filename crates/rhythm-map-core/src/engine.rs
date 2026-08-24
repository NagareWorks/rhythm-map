use thiserror::Error;

use crate::estimator::TempoMapEstimator;
use crate::{Analysis, AnalysisError, AudioActivityPoint, RhythmObservations};

/// Error returned by an observation backend.
#[derive(Debug, Clone, Error)]
#[error("{message}")]
pub struct BackendError {
    message: String,
}

impl BackendError {
    /// Construct an error without exposing backend-specific error types.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Interchangeable beat/downbeat observation provider.
pub trait RhythmObservationBackend {
    /// Analyze mono PCM and return backend-neutral observations.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when model input processing or inference fails.
    fn observe_mono(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<RhythmObservations, BackendError>;
}

/// Failure from the end-to-end engine.
#[derive(Debug, Error)]
pub enum EngineError {
    /// Invalid PCM layout.
    #[error("invalid audio buffer: {0}")]
    InvalidAudio(String),
    /// Observation backend failure.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// Tempo-map analysis failure.
    #[error(transparent)]
    Analysis(#[from] AnalysisError),
}

/// End-to-end engine composed from an observation backend and timing estimator.
pub struct Engine<B> {
    backend: B,
    estimator: TempoMapEstimator,
}

impl<B> Engine<B>
where
    B: RhythmObservationBackend,
{
    /// Compose an engine with the single shipping analysis policy.
    #[must_use]
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            estimator: TempoMapEstimator::default(),
        }
    }

    /// Compose an evaluation engine with an explicit candidate estimator.
    ///
    /// This is intentionally feature-gated out of normal product builds. A
    /// candidate that passes promotion replaces the shipping policy instead of
    /// becoming another user-facing option.
    #[cfg(feature = "experimental-policies")]
    #[must_use]
    pub const fn with_estimator(backend: B, estimator: TempoMapEstimator) -> Self {
        Self { backend, estimator }
    }

    /// Analyze interleaved PCM. Multichannel input is averaged to mono.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] for invalid PCM layout, backend inference
    /// failures, or invalid timing observations.
    pub fn analyze_pcm(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<Analysis, EngineError> {
        let observations = self.observe_pcm(samples, sample_rate, channels)?;
        Ok(self.estimator.estimate(&observations)?)
    }

    /// Decode the backend-neutral observation layer without running the timing
    /// estimator. A deterministic activity envelope is added when the backend
    /// does not provide one.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] for invalid PCM layout or backend inference
    /// failures.
    pub fn observe_pcm(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        channels: u16,
    ) -> Result<RhythmObservations, EngineError> {
        if sample_rate == 0 {
            return Err(EngineError::InvalidAudio(
                "sample rate must be non-zero".to_string(),
            ));
        }
        if channels == 0 {
            return Err(EngineError::InvalidAudio(
                "channel count must be non-zero".to_string(),
            ));
        }
        let channel_count = usize::from(channels);
        if !samples.len().is_multiple_of(channel_count) {
            return Err(EngineError::InvalidAudio(
                "interleaved sample count is not divisible by channel count".to_string(),
            ));
        }

        let mono;
        let samples = if channel_count == 1 {
            samples
        } else {
            mono = samples
                .chunks_exact(channel_count)
                .map(|frame| frame.iter().copied().sum::<f32>() / f32::from(channels))
                .collect::<Vec<_>>();
            &mono
        };

        let mut observations = self.backend.observe_mono(samples, sample_rate)?;
        if observations.activity.is_empty() {
            observations.activity = extract_audio_activity(samples, sample_rate);
        }
        Ok(observations)
    }

    /// Run the deterministic estimator on an already captured observation set.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when observations or estimator policy are
    /// invalid.
    pub fn analyze_observations(
        &self,
        observations: &RhythmObservations,
    ) -> Result<Analysis, AnalysisError> {
        self.estimator.estimate(observations)
    }

    /// Borrow the backend for backend-specific lifecycle operations.
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Mutably borrow the backend.
    pub const fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

fn extract_audio_activity(samples: &[f32], sample_rate: u32) -> Vec<AudioActivityPoint> {
    if samples.is_empty() {
        return Vec::new();
    }
    let window = usize::try_from((sample_rate / 10).max(1)).expect("sample rate fits usize");
    let hop = usize::try_from((sample_rate / 20).max(1)).expect("sample rate fits usize");
    let mut activity = (0..samples.len())
        .step_by(hop)
        .map(|center| {
            let start = center.saturating_sub(window / 2);
            let end = (center + window / 2 + 1).min(samples.len());
            let mean_square = samples[start..end]
                .iter()
                .map(|sample| f64::from(*sample).powi(2))
                .sum::<f64>()
                / usize_to_f64(end - start);
            AudioActivityPoint {
                time_s: usize_to_f64(center) / f64::from(sample_rate),
                rms: mean_square.sqrt(),
                relative_db: 0.0,
            }
        })
        .collect::<Vec<_>>();
    let peak = activity
        .iter()
        .map(|point| point.rms)
        .fold(0.0_f64, f64::max);
    for point in &mut activity {
        point.relative_db = if peak <= f64::EPSILON {
            -120.0
        } else {
            (20.0 * (point.rms / peak).max(1e-6).log10()).clamp(-120.0, 0.0)
        };
    }
    activity
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelInfo, ObservedBeat};

    #[derive(Default)]
    struct RecordingBackend {
        last_samples: Vec<f32>,
    }

    impl RhythmObservationBackend for RecordingBackend {
        fn observe_mono(
            &mut self,
            samples: &[f32],
            _sample_rate: u32,
        ) -> Result<RhythmObservations, BackendError> {
            self.last_samples = samples.to_vec();
            Ok(RhythmObservations {
                duration_s: 1.0,
                beats: vec![
                    ObservedBeat {
                        time_s: 0.0,
                        confidence: 1.0,
                        downbeat_confidence: 1.0,
                    },
                    ObservedBeat {
                        time_s: 0.5,
                        confidence: 1.0,
                        downbeat_confidence: 0.0,
                    },
                ],
                beat_candidates: Vec::new(),
                activity: Vec::new(),
                source: ModelInfo {
                    backend: "test".to_string(),
                    model: "test".to_string(),
                    version: None,
                    frame_rate_hz: None,
                },
            })
        }
    }

    #[test]
    fn downmixes_interleaved_stereo() {
        let mut engine = Engine::new(RecordingBackend::default());
        engine.analyze_pcm(&[1.0, -1.0, 0.5, 0.25], 2, 2).unwrap();
        assert_eq!(engine.backend().last_samples, [0.0, 0.375]);
    }

    #[test]
    fn rejects_invalid_interleaving() {
        let mut engine = Engine::new(RecordingBackend::default());
        assert!(matches!(
            engine.analyze_pcm(&[0.0, 1.0, 2.0], 48_000, 2),
            Err(EngineError::InvalidAudio(_))
        ));
    }

    #[test]
    fn adds_activity_envelope_to_backend_observations() {
        let mut engine = Engine::new(RecordingBackend::default());
        let observations = engine.observe_pcm(&vec![0.5; 2_000], 1_000, 1).unwrap();
        assert!(!observations.activity.is_empty());
        assert!(
            observations
                .activity
                .iter()
                .all(|point| point.relative_db.abs() < 1e-9)
        );
    }

    #[test]
    fn engine_and_observation_facade_share_the_shipping_policy() {
        let mut engine = Engine::new(RecordingBackend::default());
        let observations = engine.observe_pcm(&[0.0; 2], 2, 1).unwrap();
        let engine_analysis = engine.analyze_observations(&observations).unwrap();
        let facade_analysis = crate::analyze_observations(&observations).unwrap();

        assert_eq!(engine_analysis, facade_analysis);
    }
}
