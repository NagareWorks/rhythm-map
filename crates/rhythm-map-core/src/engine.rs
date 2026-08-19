use thiserror::Error;

use crate::{Analysis, AnalysisError, RhythmObservations, TempoMapEstimator};

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
    /// Compose an engine from explicit, independently replaceable components.
    #[must_use]
    pub const fn new(backend: B, estimator: TempoMapEstimator) -> Self {
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

        let observations = self.backend.observe_mono(samples, sample_rate)?;
        Ok(self.estimator.estimate(&observations)?)
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
        let mut engine = Engine::new(RecordingBackend::default(), TempoMapEstimator::default());
        engine.analyze_pcm(&[1.0, -1.0, 0.5, 0.25], 2, 2).unwrap();
        assert_eq!(engine.backend().last_samples, [0.0, 0.375]);
    }

    #[test]
    fn rejects_invalid_interleaving() {
        let mut engine = Engine::new(RecordingBackend::default(), TempoMapEstimator::default());
        assert!(matches!(
            engine.analyze_pcm(&[0.0, 1.0, 2.0], 48_000, 2),
            Err(EngineError::InvalidAudio(_))
        ));
    }
}
