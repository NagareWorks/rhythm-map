use realfft::RealFftPlanner;
use thiserror::Error;

use crate::estimator::TempoMapEstimator;
use crate::{
    Analysis, AnalysisError, AudioActivityPoint, AudioHarmonicChangePoint, AudioOnsetPoint,
    RhythmObservations,
};

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
    /// estimator. Deterministic activity and spectral-flux onset envelopes are
    /// added when the backend does not provide them.
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
        if observations.onsets.is_empty() {
            observations.onsets = extract_audio_onsets(samples, sample_rate);
        }
        if self.estimator.requires_harmonic_changes()
            && observations.harmonic_changes.is_empty()
            && observations.beat_candidates.len() >= 8
        {
            let mut supported_times = observations
                .beats
                .iter()
                .map(|beat| beat.time_s)
                .chain(
                    observations
                        .beat_candidates
                        .iter()
                        .map(|candidate| candidate.time_s),
                )
                .collect::<Vec<_>>();
            supported_times.sort_by(f64::total_cmp);
            supported_times.dedup_by(|left, right| (*left - *right).abs() <= f64::EPSILON);
            observations.harmonic_changes =
                extract_harmonic_changes(samples, sample_rate, &supported_times);
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

#[allow(clippy::cast_possible_truncation)]
fn extract_audio_onsets(samples: &[f32], sample_rate: u32) -> Vec<AudioOnsetPoint> {
    if samples.is_empty() {
        return Vec::new();
    }
    let sample_rate = usize::try_from(sample_rate).expect("sample rate fits usize");
    let nominal_window = (sample_rate / 25).clamp(64, 8_192);
    let fft_size = nominal_window.next_power_of_two().min(8_192);
    let hop = (sample_rate / 100).max(1);
    let window_denominator = usize_to_f64(fft_size.saturating_sub(1).max(1));
    let window = (0..fft_size)
        .map(|index| {
            let phase = 2.0 * std::f64::consts::PI * usize_to_f64(index) / window_denominator;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect::<Vec<_>>();
    let mut planner = RealFftPlanner::<f32>::new();
    let transform = planner.plan_fft_forward(fft_size);
    let mut input = transform.make_input_vec();
    let mut spectrum = transform.make_output_vec();
    let mut previous_magnitudes = vec![0.0_f64; spectrum.len()];
    let mut onsets = Vec::with_capacity(samples.len().div_ceil(hop));
    let half_window = fft_size / 2;
    let mut first_frame = true;
    for center in (0..samples.len()).step_by(hop) {
        input.fill(0.0);
        let source_start = center.saturating_sub(half_window);
        let window_offset = half_window.saturating_sub(center);
        let available = (samples.len() - source_start).min(fft_size - window_offset);
        for (target, (&sample, &weight)) in input[window_offset..window_offset + available]
            .iter_mut()
            .zip(
                samples[source_start..source_start + available]
                    .iter()
                    .zip(&window[window_offset..window_offset + available]),
            )
        {
            *target = sample * weight;
        }
        transform
            .process(&mut input, &mut spectrum)
            .expect("real FFT buffers match the planned transform");
        let mut flux = 0.0;
        let mut band_flux = [0.0_f64; 3];
        for (bin_index, (bin, previous)) in spectrum
            .iter()
            .zip(&mut previous_magnitudes)
            .enumerate()
            .skip(1)
        {
            let magnitude = f64::from(bin.norm());
            if !first_frame {
                let positive_flux = (magnitude - *previous).max(0.0);
                flux += positive_flux;
                let frequency_hz =
                    usize_to_f64(bin_index) * usize_to_f64(sample_rate) / usize_to_f64(fft_size);
                let band = if frequency_hz < 250.0 {
                    0
                } else if frequency_hz <= 2_000.0 {
                    1
                } else {
                    2
                };
                band_flux[band] += positive_flux;
            }
            *previous = magnitude;
        }
        onsets.push(AudioOnsetPoint {
            time_s: usize_to_f64(center) / usize_to_f64(sample_rate),
            strength: flux.ln_1p(),
            low_strength: onset_band_share(band_flux[0], flux),
            mid_strength: onset_band_share(band_flux[1], flux),
            high_strength: onset_band_share(band_flux[2], flux),
        });
        first_frame = false;
    }
    let peak = onsets
        .iter()
        .map(|point| point.strength)
        .fold(0.0_f64, f64::max);
    for point in &mut onsets {
        point.strength = normalize_onset_strength(point.strength, peak);
        point.low_strength *= point.strength;
        point.mid_strength *= point.strength;
        point.high_strength *= point.strength;
    }
    onsets
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn extract_harmonic_changes(
    samples: &[f32],
    sample_rate: u32,
    supported_times: &[f64],
) -> Vec<AudioHarmonicChangePoint> {
    if samples.is_empty() || supported_times.is_empty() {
        return Vec::new();
    }
    let target_window = (f64::from(sample_rate) * 0.08).round() as usize;
    let fft_size = target_window.clamp(1_024, 8_192).next_power_of_two();
    let mut planner = RealFftPlanner::<f32>::new();
    let transform = planner.plan_fft_forward(fft_size);
    let mut input = transform.make_input_vec();
    let mut spectrum = transform.make_output_vec();
    let mut pitch_class_profile = |time_s: f64| {
        let center = (time_s.max(0.0) * f64::from(sample_rate)) as usize;
        let start = center.saturating_sub(fft_size / 2).min(samples.len());
        input.fill(0.0);
        let available = samples.len().saturating_sub(start).min(fft_size);
        for (index, (&sample, target)) in samples[start..start + available]
            .iter()
            .zip(&mut input)
            .enumerate()
        {
            let phase =
                2.0 * std::f32::consts::PI * index as f32 / fft_size.saturating_sub(1) as f32;
            *target = sample * (0.5 - 0.5 * phase.cos());
        }
        transform
            .process(&mut input, &mut spectrum)
            .expect("real FFT buffers match the planned transform");
        let mut profile = [0.0_f64; 12];
        for (bin_index, bin) in spectrum.iter().enumerate().skip(1) {
            let frequency_hz =
                usize_to_f64(bin_index) * f64::from(sample_rate) / usize_to_f64(fft_size);
            if !(55.0..=5_000.0).contains(&frequency_hz) {
                continue;
            }
            let midi_note = (69.0 + 12.0 * (frequency_hz / 440.0).log2()).round() as i32;
            let pitch_class =
                usize::try_from(midi_note.rem_euclid(12)).expect("pitch class is non-negative");
            profile[pitch_class] += f64::from(bin.norm()).ln_1p();
        }
        let norm = profile
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        (norm > 1e-12).then(|| {
            for value in &mut profile {
                *value /= norm;
            }
            profile
        })
    };
    supported_times
        .iter()
        .map(|&time_s| {
            let before = pitch_class_profile(time_s - 0.1);
            let after = pitch_class_profile(time_s + 0.1);
            let strength = before.zip(after).map_or(0.0, |(before, after)| {
                (1.0 - before
                    .iter()
                    .zip(after)
                    .map(|(left, right)| left * right)
                    .sum::<f64>())
                .clamp(0.0, 1.0)
            });
            AudioHarmonicChangePoint { time_s, strength }
        })
        .collect()
}

fn onset_band_share(band_flux: f64, total_flux: f64) -> f64 {
    if total_flux <= f64::EPSILON {
        0.0
    } else {
        (band_flux / total_flux).clamp(0.0, 1.0)
    }
}

fn normalize_onset_strength(value: f64, peak: f64) -> f64 {
    if peak <= f64::EPSILON {
        0.0
    } else {
        (value / peak).clamp(0.0, 1.0)
    }
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
                activations: None,
                activity: Vec::new(),
                onsets: Vec::new(),
                harmonic_changes: Vec::new(),
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
    fn adds_normalized_spectral_flux_onsets() {
        let mut samples = vec![0.0; 4_000];
        samples[2_000] = 1.0;
        let mut engine = Engine::new(RecordingBackend::default());

        let observations = engine.observe_pcm(&samples, 4_000, 1).unwrap();

        assert!(!observations.onsets.is_empty());
        assert!(observations.onsets.iter().all(|point| {
            point.time_s.is_finite()
                && point.strength.is_finite()
                && (0.0..=1.0).contains(&point.strength)
                && point.low_strength.is_finite()
                && (0.0..=1.0).contains(&point.low_strength)
                && point.mid_strength.is_finite()
                && (0.0..=1.0).contains(&point.mid_strength)
                && point.high_strength.is_finite()
                && (0.0..=1.0).contains(&point.high_strength)
                && (point.low_strength + point.mid_strength + point.high_strength - point.strength)
                    .abs()
                    <= 1e-9
        }));
        assert!(observations.onsets[0].strength.abs() <= f64::EPSILON);
        let strongest = observations
            .onsets
            .iter()
            .max_by(|left, right| left.strength.total_cmp(&right.strength))
            .unwrap();
        assert!(strongest.strength > 0.9);
        assert!((strongest.time_s - 0.5).abs() <= 0.02);
    }

    #[test]
    fn separates_low_and_high_frequency_onset_contributions() {
        fn strongest_burst_onset(frequency_hz: f32) -> AudioOnsetPoint {
            let sample_rate = 16_000_u32;
            let mut samples = vec![0.0_f32; 16_000];
            let mut phase = 0.0_f32;
            let phase_step = std::f32::consts::TAU * frequency_hz / 16_000.0;
            for sample in &mut samples[8_000..] {
                *sample = phase.cos();
                phase += phase_step;
            }
            extract_audio_onsets(&samples, sample_rate)
                .into_iter()
                .max_by(|left, right| left.strength.total_cmp(&right.strength))
                .unwrap()
        }

        let low = strongest_burst_onset(100.0);
        let high = strongest_burst_onset(4_000.0);

        assert!((low.time_s - 0.5).abs() <= 0.02);
        assert!((high.time_s - 0.5).abs() <= 0.02);
        assert!(low.low_strength > high.low_strength);
        assert!(high.high_strength > low.high_strength);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn harmonic_change_distinguishes_a_pitch_transition_from_a_stable_tone() {
        let sample_rate = 16_000_u32;
        let mut stable = Vec::with_capacity(32_000);
        let mut changed = Vec::with_capacity(32_000);
        for index in 0..32_000 {
            let time_s = usize_to_f64(index) / f64::from(sample_rate);
            stable.push((std::f64::consts::TAU * 440.0 * time_s).sin() as f32);
            let frequency = if time_s < 1.0 { 440.0 } else { 523.251 };
            changed.push((std::f64::consts::TAU * frequency * time_s).sin() as f32);
        }

        let stable_change = extract_harmonic_changes(&stable, sample_rate, &[1.0])[0].strength;
        let pitch_change = extract_harmonic_changes(&changed, sample_rate, &[1.0])[0].strength;

        assert!(stable_change < 0.05);
        assert!(pitch_change > stable_change + 0.1);
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
