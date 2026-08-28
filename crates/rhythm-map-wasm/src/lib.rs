//! WASM bindings for training-free timing analysis from host observations.

use rhythm_map_core::{
    ANALYSIS_SCHEMA_VERSION, Analysis, BackendError, Engine, ModelInfo, ObservedBeat,
    RhythmObservationBackend, RhythmObservations,
    analyze_observations as analyze_core_observations,
};
use wasm_bindgen::prelude::*;

/// Return the serialized analysis schema version.
#[wasm_bindgen]
#[must_use]
pub fn schema_version() -> u32 {
    ANALYSIS_SCHEMA_VERSION
}

/// Convert beat and downbeat timestamps into a tempo map.
///
/// End-to-end browser audio inference will use the same observation boundary in
/// a later milestone; this function is already useful with host-provided beat
/// observations and keeps the WASM result schema identical to native output.
///
/// # Errors
///
/// Returns a JavaScript error when timestamps are invalid or serialization
/// fails.
#[wasm_bindgen]
#[allow(clippy::needless_pass_by_value)]
pub fn analyze_timing(
    beat_times_s: Vec<f64>,
    downbeat_times_s: Vec<f64>,
    duration_s: f64,
) -> Result<JsValue, JsValue> {
    let observations = observations_from_timing(beat_times_s, &downbeat_times_s, duration_s);
    analyze_to_js(&observations)
}

/// Analyze a complete backend-neutral observation object.
///
/// This preserves confidence, candidate, activation, activity, onset, harmonic,
/// and source fields supplied by the host instead of reducing input to bare
/// timestamps.
///
/// # Errors
///
/// Returns a JavaScript error when the object does not match
/// `RhythmObservations`, timing evidence is invalid, or serialization fails.
#[wasm_bindgen]
#[allow(clippy::needless_pass_by_value)]
pub fn analyze_observations(observations: JsValue) -> Result<JsValue, JsValue> {
    let observations: RhythmObservations = serde_wasm_bindgen::from_value(observations)
        .map_err(|error| js_error(format!("invalid RhythmObservations: {error}")))?;
    analyze_to_js(&observations)
}

/// Analyze host observations while enriching them from decoded interleaved PCM.
///
/// Beat and downbeat observations still come from the host. The shared Rust
/// engine downmixes PCM and adds deterministic activity, spectral-onset, and
/// supported harmonic-change evidence before running the shipping estimator.
///
/// # Errors
///
/// Returns a JavaScript error for invalid observations, PCM layout, duration
/// mismatch, timing evidence, or serialization failure.
#[wasm_bindgen]
#[allow(clippy::needless_pass_by_value)]
pub fn analyze_pcm_with_observations(
    observations: JsValue,
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
) -> Result<JsValue, JsValue> {
    let observations: RhythmObservations = serde_wasm_bindgen::from_value(observations)
        .map_err(|error| js_error(format!("invalid RhythmObservations: {error}")))?;
    let mut engine = Engine::new(HostObservationBackend { observations });
    let analysis = engine
        .analyze_pcm(&samples, sample_rate, channels)
        .map_err(|error| js_error(error.to_string()))?;
    analysis_to_js(&analysis)
}

fn observations_from_timing(
    beat_times_s: Vec<f64>,
    downbeat_times_s: &[f64],
    duration_s: f64,
) -> RhythmObservations {
    let beats = beat_times_s
        .into_iter()
        .map(|time_s| {
            let downbeat_confidence = if downbeat_times_s
                .iter()
                .any(|downbeat| (downbeat - time_s).abs() <= 0.07)
            {
                1.0
            } else {
                0.0
            };
            ObservedBeat {
                time_s,
                confidence: 1.0,
                downbeat_confidence,
            }
        })
        .collect();
    RhythmObservations {
        duration_s,
        beats,
        beat_candidates: Vec::new(),
        activations: None,
        activity: Vec::new(),
        onsets: Vec::new(),
        harmonic_changes: Vec::new(),
        source: ModelInfo {
            backend: "wasm-host-observations".to_string(),
            model: "host".to_string(),
            version: None,
            frame_rate_hz: None,
        },
    }
}

fn analyze_to_js(observations: &RhythmObservations) -> Result<JsValue, JsValue> {
    let analysis =
        analyze_core_observations(observations).map_err(|error| js_error(error.to_string()))?;
    analysis_to_js(&analysis)
}

fn analysis_to_js(analysis: &Analysis) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(analysis).map_err(|error| js_error(error.to_string()))
}

fn js_error(message: impl AsRef<str>) -> JsValue {
    JsValue::from_str(message.as_ref())
}

#[derive(Clone)]
struct HostObservationBackend {
    observations: RhythmObservations,
}

impl RhythmObservationBackend for HostObservationBackend {
    fn observe_mono(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<RhythmObservations, BackendError> {
        if sample_rate == 0 {
            return Err(BackendError::new("sample rate must be non-zero"));
        }
        let sample_count = u32::try_from(samples.len())
            .map_err(|_| BackendError::new("mono PCM has more than u32::MAX samples"))?;
        let pcm_duration_s = f64::from(sample_count) / f64::from(sample_rate);
        let tolerance_s = (1.0 / f64::from(sample_rate)).max(1e-6);
        if (self.observations.duration_s - pcm_duration_s).abs() > tolerance_s {
            return Err(BackendError::new(format!(
                "observation duration {:.9}s does not match PCM duration {:.9}s",
                self.observations.duration_s, pcm_duration_s
            )));
        }
        Ok(self.observations.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_facade_uses_the_current_core_schema() {
        let observations = observations_from_timing(
            (0..32).map(|index| f64::from(index) * 0.5).collect(),
            &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
            16.0,
        );
        let analysis = analyze_core_observations(&observations).unwrap();

        assert_eq!(analysis.schema_version, ANALYSIS_SCHEMA_VERSION);
        assert_eq!(analysis.source.backend, "wasm-host-observations");
    }

    #[test]
    fn pcm_enrichment_rejects_a_duration_mismatch() {
        let mut backend = HostObservationBackend {
            observations: observations_from_timing(vec![0.0, 0.5], &[], 2.0),
        };
        let error = backend
            .observe_mono(&vec![0.0; 44_100], 44_100)
            .unwrap_err();

        assert!(error.to_string().contains("does not match PCM duration"));
    }
}
