//! WASM bindings for training-free timing analysis from beat observations.

use rhythm_map_core::{
    ANALYSIS_SCHEMA_VERSION, ModelInfo, ObservedBeat, RhythmObservations, analyze_observations,
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
    let observations = RhythmObservations {
        duration_s,
        beats,
        beat_candidates: Vec::new(),
        activity: Vec::new(),
        onsets: Vec::new(),
        source: ModelInfo {
            backend: "wasm-host-observations".to_string(),
            model: "host".to_string(),
            version: None,
            frame_rate_hz: None,
        },
    };
    let analysis = analyze_observations(&observations)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serde_wasm_bindgen::to_value(&analysis).map_err(|error| JsValue::from_str(&error.to_string()))
}
