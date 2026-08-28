//! Analyze backend-neutral beat observations without audio or model files.

use rhythm_map_core::{ModelInfo, ObservedBeat, RhythmObservations, analyze_observations};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut beats = Vec::new();
    let mut time_s = 0.0;

    append_constant_tempo(&mut beats, &mut time_s, 120.0, 16);
    append_constant_tempo(&mut beats, &mut time_s, 150.0, 20);

    let observations = RhythmObservations {
        duration_s: time_s,
        beats,
        beat_candidates: Vec::new(),
        activations: None,
        activity: Vec::new(),
        onsets: Vec::new(),
        harmonic_changes: Vec::new(),
        source: ModelInfo {
            backend: "example-observations".to_owned(),
            model: "hand-authored-120-to-150".to_owned(),
            version: None,
            frame_rate_hz: None,
        },
    };

    let analysis = analyze_observations(&observations)?;
    println!("{}", serde_json::to_string_pretty(&analysis)?);
    Ok(())
}

fn append_constant_tempo(
    beats: &mut Vec<ObservedBeat>,
    time_s: &mut f64,
    bpm: f64,
    beat_count: usize,
) {
    let interval_s = 60.0 / bpm;
    for index in 0..beat_count {
        beats.push(ObservedBeat {
            time_s: *time_s,
            confidence: 0.95,
            downbeat_confidence: if index % 4 == 0 { 0.9 } else { 0.05 },
        });
        *time_s += interval_s;
    }
}
