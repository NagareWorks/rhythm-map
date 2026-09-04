//! Synthetic API regressions, not copies of calibration recordings.
//!
//! A hypothesis score measures evidence/regularity, not correctness. In
//! particular, a sparser path can score higher while losing genuine beats.

use rhythm_map_core::{
    Analysis, AudioActivityPoint, AudioHarmonicChangePoint, BeatCandidate, BeatSequenceHypothesis,
    BeatSequenceHypothesisKind, ModelInfo, ObservedBeat, RhythmObservations, analyze_observations,
};

fn alternating_observations() -> RhythmObservations {
    let beat_candidates = (0..=32)
        .map(|index| BeatCandidate {
            time_s: f64::from(index / 2) * 0.5 + f64::from(index % 2) * 0.2,
            confidence: 0.8,
            downbeat_confidence: 0.0,
        })
        .collect::<Vec<_>>();
    RhythmObservations {
        duration_s: 8.1,
        beats: beat_candidates
            .iter()
            .map(|event| ObservedBeat {
                time_s: event.time_s,
                confidence: event.confidence,
                downbeat_confidence: event.downbeat_confidence,
            })
            .collect(),
        harmonic_changes: beat_candidates
            .iter()
            .enumerate()
            .map(|(index, event)| AudioHarmonicChangePoint {
                time_s: event.time_s,
                strength: if index % 2 == 0 { 0.2 } else { 0.0 },
            })
            .collect(),
        beat_candidates,
        activations: None,
        activity: Vec::new(),
        onsets: Vec::new(),
        source: ModelInfo {
            backend: "test".into(),
            model: "hand-authored-alternating-intervals".into(),
            version: None,
            frame_rate_hz: None,
        },
    }
}

fn selected_times(analysis: &Analysis) -> Vec<f64> {
    analysis.beats.iter().map(|beat| beat.time_s).collect()
}

fn local_path(analysis: &Analysis) -> Option<&BeatSequenceHypothesis> {
    analysis
        .beat_hypotheses
        .iter()
        .find(|h| h.kind == BeatSequenceHypothesisKind::LocallyVarying)
}

fn assert_primary_preserved(input: &RhythmObservations, analysis: &Analysis) {
    let expected = input.beats.iter().map(|b| b.time_s).collect::<Vec<_>>();
    assert_eq!(selected_times(analysis), expected);
    for hypothesis in &analysis.beat_hypotheses {
        assert!(hypothesis.beat_times_s.windows(2).all(|p| p[0] < p[1]));
        assert!(hypothesis.beat_times_s.iter().all(|t| {
            input
                .beat_candidates
                .iter()
                .any(|event| event.time_s.to_bits() == t.to_bits())
        }));
    }
}

#[test]
fn higher_scoring_sparse_path_does_not_replace_genuine_primary_beats() {
    // All 33 input events are intended beats in this authored fixture. Harmonic
    // accents favor every other beat; they do not authorize deleting the rest.
    let input = alternating_observations();
    let analysis = analyze_observations(&input).unwrap();
    let alternative = local_path(&analysis).expect("regular sparse alternative exists");
    let selected = analysis
        .beat_hypotheses
        .iter()
        .find(|h| h.kind == BeatSequenceHypothesisKind::Selected)
        .unwrap();
    assert!(alternative.relative_score > selected.relative_score);
    assert!(alternative.beat_times_s.len() < selected.beat_times_s.len());
    assert_primary_preserved(&input, &analysis);
    assert!(!analysis.metrical_ambiguity_regions.is_empty());
}

#[test]
fn file_padding_does_not_invent_beats_when_full_duration_path_is_unavailable() {
    assert!(local_path(&analyze_observations(&alternating_observations()).unwrap()).is_some());
    for (head_s, tail_s) in [(3.0, 0.0), (0.0, 3.0)] {
        let mut input = alternating_observations();
        for event in &mut input.beats {
            event.time_s += head_s;
        }
        for event in &mut input.beat_candidates {
            event.time_s += head_s;
        }
        for point in &mut input.harmonic_changes {
            point.time_s += head_s;
        }
        input.duration_s += head_s + tail_s;
        input.activity = (0..=111)
            .map(|i| {
                let time_s = (f64::from(i) * 0.1).min(input.duration_s);
                let audible = (head_s..=head_s + 8.1).contains(&time_s);
                AudioActivityPoint {
                    time_s,
                    rms: if audible { 1.0 } else { 0.0 },
                    relative_db: if audible { 0.0 } else { -80.0 },
                }
            })
            .collect();
        let analysis = analyze_observations(&input).unwrap();
        assert_primary_preserved(&input, &analysis);
        // Characterizes the current full-duration path limitation, not a promise
        // that future active-region candidate generators must omit alternatives.
        assert!(local_path(&analysis).is_none());
    }
}

#[test]
fn an_internal_rest_is_not_filled_to_connect_a_candidate_path() {
    let mut input = alternating_observations();
    for event in &mut input.beats {
        if event.time_s >= 4.0 {
            event.time_s += 3.0;
        }
    }
    for event in &mut input.beat_candidates {
        if event.time_s >= 4.0 {
            event.time_s += 3.0;
        }
    }
    for point in &mut input.harmonic_changes {
        if point.time_s >= 4.0 {
            point.time_s += 3.0;
        }
    }
    input.duration_s += 3.0;
    input.activity = (0..=111)
        .map(|i| {
            let time_s = (f64::from(i) * 0.1).min(input.duration_s);
            let resting = (4.0..7.0).contains(&time_s);
            AudioActivityPoint {
                time_s,
                rms: if resting { 0.0 } else { 1.0 },
                relative_db: if resting { -80.0 } else { 0.0 },
            }
        })
        .collect();
    let analysis = analyze_observations(&input).unwrap();
    assert_primary_preserved(&input, &analysis);
    assert!(
        analysis
            .beats
            .iter()
            .all(|b| !(4.0..7.0).contains(&b.time_s))
    );
    assert!(local_path(&analysis).is_none());
}

#[test]
fn an_unscorable_short_primary_is_not_replaced_by_dense_candidates() {
    let mut input = alternating_observations();
    input.beats.truncate(2);
    let analysis = analyze_observations(&input).unwrap();
    assert_primary_preserved(&input, &analysis);
    assert!(analysis.global_bpm.is_none());
    let selected = analysis
        .beat_hypotheses
        .iter()
        .find(|h| h.kind == BeatSequenceHypothesisKind::Selected)
        .unwrap();
    // Public output has a single relative rank, not a calibrated confidence.
    // This is distinct from the private scoring helper's zero sentinel.
    assert!((selected.relative_score - 1.0).abs() < f64::EPSILON);
    assert_eq!(analysis.beat_hypotheses.len(), 1);
    assert!(analysis.tempo_curve.is_empty());
    assert!(
        analysis
            .warnings
            .iter()
            .any(|w| w == "too_few_beats_for_tempo_curve")
    );
}
