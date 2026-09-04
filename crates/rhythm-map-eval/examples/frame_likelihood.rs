//! Authored structural scoring gate. No audio or neural capture is consumed.
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};

#[path = "support/frame_likelihood.rs"]
mod likelihood;
use likelihood::{Evidence, State, score};

// Fixed authored geometry, not estimated timing or a production support model.
const FRAMES: usize = 1200;
const PERIOD: usize = 24;
const ORIGIN: usize = 5;
const RADIUS: usize = 1;

fn hypothesis(period: usize, meter: usize, phase: usize) -> Vec<State> {
    let mut states = vec![State::default(); FRAMES];
    for (beat, center) in (ORIGIN..FRAMES).step_by(period).enumerate() {
        for state in &mut states[center.saturating_sub(RADIUS)..(center + RADIUS + 1).min(FRAMES)] {
            state.beat = true;
            state.downbeat = beat % meter == phase;
        }
    }
    states
}

fn heads(states: &[State]) -> (Vec<f32>, Vec<f32>) {
    (
        states
            .iter()
            .map(|s| if s.beat { 8. } else { -8. })
            .collect(),
        states
            .iter()
            .map(|s| if s.downbeat { 8. } else { -8. })
            .collect(),
    )
}

/// Enumerates every meter/phase on the same given beat geometry. No truth enters
/// this ranking function. This is conditional scoring, not full clock decoding.
fn rank(beat: &[f32], downbeat: &[f32]) -> Result<Vec<(usize, usize, f64)>> {
    let evidence = Evidence {
        beat,
        downbeat,
        available: None,
    };
    let mut results = Vec::new();
    for meter in 2..=7 {
        for phase in 0..meter {
            let value = score(&evidence, &hypothesis(PERIOD, meter, phase))?;
            // Flat evidence may produce a density-based score preference. Never
            // interpret that preference as an identified meter or phase.
            if !value.downbeat_varies {
                return Ok(vec![]);
            }
            results.push((meter, phase, value.log_score));
        }
    }
    results.sort_by(|a, b| b.2.total_cmp(&a.2));
    Ok(results)
}

fn audit() -> Result<serde_json::Value> {
    let mut rows = Vec::new();
    for meter in 2..=7 {
        for phase in 0..meter {
            let (beat, downbeat) = heads(&hypothesis(PERIOD, meter, phase));
            let ranked = rank(&beat, &downbeat)?;
            ensure!(
                (ranked[0].0, ranked[0].1) == (meter, phase) && ranked[0].2 > ranked[1].2,
                "ideal meter/phase gate failed"
            );
            rows.push(json!({"authored_meter":meter,"authored_phase":phase,
                "best_meter":ranked[0].0,"best_phase":ranked[0].1,
                "score_margin_not_confidence":ranked[0].2-ranked[1].2}));
        }
    }
    let correct = hypothesis(PERIOD, 4, 0);
    let (beat, downbeat) = heads(&correct);
    let evidence = Evidence {
        beat: &beat,
        downbeat: &downbeat,
        available: None,
    };
    let reference_score = score(&evidence, &correct)?.log_score;
    let mut wrong = Vec::new();
    for (name, states) in [
        ("extra_halfway_bars", hypothesis(PERIOD, 2, 0)),
        ("missed_alternate_bars", hypothesis(PERIOD, 8, 0)),
        ("wrong_bar_phase", hypothesis(PERIOD, 4, 1)),
        ("extra_halfway_beats", hypothesis(PERIOD / 2, 4, 0)),
        ("missed_alternate_beats", hypothesis(PERIOD * 2, 4, 0)),
    ] {
        let value = score(&evidence, &states)?;
        ensure!(
            reference_score > value.log_score && value.scored_frames == FRAMES,
            "contradiction gate failed"
        );
        wrong.push(
            json!({"hypothesis":name,"score_loss_not_confidence":reference_score-value.log_score,
                          "scored_frames":value.scored_frames}),
        );
    }
    let mut corruptions = Vec::new();
    for (name, center, value) in [
        ("one_false_bar", ORIGIN + PERIOD * 2, 8.),
        ("one_erased_bar", ORIGIN + PERIOD * 4, -8.),
    ] {
        let mut altered = downbeat.clone();
        altered[center - RADIUS..=center + RADIUS].fill(value);
        let ranked = rank(&beat, &altered)?;
        ensure!(
            (ranked[0].0, ranked[0].1) == (4, 0) && ranked[0].2 > ranked[1].2,
            "isolated corruption gate failed"
        );
        corruptions.push(
            json!({"case":name,"best_meter":ranked[0].0,"best_phase":ranked[0].1,
                               "score_margin_not_confidence":ranked[0].2-ranked[1].2}),
        );
    }
    for flat in [-8., 0., 8.] {
        ensure!(
            rank(&beat, &vec![flat; FRAMES])?.is_empty(),
            "flat head must not identify meter"
        );
    }
    // Deliberately retain a weak-evidence diagnostic, without adjusting logits
    // or fitting a dropout prior to make this scoring-only checkpoint pass it.
    let weak: Vec<f32> = downbeat
        .iter()
        .map(|&v| if v > 0. { -2. } else { v })
        .collect();
    let weak_ranked = rank(&beat, &weak)?;
    let weak_evidence = Evidence {
        beat: &beat,
        downbeat: &weak,
        available: None,
    };
    let weak_correct = score(&weak_evidence, &correct)?.log_score;
    let weak_omitted = score(&weak_evidence, &hypothesis(PERIOD, 8, 0))?.log_score;
    let no_bars: Vec<State> = correct
        .iter()
        .map(|s| State {
            beat: s.beat,
            downbeat: false,
        })
        .collect();
    let weak_no_bars = score(&weak_evidence, &no_bars)?.log_score;
    Ok(
        json!({"schema_version":1,"purpose":"authored_complete_frame_score_correctness_gate",
        "production_output_changed":false,"training_run":false,"holdout_opened":false,
        "real_music_evaluated":false,"clock_decoder_implemented":false,
        "frame_rate_hz":50,"frames":FRAMES,"given_period_frames":PERIOD,
        "given_origin_frame":ORIGIN,"authored_support_radius_frames":RADIUS,
        "ideal_meter_phase_cases":rows,"contradiction_cases":wrong,"corruption_cases":corruptions,
        "flat_head_abstention_cases":3,
        "weak_repeated_bar_diagnostic":{"authored_meter":4,"authored_phase":0,
            "best_meter":weak_ranked[0].0,"best_phase":weak_ranked[0].1,
            "correct_unique_top_among_meters_2_through_7":weak_ranked[0].0==4 && weak_ranked[0].1==0 && weak_ranked[0].2>weak_ranked[1].2,
            "correct_log_score":weak_correct,"omitted_alternate_bars_log_score":weak_omitted,
            "no_bars_log_score":weak_no_bars,
            "correct_beats_omission_hypotheses":weak_correct>weak_omitted && weak_correct>weak_no_bars},
        "scorer_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/frame_likelihood.rs"))),
        "audit_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("frame_likelihood.rs")))}),
    )
}

fn main() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&audit()?)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn authored_structural_gate() {
        super::audit().unwrap();
    }

    #[test]
    fn weak_negative_peaks_expose_absolute_evidence_limitation() {
        let report = super::audit().unwrap();
        let weak = &report["weak_repeated_bar_diagnostic"];
        assert_eq!(weak["correct_unique_top_among_meters_2_through_7"], true);
        assert_eq!(weak["correct_beats_omission_hypotheses"], false);
        assert!(
            weak["no_bars_log_score"].as_f64().unwrap()
                > weak["correct_log_score"].as_f64().unwrap()
        );
    }
}
