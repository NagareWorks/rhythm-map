//! Fixed-path contextual scoring gate. No neural capture, training or decoding.
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
#[path = "support/phase_likelihood.rs"]
mod likelihood;
use likelihood::{Cell, score};

const SECTION: usize = 384;
const FRAMES: usize = SECTION * 3;

fn cells(periods: [usize; 3]) -> Vec<Cell> {
    periods
        .into_iter()
        .enumerate()
        .flat_map(|(part, period)| {
            (part * SECTION..(part + 1) * SECTION)
                .step_by(period)
                .map(move |start| Cell {
                    start,
                    end: start + period,
                    phase: 4,
                })
        })
        .collect()
}

fn heads(cells: &[Cell], n: usize) -> Vec<f32> {
    let mut values = vec![-8.; n];
    for c in cells {
        let center = c.start + c.phase;
        values[center - 1..=center + 1].fill(8.);
    }
    values
}

/// Fixed-period diagnostic: integrate ONE unknown phase shared by all cells,
/// with a uniform phase prior. Contrast with the invalid per-cell best search.
#[allow(clippy::cast_precision_loss)]
fn coherence(values: &[f32], period: usize) -> Result<serde_json::Value> {
    ensure!(
        values.len().is_multiple_of(period),
        "incomplete coherence cells"
    );
    let mut shared = vec![0.0; period];
    let mut independently_best = 0.0;
    for window in values.chunks_exact(period) {
        let phases: Vec<f64> = (0..period)
            .map(|p| likelihood::cell(window, p).map(|e| e.log_ratio_to_null))
            .collect::<Result<_>>()?;
        independently_best += phases.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for (sum, value) in shared.iter_mut().zip(phases) {
            *sum += value;
        }
    }
    let maximum = shared.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let marginal =
        maximum + (shared.iter().map(|s| (s - maximum).exp()).sum::<f64>() / period as f64).ln();
    Ok(
        json!({"given_period_frames":period,"cells":values.len()/period,
        "per_cell_max_score_not_valid_evidence":independently_best,
        "shared_phase_marginal_log_ratio_to_null":marginal}),
    )
}

fn coherence_controls() -> Result<Vec<serde_json::Value>> {
    let mut seed = 0x1357_2468_u32;
    let noise: Vec<f32> = (0..FRAMES)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            -8.0 + f32::from(u8::try_from(seed >> 24).unwrap()) / 255.0 * 6.0
        })
        .collect();
    let mut stable = vec![-8.0; FRAMES];
    let mut drifting = stable.clone();
    for (index, (a, b)) in stable
        .as_chunks_mut::<24>()
        .0
        .iter_mut()
        .zip(drifting.as_chunks_mut::<24>().0.iter_mut())
        .enumerate()
    {
        for offset in [23, 0, 1] {
            a[(4 + offset) % 24] = -2.0;
            b[(4 + index + offset) % 24] = -2.0;
        }
    }
    [
        ("fixed_seed_nonperiodic_noise", noise),
        ("coherent_weak_pulses", stable),
        ("phase_drift_against_fixed_period", drifting),
    ]
    .into_iter()
    .map(|(name, values)| Ok(json!({"case":name,"measurement":coherence(&values,24)?})))
    .collect()
}

fn audit() -> Result<serde_json::Value> {
    let candidates = [
        ("constant_125", cells([24, 24, 24])),
        ("true_half_speed", cells([24, 48, 24])),
        ("true_double_speed", cells([24, 12, 24])),
        ("true_non_octave", cells([24, 32, 24])),
    ];
    let mut rows = Vec::new();
    for (id, target, weak, erase, all_weak) in [
        ("constant_intact", 0, false, false, false),
        ("constant_weak_alternating", 0, true, false, false),
        ("constant_erased_alternating", 0, false, true, false),
        ("half_speed_intact", 1, false, false, false),
        ("double_speed_intact", 2, false, false, false),
        ("double_speed_weak_alternating", 2, true, false, false),
        ("non_octave_intact", 3, false, false, false),
        ("constant_all_weak", 0, false, false, true),
    ] {
        let mut values = heads(&candidates[target].1, FRAMES);
        if all_weak {
            for v in &mut values {
                if *v > 0. {
                    *v = -2.;
                }
            }
        }
        if weak || erase {
            let period = if target == 2 { 12 } else { 24 };
            for center in (SECTION + 4 + period..SECTION * 2).step_by(period * 2) {
                values[center - 1..=center + 1].fill(if erase { -8. } else { -2. });
            }
        }
        let mut scores = Vec::new();
        for (name, path) in &candidates {
            scores.push((*name, score(&values, path, None)?));
        }
        let maximum = scores
            .iter()
            .map(|(_, s)| s.log_ratio_to_null)
            .fold(0.0, f64::max);
        // Numerical equality tolerance only, not a tuned evidence threshold.
        let mut winners: Vec<&str> = scores
            .iter()
            .filter(|(_, s)| (s.log_ratio_to_null - maximum).abs() < 1e-9)
            .map(|(name, _)| *name)
            .collect();
        if maximum.abs() < 1e-9 {
            winners.push("null_unsupported");
        }
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| f64::from(v).to_le_bytes())
            .collect();
        rows.push(json!({"case":id,"authored_path":candidates[target].0,"best_given_paths":winners,
            "authored_path_unique_top":winners.len()==1&&winners[0]==candidates[target].0,
            "input_f64_le_sha256":format!("{:x}",Sha256::digest(bytes)),
            "null_log_ratio":0.0,"given_path_scores":scores.iter().map(|(name,s)|json!({"path":name,"score":s})).collect::<Vec<_>>()}));
    }
    ensure!(
        rows[2]["input_f64_le_sha256"] == rows[3]["input_f64_le_sha256"],
        "identical-input witness changed"
    );
    ensure!(
        rows[2]["given_path_scores"] == rows[3]["given_path_scores"],
        "identical-input scores changed"
    );
    let mut meters = Vec::new();
    for weak in [false, true] {
        let correct = cells([96, 96, 96]);
        let mut values = heads(&correct, FRAMES);
        if weak {
            for v in &mut values {
                if *v > 0. {
                    *v = -2.;
                }
            }
        }
        let mut scores = Vec::new();
        for meter in [2, 4, 8] {
            scores.push((meter, score(&values, &cells([meter * 24; 3]), None)?));
        }
        meters.push(json!({"weak":weak,"authored_meter":4,"given_meter_scores":scores.iter().map(|(m,s)|json!({"meter":m,"score":s})).collect::<Vec<_>>()}));
    }
    let flat = score(&[-8.; FRAMES], &candidates[0].1, None)?;
    Ok(
        json!({"schema_version":1,"purpose":"rotation_normalized_contextual_phase_gate",
        "production_output_changed":false,"training_run":false,"holdout_opened":false,
        "real_music_evaluated":false,"clock_decoder_implemented":false,
        "frames":FRAMES,"frame_rate_hz":50,"shape_weights":[0.25,0.5,0.25],
        "cases":rows,"meter_controls":meters,"flat_control":flat,"coherence_controls":coherence_controls()?,
        "scorer_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/phase_likelihood.rs"))),
        "audit_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("phase_likelihood.rs")))}),
    )
}

fn main() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&audit()?)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn authored_gate_preserves_the_erased_input_ambiguity() {
        let r = super::audit().unwrap();
        assert_eq!(r["cases"].as_array().unwrap().len(), 8);
        assert_eq!(r["flat_control"]["neutral_flat_cells"], 48);
        assert_eq!(r["flat_control"]["log_ratio_to_null"], 0.0);
    }

    #[test]
    fn weak_changes_and_meter_density_are_not_hidden_by_local_success() {
        let r = super::audit().unwrap();
        for (i, row) in r["cases"].as_array().unwrap().iter().enumerate() {
            assert_eq!(row["authored_path_unique_top"], i != 2);
        }
        for control in r["meter_controls"].as_array().unwrap() {
            let scores = control["given_meter_scores"].as_array().unwrap();
            let correct = scores[1]["score"]["log_ratio_to_null"].as_f64().unwrap();
            assert!(correct > scores[0]["score"]["log_ratio_to_null"].as_f64().unwrap());
            assert!(correct > scores[2]["score"]["log_ratio_to_null"].as_f64().unwrap());
        }
    }

    #[test]
    fn independent_window_search_does_not_count_as_coherent_evidence() {
        let rows = super::coherence_controls().unwrap();
        for (i, row) in rows.iter().enumerate() {
            let m = &row["measurement"];
            assert!(m["per_cell_max_score_not_valid_evidence"].as_f64().unwrap() > 0.0);
            assert_eq!(
                m["shared_phase_marginal_log_ratio_to_null"]
                    .as_f64()
                    .unwrap()
                    > 0.0,
                i == 1
            );
        }
    }
}
