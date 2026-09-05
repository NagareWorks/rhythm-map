//! Bounded order-evidence intervention on the frozen supplied clock family.
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
#[path = "support/rank_fixtures.rs"]
mod fixtures;
#[path = "support/shared_frames.rs"]
mod frames;
#[path = "support/frame_meter.rs"]
mod meter;
#[path = "support/rank_frames.rs"]
mod ranks;

#[derive(Deserialize)]
struct Template {
    clock: String,
    given_ticks: Vec<(usize, usize)>,
    duration_prior_log_weight: f64,
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn digest(values: &[f32]) -> String {
    hash(
        &values
            .iter()
            .flat_map(|&x| f64::from(x).to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn score(
    beat: &[f32],
    bar: &[f32],
    available: &[bool],
    clocks: &[Template],
    background: bool,
) -> Result<Value> {
    let smooth = frames::features(beat, bar, available, 10, false)?;
    let features = ranks::transform(&if background {
        ranks::remove_background(&smooth, 10)?
    } else {
        smooth
    })?;
    let n = available.iter().filter(|&&a| a).count();
    let table = frames::Table::new(&features, n.min(64), n.min(32))?;
    let prior_mass = clocks
        .iter()
        .map(|c| c.duration_prior_log_weight)
        .fold(f64::NEG_INFINITY, frames::add);
    let mut rows = Vec::new();
    let mut weights = Vec::new();
    for clock in clocks {
        let path = &clock.given_ticks;
        let b = path.iter().filter(|(t, _)| available[*t]).count();
        let beat_sum: f64 = path
            .iter()
            .filter_map(|(t, _)| table.centered[*t].map(|p| p[0]))
            .sum();
        let marks: Vec<_> = path
            .iter()
            .map(|(t, _)| table.centered[*t].map(|p| p[1]))
            .collect();
        let norm: Vec<_> = (0..=b.min(path.len().div_ceil(2)))
            .map(|d| table.normalizer(b - d, d))
            .collect::<Result<_>>()?;
        let result = meter::infer(&marks, &norm, 2, 7)?;
        let evidence = beat_sum + result.log_ratio_to_reference;
        let weight = evidence + clock.duration_prior_log_weight - prior_mass;
        weights.push(weight);
        rows.push(json!({"clock":clock.clock,"visible_beats":b,"unobserved_ticks":path.len()-b,
            "beat_score_sum":beat_sum,"bar_mark_scores":marks,"paired_log_normalizers":norm,
            "meter_count_probabilities":result.count_probabilities,
            "meter_log_ratio":result.log_ratio_to_reference,
            "mean_meter_change_probability":result.mean_change_probability_per_bar,
            "downbeat_probabilities":result.positions.iter().map(|p| p.downbeat_probability).collect::<Vec<_>>(),
            "joint_log_ratio":evidence,"family_log_weight":weight}));
    }
    let total = weights.iter().copied().fold(f64::NEG_INFINITY, frames::add);
    Ok(
        json!({"available_frames":table.available_frames,"clocks":rows,
        "clock_family_log_ratio":total,
        "clock_family_probabilities":weights.iter().map(|w| (w-total).exp()).collect::<Vec<_>>() }),
    )
}

fn audit() -> Result<Value> {
    let frozen_bytes = include_bytes!("../../../evaluation/parity/shared-clock-v1.json");
    let frozen: Value = serde_json::from_slice(frozen_bytes)?;
    let family = &frozen["cases"][0]["raw"]["clocks"];
    let clocks: Vec<Template> = serde_json::from_value(family.clone())?;
    let mut rows = Vec::new();
    for f in fixtures::all() {
        let beat_hash = digest(&f.beat);
        let bar_hash = digest(&f.bar);
        if let Some(old) = frozen["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["case"] == f.name)
        {
            ensure!(
                old["beat_f64_le_sha256"] == beat_hash && old["bar_f64_le_sha256"] == bar_hash,
                "changed frozen head input"
            );
        }
        eprintln!("rank supplied-clock gate: {}", f.name);
        let raw = score(&f.beat, &f.bar, &f.available, &clocks, false)?;
        let out = score(&f.beat, &f.bar, &f.available, &clocks, true)?;
        rows.push(json!({"case":f.name,"authored_clock":f.authored_clock,
            "beat_f64_le_sha256":beat_hash,"bar_f64_le_sha256":bar_hash,
            "availability_u8_sha256":hash(&f.available.iter().copied().map(u8::from).collect::<Vec<_>>()),
            "raw_rank":raw,"ranked":out}));
    }
    // This is a post-scoring identifiability witness, NOT a label-aware selector.
    let half = rows
        .iter()
        .find(|r| r["case"] == "half_speed_intact")
        .unwrap();
    let erased = rows
        .iter()
        .find(|r| r["case"] == "constant_erased_beats_and_bars")
        .unwrap();
    ensure!(
        half["ranked"] == erased["ranked"],
        "equal observations changed inference"
    );
    let templates: Vec<_> = clocks
        .iter()
        .map(|c| {
            json!({"clock":c.clock,
        "given_ticks":c.given_ticks,"duration_prior_log_weight":c.duration_prior_log_weight})
        })
        .collect();
    Ok(
        json!({"schema_version":1,"purpose":"shared_frame_rank_given_clock_gate",
        "production_output_changed":false,"training_run":false,"holdout_opened":false,
        "real_music_evaluated":false,"unrestricted_clock_search":false,
        "supplied_clock_templates":true,"truth_assisted_clock_family":true,"meter_paths_searched":true,
        "dropout_states_marginalized":false,"calibrated_confidence":false,
        "background_window_frames":9,"minimum_period_frames":10,
        "frames":fixtures::N,"frame_rate_hz":50,"clock_templates":templates,
        "ambiguous_observation_witness":{"cases":["half_speed_intact","constant_erased_beats_and_bars"],
            "diagnostic_only":true,"latent_clock_identifiable_from_these_heads":false},
        "cases":rows,
        "audit_source_sha256":hash(include_bytes!("rank_clock.rs")),
        "fixture_source_sha256":hash(include_bytes!("support/rank_fixtures.rs")),
        "rank_source_sha256":hash(include_bytes!("support/rank_frames.rs")),
        "feature_source_sha256":hash(include_bytes!("support/shared_frames.rs")),
        "meter_source_sha256":hash(include_bytes!("support/frame_meter.rs")),
        "shared_report_sha256":hash(frozen_bytes)}),
    )
}

fn main() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&audit()?)?);
    Ok(())
}
