//! Omitted pulse/accent states and automatic emission-equivalence diagnostics.
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
#[path = "support/rank_fixtures.rs"]
mod fixtures;
#[path = "support/shared_frames.rs"]
mod frames;
#[path = "support/omission.rs"]
mod omission;
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

fn score(beat: &[f32], bar: &[f32], available: &[bool], clocks: &[Template]) -> Result<Value> {
    let features = ranks::transform(&ranks::remove_background(
        &frames::features(beat, bar, available, 10, false)?,
        10,
    )?)?;
    let n = available.iter().filter(|&&a| a).count();
    let table = frames::Table::new(&features, n.min(64), n.min(32))?;
    let prior_mass = clocks
        .iter()
        .map(|c| c.duration_prior_log_weight)
        .fold(f64::NEG_INFINITY, frames::add);
    let paths: Vec<Vec<_>> = clocks
        .iter()
        .map(|c| c.given_ticks.iter().map(|p| p.0).collect())
        .collect();
    let mut results = Vec::new();
    let mut weights = Vec::new();
    let mut intact = Vec::new();
    let mut selected = 0;
    let mut map_weight = f64::NEG_INFINITY;
    for (i, (clock, path)) in clocks.iter().zip(&paths).enumerate() {
        let result = omission::infer(&table, path, 2, 7)?;
        let prior = clock.duration_prior_log_weight - prior_mass;
        weights.push(result.log_ratio + prior);
        intact.push(omission::intact_reference(&table, path, 2, 7)? + prior);
        if result.joint_map.log_weight + prior > map_weight {
            selected = i;
            map_weight = result.joint_map.log_weight + prior;
        }
        results.push(result);
    }
    let total = weights.iter().copied().fold(f64::NEG_INFINITY, frames::add);
    let intact_total = intact.iter().copied().fold(f64::NEG_INFINITY, frames::add);
    let map = &results[selected].joint_map;
    let labels: Vec<_> = paths[selected]
        .iter()
        .zip(&map.inferred_labels)
        .filter_map(|(&t, l)| l.filter(|&l| l > 0).map(|l| (t, l)))
        .collect();
    let mut class_priors = Vec::new();
    for (clock, path) in clocks.iter().zip(&paths) {
        class_priors.push(
            omission::assignment_prior(&table, path, &labels, 2, 7)?
                .map(|p| p + clock.duration_prior_log_weight - prior_mass),
        );
    }
    let class_total = class_priors
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, frames::add);
    let bars = labels.iter().filter(|p| p.1 == 2).count();
    let feature_score = labels
        .iter()
        .map(|&(t, l)| {
            let p = table.centered[t].unwrap();
            p[0] + if l == 2 { p[1] } else { 0. }
        })
        .sum::<f64>()
        - table.normalizer(labels.len() - bars, bars)?;
    ensure!(
        (feature_score - map.feature_log_ratio).abs() < 1e-8,
        "MAP assignment score drift"
    );
    Ok(json!({"available_frames":n,"clock_family_log_ratio":total,
        "clock_family_probabilities":weights.iter().map(|w| (w-total).exp()).collect::<Vec<_>>(),
        "matched_no_omission_log_weights":intact,
        "matched_no_omission_probabilities":intact.iter().map(|w| (w-intact_total).exp()).collect::<Vec<_>>(),
        "clocks":results,
        "selected_joint_map_clock":clocks[selected].clock,
        "selected_assignment_equivalence":{
            "inferred_emitted_labels":labels,"labels_are_detected_events":false,
            "shared_feature_log_ratio":feature_score,
            "multiple_compatible_latent_clocks":class_priors.iter().flatten().count()>1,
            "clock_log_priors":class_priors,
            "conditional_clock_probabilities":class_priors.iter().map(|p| p.map_or(0.,|p| (p-class_total).exp())).collect::<Vec<_>>(),
            "assignment_probability_in_full_model":(feature_score+class_total-total).exp(),
            "not_whole_posterior_ambiguity":true}}))
}

fn audit() -> Result<Value> {
    let frozen_bytes = include_bytes!("../../../evaluation/parity/rank-clock-v1.json");
    let frozen: Value = serde_json::from_slice(frozen_bytes)?;
    let clocks: Vec<Template> = serde_json::from_value(frozen["clock_templates"].clone())?;
    let mut rows = Vec::new();
    for f in fixtures::all() {
        let old = frozen["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["case"] == f.name)
            .unwrap();
        let b = digest(&f.beat);
        let d = digest(&f.bar);
        let a = hash(
            &f.available
                .iter()
                .copied()
                .map(u8::from)
                .collect::<Vec<_>>(),
        );
        ensure!(
            old["beat_f64_le_sha256"] == b
                && old["bar_f64_le_sha256"] == d
                && old["availability_u8_sha256"] == a,
            "changed frozen observations"
        );
        eprintln!("omission supplied-clock gate: {}", f.name);
        rows.push(json!({"case":f.name,"authored_clock":f.authored_clock,
            "beat_f64_le_sha256":b,"bar_f64_le_sha256":d,"availability_u8_sha256":a,
            "decoded":score(&f.beat,&f.bar,&f.available,&clocks)?}));
    }
    Ok(
        json!({"schema_version":1,"purpose":"constant_meter_supplied_clock_omission_gate",
        "production_output_changed":false,"training_run":false,"holdout_opened":false,
        "real_music_evaluated":false,"unrestricted_clock_search":false,
        "supplied_clock_templates":true,"truth_assisted_clock_family":true,
        "static_meter_marginalized":true,"meter_changes_searched":false,
        "pulse_and_accent_omissions_marginalized":true,"calibrated_confidence":false,
        "retention_priors":"independent run-wide Beta(1,1); accent conditional on retained latent bar ticks",
        "frames":fixtures::N,"frame_rate_hz":50,"clock_templates":frozen["clock_templates"],"cases":rows,
        "audit_source_sha256":hash(include_bytes!("omission_clock.rs")),
        "omission_source_sha256":hash(include_bytes!("support/omission.rs")),
        "rank_source_sha256":hash(include_bytes!("support/rank_frames.rs")),
        "feature_source_sha256":hash(include_bytes!("support/shared_frames.rs")),
        "fixture_source_sha256":hash(include_bytes!("support/rank_fixtures.rs")),
        "rank_report_sha256":hash(frozen_bytes)}),
    )
}

fn main() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&audit()?)?);
    Ok(())
}
