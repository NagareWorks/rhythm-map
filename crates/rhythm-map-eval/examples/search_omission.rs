//! Small exact unknown-clock search, not production output or musical accuracy.
#[path = "support/shared_frames.rs"]
#[allow(dead_code)]
mod frames;
#[path = "support/search_omission.rs"]
mod search;
use anyhow::Result;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn cases() -> Vec<(&'static str, Vec<Option<[f64; 2]>>)> {
    let signal = |ticks: &[usize], bars: &[usize]| {
        (0..18)
            .map(|t| {
                Some([
                    if ticks.contains(&t) { 4. } else { 0. },
                    if bars.contains(&t) { 3. } else { 0. },
                ])
            })
            .collect::<Vec<_>>()
    };
    let constant = signal(&[1, 4, 7, 10, 13, 16], &[1, 10]);
    let half = signal(&[1, 4, 7, 13], &[1, 13]);
    let mut flat_span = constant.clone();
    flat_span[6..13].fill(Some([0., 0.]));
    let mut missing_span = constant.clone();
    missing_span[6..13].fill(None);
    vec![
        ("constant", constant),
        ("half", half.clone()),
        ("same_features_erased_constant", half),
        ("double", signal(&[1, 7, 10, 13, 16], &[1, 13])),
        ("meter_change", signal(&[1, 4, 7, 10, 13, 16], &[1, 7, 16])),
        ("phase_shift", signal(&[2, 5, 8, 11, 14, 17], &[2, 11])),
        ("flat_middle", flat_span),
        ("unavailable_middle", missing_span),
        ("flat", vec![Some([0., 0.]); 18]),
        ("unavailable", vec![None; 18]),
    ]
}

#[allow(dead_code)]
fn audit() -> Result<Value> {
    let domain = search::Domain {
        min_period: 3,
        max_period: 6,
        min_meter: 2,
        max_meter: 3,
        max_states: 250_000,
    };
    let mut rows = Vec::new();
    for (name, values) in cases() {
        let observed = values.iter().flatten().count();
        let table = frames::Table::new(&values, observed, observed)?;
        let result = search::infer(&table, domain)?;
        eprintln!(
            "bounded unknown-clock gate: {name}: {} states, {} transitions",
            result.states, result.transitions
        );
        rows.push(json!({"case":name,"feature_pairs":values,"decoded":result}));
    }
    let small = vec![
        Some([0.2, 1.]),
        None,
        Some([1.4, -0.3]),
        Some([0.5, 1.5]),
        Some([-1., 0.7]),
        Some([0.1, -0.2]),
        Some([0.4, -0.7]),
        Some([1., 0.9]),
    ];
    let small_domain = search::Domain {
        min_period: 2,
        max_period: 3,
        min_meter: 2,
        max_meter: 3,
        max_states: 100_000,
    };
    let observed = small.iter().flatten().count();
    let exhaustive = search::infer(
        &frames::Table::new(&small, observed, observed)?,
        small_domain,
    )?;
    let resource_features = vec![Some([0., 0.]); 32];
    let resource_table = frames::Table::new(&resource_features, 32, 32)?;
    let resource_probe = match search::infer(&resource_table, domain) {
        Ok(result) => {
            json!({"completed":true,"states":result.states,"transitions":result.transitions})
        }
        Err(error) => {
            json!({"completed":false,"error":error.to_string(),"partial_inference_returned":false})
        }
    };
    Ok(
        json!({"schema_version":1,"purpose":"bounded_unknown_clock_omission_semantics",
        "production_output_changed":false,"user_parameters_added":false,"training_run":false,"holdout_opened":false,
        "real_music_evaluated":false,"supplied_clock_templates":false,"truth_used_in_search":false,
        "tempo_changes_searched":true,"meter_changes_searched":true,
        "feature_space_inputs":true,"rank_pipeline_accuracy_evaluated":false,"calibrated_confidence":false,
        "labels_are_detected_events":false,"full_song_search":false,"beam_pruning":false,
        "omission_and_meter_rates":"independent run-wide Beta(1,1)",
        "terminal_semantics":"next tick outside recording: marginalize future, no extra transition",
        "domain":domain,"cases":rows,"exhaustive_control":{"domain":small_domain,"feature_pairs":small,"decoded":exhaustive},
        "resource_probe":{"frames":32,"domain":domain,"result":resource_probe},
        "source_sha256":hash(include_bytes!("search_omission.rs")),
        "search_source_sha256":hash(include_bytes!("support/search_omission.rs")),
        "feature_source_sha256":hash(include_bytes!("support/shared_frames.rs")),
        "time_prior_source_sha256":hash(include_bytes!("support/time_prior.rs"))}),
    )
}

fn main() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&audit()?)?);
    Ok(())
}
