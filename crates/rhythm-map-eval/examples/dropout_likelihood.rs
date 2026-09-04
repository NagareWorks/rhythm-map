//! Authored dropout observation audit, not a searched or promoted clock decoder.
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
#[path = "support/dropout_likelihood.rs"]
mod likelihood;

const SECTION: usize = 384;
const FRAMES: usize = SECTION * 3;
const RATE: f64 = 0.1;

fn mask(periods: [usize; 3]) -> Vec<bool> {
    let mut output = vec![false; FRAMES];
    for (part, period) in periods.into_iter().enumerate() {
        for frame in (part * SECTION + 4..(part + 1) * SECTION).step_by(period) {
            output[frame - 1..=frame + 1].fill(true);
        }
    }
    output
}

fn audit() -> Result<serde_json::Value> {
    let candidates = [
        ("constant_125", mask([24, 24, 24])),
        ("true_half_speed", mask([24, 48, 24])),
        ("true_double_speed", mask([24, 12, 24])),
        ("true_non_octave", mask([24, 32, 24])),
        ("all_absent", vec![false; FRAMES]),
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
        let mut logits: Vec<f64> = candidates[target]
            .1
            .iter()
            .map(|&on| if on { 8.0 } else { -8.0 })
            .collect();
        if all_weak {
            for value in &mut logits {
                if *value > 0.0 {
                    *value = -2.0;
                }
            }
        }
        if weak || erase {
            let period = if target == 2 { 12 } else { 24 };
            for frame in (SECTION + 4 + period..2 * SECTION).step_by(2 * period) {
                logits[frame - 1..=frame + 1].fill(if erase { -8.0 } else { -2.0 });
            }
        }
        let mut scored = Vec::new();
        for (name, states) in &candidates {
            scored.push((*name, likelihood::score(&logits, states, None, RATE)?));
        }
        let best = scored
            .iter()
            .max_by(|a, b| {
                a.1.log_ratio_to_all_absent
                    .total_cmp(&b.1.log_ratio_to_all_absent)
            })
            .unwrap();
        let bytes: Vec<u8> = logits.iter().flat_map(|z| z.to_le_bytes()).collect();
        rows.push(json!({"case":id,"authored_path":candidates[target].0,"best_given_path":best.0,
            "authored_path_wins":best.0==candidates[target].0,
            "input_f64_le_sha256":format!("{:x}",Sha256::digest(bytes)),
            "given_path_scores":scored.iter().map(|(name, value)| json!({"path":name,"score":value})).collect::<Vec<_>>()}));
    }
    ensure!(
        rows[2]["input_f64_le_sha256"] == rows[3]["input_f64_le_sha256"],
        "equivalence witness changed"
    );
    ensure!(
        rows[2]["given_path_scores"] == rows[3]["given_path_scores"],
        "identical input scored differently"
    );
    Ok(
        json!({"schema_version":1,"purpose":"normalized_dropout_observation_limit",
        "production_output_changed":false,"training_run":false,"holdout_opened":false,
        "real_music_evaluated":false,"clock_decoder_implemented":false,
        "fixed_missing_rate":RATE,"frames":FRAMES,"frame_rate_hz":50,
        "log_density_measure":"real_logit_axis","single_head_only":true,
        "cases":rows,"scorer_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/dropout_likelihood.rs"))),
        "audit_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("dropout_likelihood.rs")))}),
    )
}

fn main() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&audit()?)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn fixed_path_controls_expose_the_weak_event_limit() {
        let report = super::audit().unwrap();
        let rows = report["cases"].as_array().unwrap();
        for i in [0, 3, 4, 6] {
            assert_eq!(rows[i]["authored_path_wins"], true);
        }
        for i in [1, 2, 5, 7] {
            assert_eq!(rows[i]["authored_path_wins"], false);
        }
        assert_eq!(rows[5]["best_given_path"], "constant_125");
        assert_eq!(rows[7]["best_given_path"], "all_absent");
    }
}
