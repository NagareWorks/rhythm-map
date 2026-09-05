//! Shared-frame gate over supplied clock templates; not unrestricted decoding.
use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
#[path = "support/shared_frames.rs"]
mod frames;
#[path = "support/frame_meter.rs"]
mod meter;
#[path = "support/time_prior.rs"]
mod prior;

const SECTION: usize = 384;
const N: usize = 1152;
const CLOCKS: [(&str, [usize; 3]); 4] = [
    ("constant", [24; 3]),
    ("half", [24, 48, 24]),
    ("double", [24, 12, 24]),
    ("non_octave", [24, 32, 24]),
];

fn ticks(periods: [usize; 3]) -> Vec<(usize, usize)> {
    periods
        .into_iter()
        .enumerate()
        .flat_map(|(part, p)| {
            (part * SECTION + 4..(part + 1) * SECTION)
                .step_by(p)
                .map(move |t| (t, p))
        })
        .collect()
}

fn heads(periods: [usize; 3], weak: bool) -> (Vec<f32>, Vec<f32>) {
    let mut beat = vec![-8.; N];
    let mut bar = beat.clone();
    for (part, p) in periods.into_iter().enumerate() {
        for (i, t) in (part * SECTION + 4..(part + 1) * SECTION)
            .step_by(p)
            .enumerate()
        {
            beat[t - 1..=t + 1].fill(if weak { -2. } else { 8. });
            if i.is_multiple_of(4) {
                bar[t - 1..=t + 1].fill(if weak { -2. } else { 8. });
            }
        }
    }
    (beat, bar)
}

fn digest(values: &[f32]) -> String {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|&x| f64::from(x).to_le_bytes())
        .collect();
    format!("{:x}", Sha256::digest(bytes))
}

#[allow(clippy::cast_precision_loss)]
fn duration_weight(path: &[(usize, usize)]) -> f64 {
    let p = prior::Prior::new(&(10..=75).collect::<Vec<_>>());
    let mut weight = -66_f64.ln();
    for pair in path.windows(2) {
        let from = pair[0].1 - 10;
        let to = pair[1].1 - 10;
        weight += if from == to {
            p.log_survival[from]
        } else {
            p.log_jump_base[from] - (p.coordinates[from] - p.coordinates[to]).abs()
        };
    }
    // The last duration is censored at the SAME recording endpoint for every clock.
    weight - p.rate_per_frame * (N - path.last().unwrap().0) as f64
}

fn compare(
    beat: &[f32],
    bar: &[f32],
    available: &[bool],
    contextual: bool,
) -> Result<serde_json::Value> {
    let features = frames::features(beat, bar, available, 10, contextual)?;
    let observed = available.iter().filter(|&&a| a).count();
    let table = frames::Table::new(&features, 64.min(observed), 32.min(observed))?;
    let paths: Vec<_> = CLOCKS.iter().map(|(_, p)| ticks(*p)).collect();
    let priors: Vec<_> = paths.iter().map(|p| duration_weight(p)).collect();
    let prior_mass = priors.iter().copied().fold(f64::NEG_INFINITY, frames::add);
    let mut rows = Vec::new();
    let mut weights = Vec::new();
    for ((name, _), (path, duration)) in CLOCKS.iter().zip(paths.iter().zip(&priors)) {
        let b = path.iter().filter(|(t, _)| available[*t]).count();
        let beat_sum = path
            .iter()
            .filter_map(|(t, _)| table.centered[*t].map(|p| p[0]))
            .sum::<f64>();
        let marks: Vec<_> = path
            .iter()
            .map(|(t, _)| table.centered[*t].map(|p| p[1]))
            .collect();
        let norm: Vec<_> = (0..=b.min(path.len().div_ceil(2)))
            .map(|d| table.normalizer(b - d, d))
            .collect::<Result<_>>()?;
        let inference = meter::infer(&marks, &norm, 2, 7)?;
        let evidence = inference.log_ratio_to_reference + beat_sum;
        let weight = evidence + duration - prior_mass;
        weights.push(weight);
        rows.push(json!({"clock":name,"given_ticks":path,"visible_beats":b,
            "unobserved_ticks":path.len()-b,"beat_score_sum":beat_sum,
            "bar_mark_scores":marks,"paired_log_normalizers":norm,
            "meter":inference,"joint_log_ratio":evidence,
            "duration_prior_log_weight":duration,"family_log_weight":weight}));
    }
    let total = weights.iter().copied().fold(f64::NEG_INFINITY, frames::add);
    let posterior: Vec<_> = weights.iter().map(|w| (w - total).exp()).collect();
    Ok(
        json!({"available_frames":table.available_frames,"unavailable_frames":N-table.available_frames,
        "feature_mode":if contextual {"fixed_local_context"} else {"raw_smoothed_diagnostic"},
        "clock_family_log_ratio":total,"clock_family_probabilities":posterior,
        "normalized_duration_prior":priors.iter().map(|p| p-prior_mass).collect::<Vec<_>>(),"clocks":rows}),
    )
}

fn audit() -> Result<serde_json::Value> {
    let frozen: serde_json::Value = serde_json::from_str(include_str!(
        "../../../evaluation/parity/time-clock-v1.json"
    ))?;
    let mut rows = Vec::new();
    for (name, target, weak, alternating) in [
        ("constant_intact", 0, false, false),
        ("constant_weak_alternating", 0, false, true),
        ("half_speed_intact", 1, false, false),
        ("double_speed_intact", 2, false, false),
        ("double_speed_weak_alternating", 2, false, true),
        ("non_octave_intact", 3, false, false),
        ("constant_all_weak", 0, true, false),
        ("constant_erased_beats", 0, false, false),
        ("constant_erased_beats_and_bars", 0, false, false),
        ("flat", 0, false, false),
        ("fixed_seed_noise", 0, false, false),
        ("flat_middle", 0, false, false),
        ("unavailable_gap", 0, false, false),
        ("all_unavailable", 0, false, false),
    ] {
        let (mut beat, mut bar) = heads(CLOCKS[target].1, weak);
        let mut available = vec![true; N];
        if alternating {
            let p = CLOCKS[target].1[1];
            for t in (SECTION + 4 + p..SECTION * 2).step_by(p * 2) {
                beat[t - 1..=t + 1].fill(-2.);
            }
        }
        if name.starts_with("constant_erased") {
            for t in (SECTION + 28..SECTION * 2).step_by(48) {
                beat[t - 1..=t + 1].fill(-8.);
            }
            if name == "constant_erased_beats_and_bars" {
                for t in (SECTION + 100..SECTION * 2).step_by(192) {
                    bar[t - 1..=t + 1].fill(-8.);
                }
            }
        }
        match name {
            "flat" => {
                beat.fill(-8.);
                bar.fill(-8.);
            }
            "flat_middle" => {
                beat[480..672].fill(-8.);
                bar[480..672].fill(-8.);
            }
            "unavailable_gap" => available[480..672].fill(false),
            "all_unavailable" => available.fill(false),
            "fixed_seed_noise" => {
                let mut seed = 0x1357_2468_u32;
                for values in [&mut beat, &mut bar] {
                    for value in values {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        *value = -8. + f32::from(u8::try_from(seed >> 24).unwrap()) / 255. * 6.;
                    }
                }
            }
            _ => {}
        }
        if let Some(previous) = frozen["cases"]
            .as_array()
            .unwrap()
            .iter()
            .chain(frozen["controls"].as_array().unwrap())
            .find(|r| r["case"] == name)
        {
            ensure!(
                previous["beat_f64_le_sha256"] == digest(&beat)
                    && previous["bar_f64_le_sha256"] == digest(&bar),
                "changed frozen head input"
            );
        }
        eprintln!("shared-frame supplied-clock gate: {name}");
        let raw = compare(&beat, &bar, &available, false)?;
        let contextual = compare(&beat, &bar, &available, true)?;
        rows.push(json!({"case":name,"authored_clock":CLOCKS[target].0,
            "beat_f64_le_sha256":digest(&beat),"bar_f64_le_sha256":digest(&bar),
            "raw":raw,"contextual":contextual}));
    }
    Ok(
        json!({"schema_version":1,"purpose":"shared_frame_given_clock_joint_gate",
        "production_output_changed":false,"training_run":false,"holdout_opened":false,
        "real_music_evaluated":false,"unrestricted_clock_search":false,
        "supplied_clock_templates":true,"truth_assisted_clock_family":true,
        "meter_paths_searched":true,"minimum_period_frames":10,"feature_window_frames":9,
        "frames":N,"frame_rate_hz":50,"cases":rows,
        "audit_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("shared_clock.rs"))),
        "feature_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/shared_frames.rs"))),
        "meter_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/frame_meter.rs"))),
        "prior_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/time_prior.rs"))),
        "clock_report_sha256":format!("{:x}",Sha256::digest(include_bytes!("../../../evaluation/parity/time-clock-v1.json")))}),
    )
}

fn main() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&audit()?)?);
    Ok(())
}
