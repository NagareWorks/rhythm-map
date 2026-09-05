//! Conditional meter audit on frozen inferred ticks, with censored edge bars.
use anyhow::Result;
use serde_json::json;
use sha2::{Digest, Sha256};
#[path = "support/censored_meter.rs"]
mod clock;
#[path = "support/phase_likelihood.rs"]
#[allow(dead_code)] // Reuse the frozen cell kernel; its complete-path API is not needed here.
mod reference;

const SECTION: usize = 384;
const FRAMES: usize = SECTION * 3;

fn pulses(periods: [usize; 3], weak: bool) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let mut beat = vec![-8.; FRAMES];
    let mut bar = beat.clone();
    let mut truth = Vec::new();
    for (part, period) in periods.into_iter().enumerate() {
        for (index, center) in (part * SECTION + 4..(part + 1) * SECTION)
            .step_by(period)
            .enumerate()
        {
            beat[center - 1..=center + 1].fill(if weak { -2. } else { 8. });
            if index.is_multiple_of(4) {
                bar[center - 1..=center + 1].fill(if weak { -2. } else { 8. });
            }
            truth.push(center);
        }
    }
    (beat, bar, truth)
}

fn digest(values: &[f32]) -> String {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&x| f64::from(x).to_le_bytes())
        .collect();
    format!("{:x}", Sha256::digest(bytes))
}

#[allow(clippy::too_many_lines)]
fn audit() -> Result<serde_json::Value> {
    let mut rows = Vec::new();
    for (name, periods, weak, alternating) in [
        ("constant_intact", [24, 24, 24], false, false),
        ("constant_weak_alternating", [24, 24, 24], false, true),
        ("half_speed_intact", [24, 48, 24], false, false),
        ("double_speed_intact", [24, 12, 24], false, false),
        ("double_speed_weak_alternating", [24, 12, 24], false, true),
        ("non_octave_intact", [24, 32, 24], false, false),
        ("constant_all_weak", [24, 24, 24], true, false),
    ] {
        let (mut beat, bar, _truth) = pulses(periods, weak);
        if alternating {
            for center in (SECTION + 4 + periods[1]..SECTION * 2).step_by(periods[1] * 2) {
                beat[center - 1..=center + 1].fill(-2.);
            }
        }
        eprintln!("conditional meter authored control: {name}");
        let started = std::time::Instant::now();
        // Truth and authored section boundaries are never passed to the decoder.
        let decoded = conditional(name, &beat, &bar)?;
        eprintln!(
            "completed {name} in {:.2}s",
            started.elapsed().as_secs_f64()
        );
        rows.push(
            json!({"case":name,"beat_f64_le_sha256":digest(&beat),"bar_f64_le_sha256":digest(&bar),
            "conditional_meter":decoded}),
        );
    }
    let mut controls = Vec::new();
    for name in [
        "flat",
        "fixed_seed_noise",
        "unavailable_gap",
        "flat_middle",
        "intra_bar_change",
        "extra_offbeat_pulses",
        "three_beat_meter",
        "edge_phase_zero",
    ] {
        let (mut beat, mut bar, _) = pulses([24; 3], false);
        match name {
            "flat" => {
                beat.fill(-8.);
                bar.fill(-8.);
            }
            "fixed_seed_noise" => {
                let mut seed = 0x1357_2468_u32;
                for values in [&mut beat, &mut bar] {
                    for value in values {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        *value = -8. + f32::from(u8::try_from(seed >> 24).unwrap()) / 255. * 6.;
                    }
                }
            }
            // The frozen clock report supplies the unavailable spans/run split.
            "unavailable_gap" => {}
            "flat_middle" => {
                beat[480..672].fill(-8.);
                bar[480..672].fill(-8.);
            }
            "intra_bar_change" => {
                beat.fill(-8.);
                bar.fill(-8.);
                let mut frame = 4;
                let mut index = 0_usize;
                while frame + 1 < FRAMES {
                    beat[frame - 1..=frame + 1].fill(-2.);
                    if index.is_multiple_of(4) {
                        bar[frame - 1..=frame + 1].fill(-2.);
                    }
                    frame += if index < 19 { 24 } else { 32 };
                    index += 1;
                }
            }
            "extra_offbeat_pulses" => {
                for center in [515, 559, 602, 653, 701] {
                    beat[center - 1..=center + 1].fill(-2.);
                }
            }
            "three_beat_meter" => {
                bar.fill(-8.);
                for center in (4..FRAMES).step_by(72) {
                    bar[center - 1..=center + 1].fill(-2.);
                }
            }
            "edge_phase_zero" => {
                beat.fill(-8.);
                bar.fill(-8.);
                for (index, center) in (0..FRAMES).step_by(24).enumerate() {
                    beat[center.saturating_sub(1)..=center + 1].fill(-2.);
                    if index.is_multiple_of(4) {
                        bar[center.saturating_sub(1)..=center + 1].fill(-2.);
                    }
                }
            }
            _ => unreachable!(),
        }
        eprintln!("conditional meter boundary control: {name}");
        let started = std::time::Instant::now();
        let decoded = conditional(name, &beat, &bar)?;
        eprintln!(
            "completed {name} in {:.2}s",
            started.elapsed().as_secs_f64()
        );
        controls.push(json!({"case":name,"beat_f64_le_sha256":digest(&beat),"bar_f64_le_sha256":digest(&bar),"conditional_meter":decoded}));
    }
    Ok(
        json!({"schema_version":1,"purpose":"censored_conditional_meter_gate",
        "production_output_changed":false,"training_run":false,"holdout_opened":false,
        "real_music_evaluated":false,"truth_supplied_to_decoder":false,
        "beat_clock_searched":false,"conditions_on_frozen_inferred_ticks":true,
        "crop_controls":crop_controls()?,
        "meter_change_controls":meter_change_controls()?,
        "clock_report_sha256":format!("{:x}",Sha256::digest(include_bytes!("../../../evaluation/parity/time-clock-v1.json"))),
        "period_frames":[10,75],"meters":[2,7],"frame_rate_hz":50,
        "cases":rows,"controls":controls,
        "decoder_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/censored_meter.rs"))),
        "cell_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/phase_likelihood.rs"))),
        "audit_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("censored_meter.rs")))}),
    )
}

fn conditional(name: &str, beat: &[f32], bar: &[f32]) -> Result<serde_json::Value> {
    let report: serde_json::Value = serde_json::from_str(include_str!(
        "../../../evaluation/parity/time-clock-v1.json"
    ))?;
    let row = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(report["controls"].as_array().unwrap())
        .find(|r| r["case"] == name)
        .unwrap();
    anyhow::ensure!(
        row["beat_f64_le_sha256"] == digest(beat) && row["bar_f64_le_sha256"] == digest(bar),
        "changed frozen input"
    );
    let mut runs = Vec::new();
    for run in row["decoded"]["runs"].as_array().unwrap() {
        let ticks = run["map_ticks"].as_array().unwrap();
        let mut marks = Vec::new();
        for tick in ticks {
            let start = usize::try_from(tick["frame"].as_u64().unwrap())?;
            let duration = usize::try_from(tick["period_frames"].as_u64().unwrap())?;
            marks.push(reference::cell(&bar[start..start + duration], 0)?.log_ratio_to_null);
        }
        let inference = clock::infer(&marks, 2, 7)?;
        runs.push(json!({"frozen_ticks":ticks,"mark_log_ratios":marks,"meter":inference,
            "start":run["start"],"end":run["end"],"unchanged_edge_reference_frames":run["edge_reference_frames"]}));
    }
    Ok(json!({"runs":runs,"unavailable_spans":row["decoded"]["unavailable_spans"]}))
}

fn crop_controls() -> Result<Vec<serde_json::Value>> {
    let mut rows = Vec::new();
    for meter in 2..=7 {
        for phase in 0..meter {
            let marks: Vec<f64> = (0..48)
                .map(|i| {
                    let mut cell = vec![-8.; 24];
                    if (i + phase) % meter == 0 {
                        cell[0..=1].fill(-2.);
                        cell[23] = -2.;
                    }
                    reference::cell(&cell, 0).unwrap().log_ratio_to_null
                })
                .collect();
            for removed in 0..meter {
                let length = marks.len() - removed;
                let inference = clock::infer(&marks[..length], 2, 7)?;
                rows.push(
                    json!({"authored_meter":meter,"authored_initial_phase":phase,
                    "visible_beats":length,"mark_log_ratios":&marks[..length],"inference":inference}),
                );
            }
        }
    }
    Ok(rows)
}

fn meter_change_controls() -> Result<Vec<serde_json::Value>> {
    let mut rows = Vec::new();
    for (name, first, second) in [
        ("four_to_three", 4, 3),
        ("three_to_four", 3, 4),
        ("four_to_two", 4, 2),
        ("two_to_four", 2, 4),
    ] {
        let mut marks = Vec::new();
        let mut meters = Vec::new();
        for meter in [first, second] {
            for i in 0..24 {
                let mut cell = vec![-8.; 24];
                if i % meter == 0 {
                    cell[0..=1].fill(-2.);
                    cell[23] = -2.;
                }
                marks.push(reference::cell(&cell, 0)?.log_ratio_to_null);
                meters.push(meter);
            }
        }
        let inference = clock::infer(&marks, 2, 7)?;
        rows.push(json!({"case":name,"authored_meters":meters,
            "mark_log_ratios":marks,"inference":inference}));
    }
    Ok(rows)
}

fn main() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&audit()?)?);
    Ok(())
}
