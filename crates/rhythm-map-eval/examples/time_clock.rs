//! Truth-free time clock reference: authored controls, not production inference.
use anyhow::Result;
use serde_json::json;
use sha2::{Digest, Sha256};
#[path = "support/time_clock.rs"]
mod clock;
#[cfg(test)]
#[path = "support/phase_likelihood.rs"]
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
        let (mut beat, bar, truth) = pulses(periods, weak);
        if alternating {
            for center in (SECTION + 4 + periods[1]..SECTION * 2).step_by(periods[1] * 2) {
                beat[center - 1..=center + 1].fill(-2.);
            }
        }
        eprintln!("time clock authored control: {name}");
        let started = std::time::Instant::now();
        // Truth and authored section boundaries are never passed to the decoder.
        let decoded = clock::decode(&beat, &bar, None, clock::Domain::default())?;
        eprintln!(
            "completed {name} in {:.2}s",
            started.elapsed().as_secs_f64()
        );
        let run = &decoded.runs[0];
        let exact_matches = run
            .map_ticks
            .iter()
            .filter(|t| truth.contains(&t.frame))
            .count();
        let correct_periods = run
            .map_ticks
            .iter()
            .filter(|t| t.period_frames == periods[t.frame / SECTION])
            .count();
        rows.push(
            json!({"case":name,"beat_f64_le_sha256":digest(&beat),"bar_f64_le_sha256":digest(&bar),
            "authored_beats":truth.len(),"exact_frame_matches":exact_matches,
            "ticks_with_authored_period":correct_periods,"decoded":decoded}),
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
        let mut available = vec![true; FRAMES];
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
            "unavailable_gap" => {
                available[480..672].fill(false);
            }
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
        eprintln!("time clock boundary control: {name}");
        let started = std::time::Instant::now();
        let decoded = clock::decode(&beat, &bar, Some(&available), clock::Domain::default())?;
        eprintln!(
            "completed {name} in {:.2}s",
            started.elapsed().as_secs_f64()
        );
        controls.push(json!({"case":name,"beat_f64_le_sha256":digest(&beat),"bar_f64_le_sha256":digest(&bar),"decoded":decoded}));
    }
    Ok(
        json!({"schema_version":1,"purpose":"time_exposure_joint_clock_authored_gate",
        "production_output_changed":false,"training_run":false,"holdout_opened":false,
        "real_music_evaluated":false,"truth_supplied_to_decoder":false,
        "rate_per_frame":clock::time_prior::Prior::new(&(10..=75).collect::<Vec<_>>()).rate_per_frame,
        "prior_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/time_prior.rs"))),
        "period_frames":[10,75],"meters":[2,7],"frame_rate_hz":50,
        "cases":rows,"controls":controls,
        "decoder_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("support/time_clock.rs"))),
        "audit_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("time_clock.rs")))}),
    )
}

fn main() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&audit()?)?);
    Ok(())
}
