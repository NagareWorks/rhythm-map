//! Generated-signal preprocessing probes, independent of music and models.

use anyhow::{Result, ensure};
use clap::Parser;
use rhythm_map_beat_this::{OBSERVATION_CONTRACT, preprocessing_diagnostics::prepare_mono};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

const RATES: [u32; 7] = [8000, 16000, 22050, 44100, 48000, 96000, 192_000];

#[path = "support/reference_resampler.rs"]
mod reference_resampler;

#[derive(Parser)]
struct Args {
    /// New generated-signal trace; use a data-drive artifact directory.
    #[arg(long)]
    output: PathBuf,
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn signals(rate: u32) -> Vec<(String, Vec<f32>, Option<f64>)> {
    let len = rate as usize / 2;
    let period = (rate / gcd(rate, 22050)) as usize;
    let center = (len / 2 / period) * period;
    let mut result = Vec::new();
    for (name, position) in [
        ("impulse_start", 0),
        ("impulse_center", center),
        ("impulse_fractional", center + 1),
        ("impulse_tail", len - 1),
    ] {
        let mut samples = vec![0.0; len];
        samples[position] = 1.0;
        result.push((
            name.into(),
            samples,
            Some(position as f64 / f64::from(rate)),
        ));
    }
    result.push(("dc".into(), vec![0.25; len], None));
    result.push((
        "step".into(),
        (0..len)
            .map(|i| if i < center { 0.0 } else { 0.25 })
            .collect(),
        None,
    ));
    let nyquist = f64::from(rate.min(22050)) / 2.0;
    let sweep_end = f64::from(rate) * 0.49;
    let sweep = (0..len)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            (0.5 * (std::f64::consts::TAU * (30.0 * t + (sweep_end - 30.0) * t * t)).sin()) as f32
        })
        .collect();
    result.push(("linear_sweep".into(), sweep, None));
    for relative in [0.25, 0.85, 0.913, 0.94, 0.95, 0.97, 1.02, 1.10] {
        let hz = relative * nyquist;
        if hz >= f64::from(rate) / 2.0 {
            continue;
        }
        let samples = (0..len)
            .map(|i| (0.5 * (std::f64::consts::TAU * hz * i as f64 / f64::from(rate)).sin()) as f32)
            .collect();
        result.push((format!("tone_{relative:.3}"), samples, Some(hz)));
    }
    result
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "refusing to replace a generated trace"
    );
    let mut cases = Vec::new();
    for rate in RATES {
        for (signal, samples, parameter) in signals(rate) {
            let started = Instant::now();
            let current = prepare_mono(&samples, rate)?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let candidate = reference_resampler::resample(&samples, rate)?;
            let candidate_elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            cases.push(
                json!({"sample_rate": rate, "signal": signal, "parameter": parameter,
                              "input_pcm": samples, "current_pcm": current,
                              "current_elapsed_ms": elapsed_ms, "candidate_pcm": candidate,
                              "candidate_elapsed_ms": candidate_elapsed_ms}),
            );
        }
    }
    let report = json!({"schema_version": 1, "purpose": "generated_resampler_characterization",
        "model_sample_rate": 22050, "observation_contract": OBSERVATION_CONTRACT,
        "probe_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("resampler_probe.rs"))),
        "adapter_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../rhythm-map-beat-this/src/lib.rs"))),
        "audio_preprocessing_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../rhythm-map-beat-this/src/audio.rs"))),
        "candidate_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("support/reference_resampler.rs"))),
        "candidate": reference_resampler::ID,
        "cases": cases});
    let mut writer = BufWriter::new(File::create_new(args.output)?);
    serde_json::to_writer(&mut writer, &report)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_are_deterministic_and_center_impulses_align_to_both_grids() {
        for rate in RATES {
            let cases = signals(rate);
            assert_eq!(cases, signals(rate));
            for (_, samples, _) in &cases {
                assert_eq!(samples.len(), rate as usize / 2);
                assert!(samples.iter().all(|s| s.is_finite()));
            }
            let center = cases.iter().find(|c| c.0 == "impulse_center").unwrap();
            let pos = center.1.iter().position(|&x| x == 1.0).unwrap();
            assert_eq!(pos * 22050 % rate as usize, 0);
        }
    }
}
