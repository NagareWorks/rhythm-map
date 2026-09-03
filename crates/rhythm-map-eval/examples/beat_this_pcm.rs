//! Private, model-free controlled PCM export. Never ship or commit its outputs.

use anyhow::{Result, ensure};
use clap::Parser;
use rhythm_map_beat_this::{
    OBSERVATION_CONTRACT, decode_audio,
    preprocessing_diagnostics::{decode_native, prepare_mono},
};
use rhythm_map_eval::{EvaluationSuite, ExternalAudioResolver};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

mod support;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    case: String,
    #[arg(long)]
    audio_dir: PathBuf,
    /// Official native-rate mono float32 PCM, with matching source identity.
    #[arg(long)]
    reference_native: PathBuf,
    /// New private file containing PCM. Never commit or redistribute it.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize)]
struct ReferenceNative {
    schema_version: u32,
    purpose: String,
    case_id: String,
    suite_sha256: String,
    audio_sha256: String,
    sample_rate: u32,
    mono_samples: Vec<f32>,
}

impl ReferenceNative {
    fn validate(&self, case: &str, suite_sha: &str, audio_sha: &str, rate: u32) -> Result<()> {
        ensure!(
            self.schema_version == 1 && self.purpose == "calibration_native_pcm_private",
            "invalid native PCM schema/purpose"
        );
        ensure!(
            self.case_id == case
                && self.suite_sha256 == suite_sha
                && self.audio_sha256 == audio_sha,
            "native PCM source identity mismatch"
        );
        ensure!(
            self.sample_rate == rate && rate > 0,
            "native PCM rate mismatch"
        );
        ensure!(
            !self.mono_samples.is_empty() && self.mono_samples.iter().all(|s| s.is_finite()),
            "invalid native mono PCM"
        );
        ensure!(
            self.mono_samples.len() as u128 <= u128::from(rate) * 60,
            "native PCM diagnosis limited to complete clips of at most 60 seconds"
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "refusing to replace an existing PCM trace"
    );
    let suite_bytes = fs::read(&args.suite)?;
    let suite: EvaluationSuite = serde_json::from_slice(&suite_bytes)?;
    let audio = support::calibration_audio(&suite, &args.case)?;
    let path = ExternalAudioResolver::new(&args.audio_dir)?.resolve(&audio)?;
    let native = decode_native(&path)?;
    ensure!(
        native.samples.len() as u128 <= u128::from(native.sample_rate) * 60,
        "native PCM diagnosis requires complete clips of at most 60 seconds"
    );
    let suite_sha = format!("{:x}", Sha256::digest(&suite_bytes));
    let reference_bytes = fs::read(&args.reference_native)?;
    let reference: ReferenceNative = serde_json::from_slice(&reference_bytes)?;
    reference.validate(&args.case, &suite_sha, &audio.sha256, native.sample_rate)?;
    let rust_resampled = prepare_mono(&native.samples, native.sample_rate)?;
    let shipping = decode_audio(&path)?;
    ensure!(
        shipping.sample_rate == 22_050
            && shipping
                .samples
                .iter()
                .map(|s| s.to_bits())
                .eq(rust_resampled.iter().map(|s| s.to_bits())),
        "native-stage reconstruction differs from shipping decode"
    );
    let reference_resampled = prepare_mono(&reference.mono_samples, reference.sample_rate)?;
    let report = json!({
        "schema_version": 1, "purpose": "calibration_native_pcm_private",
        "suite_id": suite.id, "suite_sha256": suite_sha, "case_id": args.case,
        "audio_sha256": audio.sha256, "observation_contract": OBSERVATION_CONTRACT,
        "reference_native_sha256": format!("{:x}", Sha256::digest(&reference_bytes)),
        "exporter_sha256": format!("{:x}", Sha256::digest(include_bytes!("beat_this_pcm.rs"))),
        "support_sha256": format!("{:x}", Sha256::digest(include_bytes!("support/mod.rs"))),
        "adapter_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../rhythm-map-beat-this/src/lib.rs"))),
        "audio_preprocessing_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../rhythm-map-beat-this/src/audio.rs"))),
        "native_sample_rate": native.sample_rate,
        "rust_native_mono": native.samples,
        "model_sample_rate": shipping.sample_rate,
        "rust_native_rust_resampled": rust_resampled,
        "reference_native_rust_resampled": reference_resampled,
        "shipping_reconstruction_bit_exact": true,
    });
    let mut writer = BufWriter::new(File::create_new(&args.output)?);
    serde_json::to_writer(&mut writer, &report)?;
    writer.flush()?;
    eprintln!("Private native PCM trace written; no model inference or timestamp shifting.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> ReferenceNative {
        ReferenceNative {
            schema_version: 1,
            purpose: "calibration_native_pcm_private".into(),
            case_id: "case".into(),
            suite_sha256: "suite".into(),
            audio_sha256: "audio".into(),
            sample_rate: 44100,
            mono_samples: vec![0.25],
        }
    }

    #[test]
    fn rejects_wrong_source_rate_or_purpose() {
        let mut input = reference();
        assert!(input.validate("case", "suite", "audio", 44100).is_ok());
        for (case, suite, audio, rate) in [
            ("other", "suite", "audio", 44100),
            ("case", "other", "audio", 44100),
            ("case", "suite", "other", 44100),
            ("case", "suite", "audio", 48000),
        ] {
            assert!(input.validate(case, suite, audio, rate).is_err());
        }
        input.purpose = "holdout".into();
        assert!(input.validate("case", "suite", "audio", 44100).is_err());
    }

    #[test]
    fn rejects_nonfinite_empty_or_overlong_pcm() {
        for samples in [vec![], vec![f32::NAN], vec![f32::INFINITY], vec![0.0; 61]] {
            let mut input = reference();
            input.sample_rate = 1;
            input.mono_samples = samples;
            assert!(input.validate("case", "suite", "audio", 1).is_err());
        }
    }
}
