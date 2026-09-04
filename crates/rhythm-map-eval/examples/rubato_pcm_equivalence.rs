//! Model-free, full-recording input audit. Reports contain no audio samples.
use anyhow::{Result, ensure};
use clap::Parser;
use rhythm_map_beat_this::{OBSERVATION_CONTRACT, decode_audio, preprocessing_diagnostics};
use rhythm_map_eval::{EvaluationSuite, ExternalAudioResolver};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf};

mod support;

const LOCK: &[u8] =
    include_bytes!("../../../evaluation/parity/rubato-pcm-equivalence-lock-v1.json");

#[derive(Parser)]
struct Args {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    audio_dir: PathBuf,
    /// New summary file; never contains PCM or private paths.
    #[arg(long)]
    output: PathBuf,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn pcm_sha(samples: &[f32]) -> String {
    let mut digest = Sha256::new();
    for sample in samples {
        digest.update(sample.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn locked_suite(bytes: &[u8]) -> Result<EvaluationSuite> {
    let lock: Value = serde_json::from_slice(LOCK)?;
    ensure!(
        sha(bytes) == lock["suite_sha256"],
        "suite bytes differ from lock"
    );
    let suite: EvaluationSuite = serde_json::from_slice(bytes)?;
    ensure!(
        suite.id == lock["suite_id"] && suite.cases.len() == 25,
        "wrong cohort"
    );
    ensure!(
        OBSERVATION_CONTRACT == lock["shipping_contract"],
        "contract changed"
    );
    for case in &suite.cases {
        support::calibration_audio(&suite, &case.id)?;
    }
    Ok(suite)
}

fn compare(left: &[f32], right: &[f32]) -> Result<Value> {
    ensure!(!left.is_empty() && !right.is_empty(), "empty PCM");
    ensure!(
        left.iter().chain(right).all(|x| x.is_finite()),
        "nonfinite PCM"
    );
    let different = left
        .iter()
        .zip(right)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    Ok(json!({
        "legacy_sample_count": left.len(), "shipping_sample_count": right.len(),
        "legacy_pcm_sha256": pcm_sha(left), "shipping_pcm_sha256": pcm_sha(right),
        "differing_shared_samples": different,
        "unpaired_samples": left.len().abs_diff(right.len()),
        "bit_identical": left.len() == right.len() && different == 0
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refusing to replace existing report");
    let suite_bytes = fs::read(&args.suite)?;
    let suite = locked_suite(&suite_bytes)?;
    let resolver = ExternalAudioResolver::new(&args.audio_dir)?;
    let mut cases = Vec::new();
    for case in &suite.cases {
        let audio = support::calibration_audio(&suite, &case.id)?;
        let path = resolver.resolve(&audio)?;
        let native = preprocessing_diagnostics::decode_native(&path)?;
        ensure!(
            native.sample_rate == 22_050,
            "non-native model rate: {}",
            case.id
        );
        let native_count = native.samples.len();
        let native_hash = pcm_sha(&native.samples);
        drop(native);
        let legacy = beat_this::load_audio(&path, 22_050)?;
        let shipping = decode_audio(&path)?;
        ensure!(
            legacy.sample_rate == 22_050 && shipping.sample_rate == 22_050,
            "wrong decoded rate"
        );
        ensure!(
            shipping.samples.len() == native_count && pcm_sha(&shipping.samples) == native_hash,
            "native bypass changed samples"
        );
        // Validate the source again after both decoders, before accepting evidence.
        resolver.resolve(&audio)?;
        let comparison = compare(&legacy.samples, &shipping.samples)?;
        eprintln!("{}: bit_identical={}", case.id, comparison["bit_identical"]);
        cases.push(json!({"id": case.id, "audio_sha256": audio.sha256,
            "native_sample_rate_hz": 22050, "comparison": comparison}));
    }
    let identical = cases
        .iter()
        .filter(|c| c["comparison"]["bit_identical"] == true)
        .count();
    let report = json!({
        "schema_version": 1, "purpose": "calibration_pcm_equivalence_summary",
        "lock_sha256": sha(LOCK), "suite_sha256": sha(&suite_bytes),
        "auditor_sha256": sha(include_bytes!("rubato_pcm_equivalence.rs")),
        "support_sha256": sha(include_bytes!("support/mod.rs")),
        "cargo_lock_sha256": sha(include_bytes!("../../../Cargo.lock")),
        "adapter_sha256": sha(include_bytes!("../../rhythm-map-beat-this/src/lib.rs")),
        "preprocessing_sha256": sha(include_bytes!("../../rhythm-map-beat-this/src/audio.rs")),
        "case_count": cases.len(), "bit_identical_cases": identical,
        "cache_reuse_authorized": false, "model_inference_runs": 0,
        "cache_reads": 0, "cache_writes": 0, "cases": cases
    });
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    writeln!(file)?;
    ensure!(identical == 25, "input equivalence failed; see summary");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_bits_and_full_length_not_numeric_equality() {
        assert_eq!(
            compare(&[0.0, 1.0], &[0.0, 1.0]).unwrap()["bit_identical"],
            true
        );
        let sign = compare(&[0.0], &[-0.0]).unwrap();
        assert_eq!(sign["bit_identical"], false);
        assert_eq!(sign["differing_shared_samples"], 1);
        let tail = compare(&[1.0, 2.0], &[1.0]).unwrap();
        assert_eq!(tail["bit_identical"], false);
        assert_eq!(tail["unpaired_samples"], 1);
        assert_eq!(compare(&[1.0], &[2.0]).unwrap()["bit_identical"], false);
    }

    #[test]
    fn rejects_invalid_pcm() {
        for samples in [
            vec![],
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![f32::NEG_INFINITY],
        ] {
            assert!(compare(&samples, &[0.0]).is_err());
            assert!(compare(&[0.0], &samples).is_err());
        }
    }

    #[test]
    fn locks_full_cohort_before_decoding() {
        let bytes = include_bytes!("../../../evaluation/suites/rubato-calibration-v1.json");
        locked_suite(bytes).unwrap();
        let mut suite: Value = serde_json::from_slice(bytes).unwrap();
        suite["purpose"] = json!("holdout");
        assert!(locked_suite(&serde_json::to_vec(&suite).unwrap()).is_err());
        let mut suite: Value = serde_json::from_slice(bytes).unwrap();
        suite["cases"].as_array_mut().unwrap().swap(0, 1);
        assert!(locked_suite(&serde_json::to_vec(&suite).unwrap()).is_err());
        suite["cases"].as_array_mut().unwrap().pop();
        assert!(locked_suite(&serde_json::to_vec(&suite).unwrap()).is_err());
    }
}
