//! Export private, calibration-only parity tensors; not a product interface.

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rhythm_map_beat_this::{
    BeatThisBackend, DecodedAudio, OBSERVATION_CONTRACT, decode_audio,
    preprocessing_diagnostics::decode_native,
};
use rhythm_map_eval::{EvaluationSuite, ExternalAudioResolver};
use rhythm_map_models::{ModelArtifactRole, verify_model_pack};
use serde_json::json;
use sha2::{Digest, Sha256};

mod support;
use support::calibration_audio;

#[path = "support/reference_resampler.rs"]
mod reference_resampler;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    case: String,
    #[arg(long)]
    audio_dir: PathBuf,
    #[arg(long, default_value = "models/beat-this-full-v1.json")]
    model_pack: PathBuf,
    #[arg(long)]
    model_dir: PathBuf,
    /// Limit to the first N seconds AFTER the shipping file decoder/resampler.
    #[arg(long, default_value_t = 35)]
    seconds: u32,
    /// Include the former upstream PCM for a private phase/tail ablation.
    /// This does not select a different backend or change the traced inference.
    #[arg(long)]
    include_legacy_pcm: bool,
    /// Evaluate the frozen reference-bandwidth candidate, never a product option.
    #[arg(long, conflicts_with = "include_legacy_pcm")]
    reference_resampler: bool,
    /// New private file. Contains audio samples; never commit or redistribute it.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=60).contains(&args.seconds),
        "trace duration must be 1..=60 seconds"
    );
    ensure!(
        !args.output.exists(),
        "refusing to replace an existing trace"
    );
    let suite_bytes = fs::read(&args.suite)?;
    let suite: EvaluationSuite = serde_json::from_slice(&suite_bytes)?;
    let audio = calibration_audio(&suite, &args.case)?;
    let path = ExternalAudioResolver::new(&args.audio_dir)?.resolve(&audio)?;
    let pack = verify_model_pack(&args.model_pack, &args.model_dir)?;
    ensure!(
        pack.manifest().backend == "beat-this-rten",
        "expected Beat This pack"
    );
    let mut decoded = if args.reference_resampler {
        let native = decode_native(&path)?;
        DecodedAudio {
            samples: reference_resampler::resample(&native.samples, native.sample_rate)?,
            sample_rate: 22050,
        }
    } else {
        decode_audio(&path)?
    };
    let observation_contract = if args.reference_resampler {
        format!("{OBSERVATION_CONTRACT}+{}", reference_resampler::ID)
    } else {
        OBSERVATION_CONTRACT.to_owned()
    };
    let decoded_sample_count = decoded.samples.len();
    let legacy_audio = if args.include_legacy_pcm {
        let mut legacy = beat_this::load_audio(&path, 22_050)?;
        let sample_count = legacy.samples.len();
        legacy
            .samples
            .truncate(usize::try_from(args.seconds)? * 22_050);
        Some(json!({
            "implementation": "beat-this-1.0.0",
            "sample_rate": legacy.sample_rate,
            "decoded_sample_count": sample_count,
            "mono_samples": legacy.samples,
        }))
    } else {
        None
    };
    decoded
        .samples
        .truncate(usize::try_from(args.seconds)? * usize::try_from(decoded.sample_rate)?);
    let mut backend = BeatThisBackend::load_with_model_identity(
        pack.path_for(ModelArtifactRole::MelFrontend)
            .context("missing mel model")?,
        pack.path_for(ModelArtifactRole::BeatModel)
            .context("missing beat model")?,
        pack.manifest().id.clone(),
        Some(pack.manifest_sha256().to_owned()),
    )?;
    eprintln!("Tracing {}: {} samples", args.case, decoded.samples.len());
    let trace = backend.trace_mono(&decoded.samples, decoded.sample_rate)?;
    let report = json!({
        "schema_version": 1, "purpose": "calibration_parity_private",
        "suite_id": suite.id, "case_id": args.case,
        "suite_sha256": format!("{:x}", Sha256::digest(&suite_bytes)),
        "trace_exporter_sha256": format!("{:x}", Sha256::digest(include_bytes!("beat_this_trace.rs"))),
        "trace_support_sha256": format!("{:x}", Sha256::digest(include_bytes!("support/mod.rs"))),
        "adapter_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../rhythm-map-beat-this/src/lib.rs"))),
        "audio_preprocessing_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../rhythm-map-beat-this/src/audio.rs"))),
        "audio_sha256": audio.sha256,
        "model_manifest_sha256": pack.manifest_sha256(),
        "observation_contract": observation_contract,
        "preprocessing_candidate": args.reference_resampler.then_some(reference_resampler::ID),
        "candidate_source_sha256": args.reference_resampler.then(|| format!("{:x}", Sha256::digest(include_bytes!("support/reference_resampler.rs")))),
        "prefix_seconds": args.seconds, "sample_rate": decoded.sample_rate,
        "decoded_sample_count": decoded_sample_count,
        "legacy_audio": legacy_audio,
        "mono_samples": decoded.samples,
        "mel_shape": trace.mel_shape, "mel_values": trace.mel_values,
        "beat_logits": trace.inference.beat_logits(),
        "downbeat_logits": trace.inference.downbeat_logits(),
        "upstream_beats": trace.upstream_beats,
        "upstream_downbeats": trace.upstream_downbeats,
        "observations": trace.observations,
    });
    let mut writer = BufWriter::new(File::create_new(&args.output)?);
    serde_json::to_writer(&mut writer, &report)?;
    writer.flush()?;
    eprintln!("Private trace written; contains PCM and must stay outside Git.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhythm_map_eval::SuitePurpose;

    #[test]
    fn rejects_holdout_before_resolving_audio_or_loading_models() {
        let mut suite: EvaluationSuite =
            serde_json::from_str(include_str!("../../../evaluation/suites/artbeat-v1.json"))
                .unwrap();
        let id = suite.cases[0].id.clone();
        assert!(calibration_audio(&suite, &id).is_ok());
        for purpose in [SuitePurpose::Holdout, SuitePurpose::Regression] {
            suite.purpose = purpose;
            assert!(calibration_audio(&suite, &id).is_err());
        }
    }
}
