//! Private full-recording frame evidence, never a production cache or decoder.

use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rhythm_map_beat_this::{
    BeatThisBackend, OBSERVATION_CONTRACT, PeakPickingOptions, decode_audio,
};
use rhythm_map_core::{BeatCandidate, ModelInfo, ObservedBeat, RhythmObservations};
use rhythm_map_eval::{EvaluationSuite, ExternalAudioResolver, SuitePurpose};
use rhythm_map_models::{ModelArtifactRole, verify_model_pack};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

mod support;

const MODEL: &str = "ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    evidence: PathBuf,
    #[arg(long)]
    audio_dir: PathBuf,
    #[arg(long, default_value = "models/beat-this-full-v1.json")]
    model_pack: PathBuf,
    #[arg(long)]
    model_dir: PathBuf,
    /// New private directory outside every Git worktree; no PCM/mel export.
    #[arg(long)]
    output_dir: PathBuf,
}

// Deserialize only the required projection: dense acoustic data and truth
// coordinates in the historical evidence are skipped, not used for inference.
#[derive(Deserialize)]
struct Evidence {
    cases: Vec<FrozenCase>,
}

#[derive(Deserialize)]
struct FrozenCase {
    id: String,
    audio_sha256: String,
    pcm_sha256: String,
    sample_count: usize,
    sample_rate: u32,
    score_replay_exact: bool,
    observations: FrozenObservations,
}

#[derive(Clone, Deserialize)]
struct FrozenObservations {
    duration_s: f64,
    beats: Vec<ObservedBeat>,
    beat_candidates: Vec<BeatCandidate>,
    source: ModelInfo,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn pcm_sha(samples: &[f32]) -> String {
    let mut hash = Sha256::new();
    for sample in samples {
        hash.update(sample.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn calibration_lock(suite: &EvaluationSuite, bytes: &[u8]) -> Result<(usize, &'static str)> {
    suite.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        suite.purpose == SuitePurpose::Calibration,
        "calibration only; holdout/regression rejected"
    );
    let (count, suite_digest, evidence_digest) = match suite.id.as_str() {
        "artbeat-v1" => (
            15,
            "21f3d44bacbfe9c50dfbc889990c563d44e406d56558492627402d21e5a7e81b",
            "3f1ba43fd4f373579727a48668d8de8e00166523d2d1141e072bc3471a71ab3e",
        ),
        "rubato-calibration-v1" => (
            25,
            "c10c229bbf7b89ebd23dd2b4ff2a2d19aaec9b5f28d2b5eb6d121d950fb62653",
            "ce5e678276888a0e430c004444dce4b27f0cfac0761767736abee2ec3fc05937",
        ),
        _ => anyhow::bail!("only the two frozen calibration cohorts are allowed"),
    };
    ensure!(
        sha(bytes) == suite_digest && suite.cases.len() == count,
        "suite identity/coverage changed"
    );
    Ok((count, evidence_digest))
}

fn validate_heads(beat: &[f32], downbeat: &[f32]) -> Result<()> {
    ensure!(
        !beat.is_empty() && beat.len() == downbeat.len(),
        "empty/mismatched frame heads"
    );
    ensure!(
        beat.iter().chain(downbeat).all(|v| v.is_finite()),
        "non-finite frame logits"
    );
    Ok(())
}

fn compare(actual: &RhythmObservations, expected: &FrozenObservations) -> Value {
    let beat_times = actual
        .beats
        .iter()
        .map(|b| b.time_s.to_bits())
        .eq(expected.beats.iter().map(|b| b.time_s.to_bits()));
    let candidate_times = actual
        .beat_candidates
        .iter()
        .map(|b| b.time_s.to_bits())
        .eq(expected.beat_candidates.iter().map(|b| b.time_s.to_bits()));
    let beats = actual.beats == expected.beats;
    let candidates = actual.beat_candidates == expected.beat_candidates;
    let source = actual.source == expected.source;
    let duration = actual.duration_s.to_bits() == expected.duration_s.to_bits();
    let raw_only = actual.activations.is_none()
        && actual.activity.is_empty()
        && actual.onsets.is_empty()
        && actual.harmonic_changes.is_empty();
    let confidence_delta = (actual.beats.len() == expected.beats.len()).then(|| {
        actual
            .beats
            .iter()
            .zip(&expected.beats)
            .map(|(a, b)| {
                (a.confidence - b.confidence)
                    .abs()
                    .max((a.downbeat_confidence - b.downbeat_confidence).abs())
            })
            .fold(0.0_f64, f64::max)
    });
    json!({"exact": beats && candidates && source && duration && raw_only && beat_times && candidate_times,
        "beat_timestamps_exact": beat_times, "candidate_timestamps_exact": candidate_times,
        "beats_with_confidence_exact": beats, "candidates_with_confidence_exact": candidates,
        "source_metadata_exact": source, "duration_bits_exact": duration, "raw_only": raw_only,
        "expected_beat_count": expected.beats.len(), "actual_beat_count": actual.beats.len(),
        "expected_candidate_count": expected.beat_candidates.len(), "actual_candidate_count": actual.beat_candidates.len(),
        "max_paired_beat_confidence_abs_error": confidence_delta})
}

fn private_destination(path: &Path) -> Result<PathBuf> {
    ensure!(
        !path.exists(),
        "refusing to overwrite an existing capture directory"
    );
    let parent = path
        .parent()
        .context("output needs a parent")?
        .canonicalize()?;
    ensure!(
        !parent.ancestors().any(|p| p.join(".git").exists()),
        "dense evidence must stay outside Git worktrees"
    );
    Ok(parent.join(path.file_name().context("output needs a directory name")?))
}

fn write_json(path: &Path, value: &Value) -> Result<String> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    let mut file = File::create_new(path)?;
    file.write_all(&bytes)?;
    file.flush()?;
    Ok(sha(&bytes))
}

fn complete(expected: usize, records: &[Value]) -> bool {
    expected > 0
        && records.len() == expected
        && records.iter().all(|r| r["replay"]["exact"] == true)
}

fn load_evidence(
    path: &Path,
    suite: &EvaluationSuite,
    count: usize,
    evidence_digest: &str,
) -> Result<Evidence> {
    let evidence_bytes = fs::read(path)?;
    ensure!(
        sha(&evidence_bytes) == evidence_digest,
        "frozen evidence identity changed"
    );
    let evidence: Evidence = serde_json::from_slice(&evidence_bytes)?;
    drop(evidence_bytes);
    ensure!(evidence.cases.len() == count, "evidence cohort incomplete");
    for (case, frozen) in suite.cases.iter().zip(&evidence.cases) {
        ensure!(
            case.id == frozen.id && frozen.score_replay_exact,
            "case order/replay mismatch"
        );
        ensure!(
            case.id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "unsafe case filename"
        );
        let audio = support::calibration_audio(suite, &case.id)?;
        ensure!(
            audio.sha256 == frozen.audio_sha256,
            "audio reference changed"
        );
    }
    Ok(evidence)
}

fn capture(args: &Args) -> Result<Value> {
    let suite_bytes = fs::read(&args.suite)?;
    let suite: EvaluationSuite = serde_json::from_slice(&suite_bytes)?;
    let (count, evidence_digest) = calibration_lock(&suite, &suite_bytes)?;
    let evidence = load_evidence(&args.evidence, &suite, count, evidence_digest)?;
    let output = private_destination(&args.output_dir)?;
    let pack = verify_model_pack(&args.model_pack, &args.model_dir)?;
    ensure!(
        pack.manifest_sha256() == MODEL && pack.manifest().backend == "beat-this-rten",
        "model identity changed"
    );
    let mut backend = BeatThisBackend::load(
        pack.path_for(ModelArtifactRole::MelFrontend)
            .context("missing frontend")?,
        pack.path_for(ModelArtifactRole::BeatModel)
            .context("missing beat model")?,
    )?;
    let resolver = ExternalAudioResolver::new(&args.audio_dir)?;
    fs::create_dir(&output)?;
    let mut records = Vec::new();
    for (index, frozen) in evidence.cases.iter().enumerate() {
        eprintln!(
            "Dense capture {}/{count}: {} (complete recording)",
            index + 1,
            frozen.id
        );
        let audio = support::calibration_audio(&suite, &frozen.id)?;
        let decoded = decode_audio(resolver.resolve(&audio)?)?;
        let pcm_digest = pcm_sha(&decoded.samples);
        ensure!(
            decoded.sample_rate == frozen.sample_rate
                && decoded.samples.len() == frozen.sample_count
                && pcm_digest == frozen.pcm_sha256,
            "full PCM identity mismatch: {}",
            frozen.id
        );
        let start = Instant::now();
        let inference = backend.infer_mono(&decoded.samples, decoded.sample_rate)?;
        let elapsed = start.elapsed().as_secs_f64();
        validate_heads(inference.beat_logits(), inference.downbeat_logits())?;
        let observations = backend.decode_inference(&inference, PeakPickingOptions::default())?;
        let replay = compare(&observations, &frozen.observations);
        let metadata = json!({"schema_version": 1, "purpose": "private_full_recording_dense_evidence",
            "suite_id": suite.id, "suite_sha256": sha(&suite_bytes), "case_id": frozen.id,
            "frozen_evidence_sha256": evidence_digest, "audio_sha256": frozen.audio_sha256,
            "pcm_sha256": pcm_digest, "sample_count": decoded.samples.len(), "sample_rate": decoded.sample_rate,
            "model_manifest_sha256": MODEL, "observation_contract": OBSERVATION_CONTRACT,
            "exporter_source_sha256": sha(include_bytes!("dense_beat_evidence.rs")),
            "adapter_source_sha256": sha(include_bytes!("../../rhythm-map-beat-this/src/lib.rs")),
            "audio_source_sha256": sha(include_bytes!("../../rhythm-map-beat-this/src/audio.rs")),
            "cargo_lock_sha256": sha(include_bytes!("../../../Cargo.lock")),
            "runtime_os": std::env::consts::OS, "runtime_arch": std::env::consts::ARCH,
            "rten_num_threads": std::env::var("RTEN_NUM_THREADS").ok(),
            "frame_rate_hz": 50, "start_time_s": 0, "frame_count": inference.beat_logits().len(),
            "inference_elapsed_s": elapsed, "replay": replay});
        let mut payload = metadata.clone();
        payload["beat_logits"] = json!(inference.beat_logits());
        payload["downbeat_logits"] = json!(inference.downbeat_logits());
        payload["observations"] = json!(observations);
        let digest = write_json(&output.join(format!("{}.json", frozen.id)), &payload)?;
        let mut record = metadata;
        record["capture_sha256"] = json!(digest);
        records.push(record);
        eprintln!(
            "  {} frames; {:.2}s inference; exact replay: {}",
            inference.beat_logits().len(),
            elapsed,
            replay["exact"]
        );
        if replay["exact"] != true {
            eprintln!("Stopping at the first mismatch; no complete-cohort or accuracy claim.");
            break;
        }
    }
    let report = json!({"schema_version": 1, "purpose": "full_recording_dense_capture_summary",
        "complete": complete(count, &records), "expected_case_count": count,
        "completed_inference_count": records.len(), "cache_writes": 0, "training_run": false,
        "production_observations_changed": false, "accuracy_improvement_claimed": false, "cases": records});
    write_json(&output.join("summary.json"), &report)?;
    Ok(report)
}

fn main() -> Result<()> {
    let report = capture(&Args::parse())?;
    ensure!(
        report["complete"] == true,
        "capture retained for diagnosis; exact replay failed"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observations() -> RhythmObservations {
        RhythmObservations {
            duration_s: 2.0,
            beats: vec![ObservedBeat {
                time_s: 1.0,
                confidence: 0.8,
                downbeat_confidence: 0.1,
            }],
            beat_candidates: vec![BeatCandidate {
                time_s: 1.0,
                confidence: 0.8,
                downbeat_confidence: 0.1,
            }],
            activations: None,
            activity: vec![],
            onsets: vec![],
            harmonic_changes: vec![],
            source: ModelInfo {
                backend: "authored".into(),
                model: "authored".into(),
                version: None,
                frame_rate_hz: Some(50.0),
            },
        }
    }

    fn frozen(o: &RhythmObservations) -> FrozenObservations {
        FrozenObservations {
            duration_s: o.duration_s,
            beats: o.beats.clone(),
            beat_candidates: o.beat_candidates.clone(),
            source: o.source.clone(),
        }
    }

    #[test]
    fn unchanged_events_confidence_and_metadata_are_exact() {
        let o = observations();
        assert_eq!(compare(&o, &frozen(&o))["exact"], true);
    }

    #[test]
    fn confidence_only_change_is_not_exact_replay() {
        let mut o = observations();
        let f = frozen(&o);
        o.beats[0].confidence += 1e-8;
        let result = compare(&o, &f);
        assert_eq!(result["beat_timestamps_exact"], true);
        assert_eq!(result["exact"], false);
    }

    #[test]
    fn candidate_change_or_missing_event_is_rejected() {
        let mut o = observations();
        let f = frozen(&o);
        o.beat_candidates[0].time_s += 0.02;
        assert_eq!(compare(&o, &f)["exact"], false);
        o.beats.clear();
        assert_eq!(compare(&o, &f)["beat_timestamps_exact"], false);
    }

    #[test]
    fn metadata_and_duration_are_not_relabelled() {
        let mut o = observations();
        let f = frozen(&o);
        o.source.version = Some("changed".into());
        assert_eq!(compare(&o, &f)["exact"], false);
        o.duration_s = 1.0;
        assert_eq!(compare(&o, &f)["duration_bits_exact"], false);
    }

    #[test]
    fn heads_must_be_complete_finite_and_paired() {
        assert!(validate_heads(&[0.0, 1.0], &[0.0, -1.0]).is_ok());
        assert!(validate_heads(&[], &[]).is_err());
        assert!(validate_heads(&[0.0], &[]).is_err());
        assert!(validate_heads(&[f32::NAN], &[0.0]).is_err());
    }

    #[test]
    fn partial_or_mismatched_cohort_never_passes() {
        let good = json!({"replay": {"exact": true}});
        let bad = json!({"replay": {"exact": false}});
        assert!(complete(1, std::slice::from_ref(&good)));
        assert!(!complete(2, &[good, bad]));
        assert!(!complete(1, &[]));
        assert!(!complete(0, &[]));
    }

    #[test]
    fn frozen_suite_and_roles_are_checked_before_assets() {
        let bytes = include_bytes!("../../../evaluation/suites/artbeat-v1.json");
        let mut suite: EvaluationSuite = serde_json::from_slice(bytes).unwrap();
        assert!(calibration_lock(&suite, bytes).is_ok());
        assert!(calibration_lock(&suite, b"changed").is_err());
        for purpose in [SuitePurpose::Holdout, SuitePurpose::Regression] {
            suite.purpose = purpose;
            assert!(calibration_lock(&suite, bytes).is_err());
        }
    }

    #[test]
    fn output_inside_worktree_is_rejected_without_creation() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("new-private-dense-evidence");
        assert!(private_destination(&path).is_err());
    }

    #[test]
    fn pcm_hash_preserves_float_bits() {
        assert_ne!(pcm_sha(&[0.0]), pcm_sha(&[-0.0]));
    }
}
