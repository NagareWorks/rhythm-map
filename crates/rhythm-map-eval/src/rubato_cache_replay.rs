//! Scoped historical-cache evidence, never a production contract fallback.
use std::{fs, path::Path};

use anyhow::{Context, Result, ensure};
use rhythm_map_beat_this::{OBSERVATION_CONTRACT, decode_audio};
use rhythm_map_core::{BackendError, Engine, RhythmObservationBackend, RhythmObservations};
use rhythm_map_models::verify_model_pack;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    AttributionCase, BottleneckEvaluation, CaseInput, EvaluationSuite, ExternalAudioResolver,
    SuitePurpose, evaluate_analysis,
    metrics::match_event_pairs,
    observation_cache::{DecodedAudioIdentity, ObservationCache, ObservationCacheKey},
    runner::{load_case_truth, load_suite},
};

const LOCK: &[u8] = include_bytes!("../../../evaluation/parity/rubato-cache-replay-lock-v1.json");
const PROOF: &[u8] = include_bytes!("../../../evaluation/parity/rubato-pcm-equivalence-v1.json");
const SOURCES: &[(&str, &[u8])] = &[
    (
        "crates/rhythm-map-beat-this/src/lib.rs",
        include_bytes!("../../rhythm-map-beat-this/src/lib.rs"),
    ),
    (
        "crates/rhythm-map-beat-this/src/audio.rs",
        include_bytes!("../../rhythm-map-beat-this/src/audio.rs"),
    ),
    (
        "crates/rhythm-map-core/src/engine.rs",
        include_bytes!("../../rhythm-map-core/src/engine.rs"),
    ),
    (
        "crates/rhythm-map-core/src/estimator.rs",
        include_bytes!("../../rhythm-map-core/src/estimator.rs"),
    ),
    (
        "crates/rhythm-map-eval/src/metrics.rs",
        include_bytes!("metrics.rs"),
    ),
    (
        "crates/rhythm-map-eval/src/observation_cache.rs",
        include_bytes!("observation_cache.rs"),
    ),
    ("Cargo.lock", include_bytes!("../../../Cargo.lock")),
];

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn pcm_sha(samples: &[f32]) -> String {
    let mut hash = Sha256::new();
    for sample in samples {
        hash.update(sample.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn validate_suite(suite: &EvaluationSuite, bytes: &[u8], lock: &Value) -> Result<()> {
    ensure!(
        suite.purpose == SuitePurpose::Calibration,
        "calibration only; holdout rejected"
    );
    ensure!(
        suite.id == lock["suite_id"] && sha(bytes) == lock["suite_sha256"],
        "suite identity changed"
    );
    ensure!(
        suite.cases.len() == 25 && lock["cases"].as_array().is_some_and(|c| c.len() == 25),
        "incomplete cohort"
    );
    ensure!(
        suite
            .cases
            .iter()
            .all(|c| c.provenance.commercial_evaluation_allowed),
        "commercial permission required"
    );
    ensure!(
        lock["shipping_observation_contract"] == OBSERVATION_CONTRACT,
        "shipping contract changed"
    );
    ensure!(
        lock["source_observation_contract"] == "beat-this-rten-observations-v1+decode-audio-v1"
            && lock["decoder_policy"] == "upstream-default",
        "only locked historical default cache allowed"
    );
    Ok(())
}

fn frozen_cache(
    root: &Path,
    key: &ObservationCacheKey,
    shape: DecodedAudioIdentity,
    digest: &str,
) -> Result<RhythmObservations> {
    // Same serialized key/address as the existing cache; decoding and validation
    // remain owned by ObservationCache. Do not add a fallback to that cache.
    let address = sha(&serde_json::to_vec(key)?);
    let path = root
        .join("v1")
        .join(&address[..2])
        .join(format!("{address}.json"));
    let before = fs::read(&path).context("missing frozen cache; inference is forbidden")?;
    ensure!(sha(&before) == digest, "cache payload changed");
    let raw = ObservationCache::new(root)?
        .load(key, shape)?
        .context("missing cache; no inference fallback")?;
    ensure!(fs::read(&path)? == before, "cache changed during read");
    Ok(raw)
}

fn validate_raw(raw: &RhythmObservations, original: &AttributionCase) -> Result<()> {
    ensure!(
        raw.beats == original.observations.raw_beats,
        "raw events/confidence changed"
    );
    ensure!(
        raw.source == original.observations.source,
        "raw model metadata changed"
    );
    ensure!(
        raw.beat_candidates.len() == original.observations.candidate_beat_count,
        "candidate count changed"
    );
    ensure!(
        raw.activity.is_empty() && raw.onsets.is_empty() && raw.harmonic_changes.is_empty(),
        "require original raw cache; acoustic evidence must be regenerated from PCM"
    );
    Ok(())
}

struct Replay(Option<RhythmObservations>);
impl RhythmObservationBackend for Replay {
    fn observe_mono(&mut self, _: &[f32], _: u32) -> Result<RhythmObservations, BackendError> {
        self.0
            .take()
            .ok_or_else(|| BackendError::new("single-use replay consumed"))
    }
}

/// Replay only the locked RUBATO v1 cache, returning (public summary, private evidence).
///
/// No neural model is loaded and no cache entry is written or relabeled. The
/// second result contains dense observations and truth: keep it outside Git.
///
/// # Errors
/// Rejects changed sources, non-calibration suites, missing/changed caches,
/// model/PCM/truth mismatches, and any non-exact historical score replay.
// Keep the identity gates and their ordered, fail-closed replay together.
#[allow(clippy::too_many_lines)]
pub fn replay_rubato_cache(
    suite_path: &Path,
    model_manifest: &Path,
    model_root: &Path,
    audio_root: &Path,
    cache_root: &Path,
    baseline_path: &Path,
) -> Result<(Value, Value)> {
    let lock: Value = serde_json::from_slice(LOCK)?;
    let (suite, root) = load_suite(suite_path)?;
    validate_suite(&suite, &fs::read(suite_path)?, &lock)?;
    for (path, bytes) in SOURCES {
        ensure!(
            sha(bytes) == lock["sources"][path],
            "source changed: {path}"
        );
    }
    ensure!(sha(PROOF) == lock["pcm_proof_sha256"], "PCM proof changed");
    let proof: Value = serde_json::from_slice(PROOF)?;
    ensure!(
        proof["suite_sha256"] == lock["suite_sha256"] && proof["bit_identical_cases"] == 25,
        "incomplete PCM proof"
    );
    let baseline_bytes = fs::read(baseline_path)?;
    ensure!(
        sha(&baseline_bytes) == lock["baseline_sha256"],
        "historical report changed"
    );
    let baseline: BottleneckEvaluation = serde_json::from_slice(&baseline_bytes)?;
    ensure!(
        baseline.suite_id == suite.id
            && baseline.suite_purpose == SuitePurpose::Calibration
            && baseline.cases.len() == 25
            && baseline.observation_cache_contract.as_deref()
                == lock["source_observation_contract"].as_str(),
        "historical cohort/contract mismatch"
    );
    let pack = verify_model_pack(model_manifest, model_root)?;
    ensure!(
        pack.manifest_sha256() == lock["model_manifest_sha256"]
            && pack.manifest_sha256() == baseline.model_pack.manifest_sha256,
        "model assets changed"
    );
    let resolver = ExternalAudioResolver::new(audio_root)?;
    let mut cases = Vec::new();
    let mut private_cases = Vec::new();
    for (index, case) in suite.cases.iter().enumerate() {
        eprintln!(
            "RUBATO cache replay {}/25: {} (no inference)",
            index + 1,
            case.id
        );
        let pinned = &lock["cases"][index];
        let input_proof = &proof["cases"][index];
        let previous = &baseline.cases[index];
        let CaseInput::External {
            audio,
            truth: truth_file,
        } = &case.input
        else {
            anyhow::bail!("external calibration input required");
        };
        ensure!(
            pinned["id"] == case.id && input_proof["id"] == case.id && previous.id == case.id,
            "case order changed"
        );
        ensure!(
            pinned["audio_sha256"] == audio.sha256 && input_proof["audio_sha256"] == audio.sha256,
            "audio reference changed"
        );
        ensure!(
            sha(&fs::read(root.join(truth_file))?) == pinned["truth_sha256"],
            "truth changed"
        );
        let truth = load_case_truth(case, &root)?;
        ensure!(!truth.beats.is_empty(), "beat truth required");
        let decoded = decode_audio(resolver.resolve(audio)?)?;
        let pcm_digest = pcm_sha(&decoded.samples);
        ensure!(
            decoded.sample_rate == 22_050
                && input_proof["comparison"]["bit_identical"] == true
                && decoded.samples.len() == input_proof["comparison"]["shipping_sample_count"]
                && pcm_digest == input_proof["comparison"]["shipping_pcm_sha256"],
            "full PCM proof mismatch"
        );
        let key = ObservationCacheKey::new(
            audio.sha256.clone(),
            pack.manifest_sha256().into(),
            lock["source_observation_contract"]
                .as_str()
                .context("missing source contract")?
                .into(),
            "upstream-default".into(),
        )?;
        let raw = frozen_cache(
            cache_root,
            &key,
            DecodedAudioIdentity::new(decoded.sample_rate, decoded.samples.len())?,
            pinned["cache_entry_sha256"]
                .as_str()
                .context("missing cache digest")?,
        )?;
        validate_raw(&raw, previous)?;
        #[allow(clippy::cast_precision_loss)]
        let duration = decoded.samples.len() as f64 / f64::from(decoded.sample_rate);
        ensure!(
            raw.duration_s.to_bits() == duration.to_bits(),
            "cached duration differs from full PCM"
        );
        let mut engine = Engine::new(Replay(Some(raw)));
        let observations = engine.observe_pcm(&decoded.samples, decoded.sample_rate, 1)?;
        let analysis = engine.analyze_observations(&observations)?;
        let thresholds = case.thresholds.as_ref().unwrap_or(&suite.thresholds);
        let scored = evaluate_analysis(&case.id, &analysis, &truth, thresholds);
        ensure!(
            scored == previous.end_to_end,
            "historical score changed: {}",
            case.id
        );
        let raw_times = observations
            .beats
            .iter()
            .map(|b| b.time_s)
            .collect::<Vec<_>>();
        let truth_times = truth.beats.iter().map(|b| b.time_s).collect::<Vec<_>>();
        let pairs = match_event_pairs(
            &raw_times,
            &truth_times,
            thresholds.beat_tolerance_ms / 1000.0,
        );
        cases.push(json!({"id": case.id, "audio_sha256": audio.sha256,
            "truth_sha256": pinned["truth_sha256"], "cache_entry_sha256": pinned["cache_entry_sha256"],
            "pcm_sha256": pcm_digest, "sample_count": decoded.samples.len(), "sample_rate": decoded.sample_rate,
            "raw_events_exact": true, "score_replay_exact": true, "source_metadata_exact": true,
            "raw_beat_count": observations.beats.len(), "candidate_count": observations.beat_candidates.len(),
            "activity_point_count": observations.activity.len(), "onset_point_count": observations.onsets.len(),
            "harmonic_point_count": observations.harmonic_changes.len(), "selected_score": scored}));
        private_cases.push(json!({"id": case.id, "audio_sha256": audio.sha256,
            "truth_sha256": pinned["truth_sha256"], "pcm_sha256": pcm_digest,
            "sample_count": decoded.samples.len(), "sample_rate": decoded.sample_rate,
            "truth_times_s": truth_times, "raw_truth_pairs": pairs,
            "beat_tolerance_s": thresholds.beat_tolerance_ms / 1000.0,
            "observations": observations, "selected_score": scored, "score_replay_exact": true}));
    }
    let summary = json!({"schema_version": 1, "purpose": "calibration_read_only_v1_cache_replay_summary",
        "lock_sha256": sha(LOCK), "exporter_sha256": sha(include_bytes!("rubato_cache_replay.rs")),
        "suite_sha256": lock["suite_sha256"], "pcm_proof_sha256": lock["pcm_proof_sha256"],
        "baseline_sha256": lock["baseline_sha256"], "model_manifest_sha256": pack.manifest_sha256(),
        "source_observation_contract": lock["source_observation_contract"],
        "shipping_observation_contract": OBSERVATION_CONTRACT, "sources": lock["sources"],
        "compatibility_scope": "locked historical observations replayed with verified identical PCM; not fresh v2 inference",
        "case_count": cases.len(), "cache_hits": cases.len(), "neural_inferences": 0,
        "cache_writes": 0, "cache_relabeling": false, "production_fallback": false, "cases": cases});
    let evidence = json!({"schema_version": 1, "purpose": "private_rubato_cache_replay_evidence",
        "summary": summary, "cases": private_cases});
    Ok((summary, evidence))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_changed_cohort_contract_and_sources_before_io() {
        let bytes = include_bytes!("../../../evaluation/suites/rubato-calibration-v1.json");
        let lock: Value = serde_json::from_slice(LOCK).unwrap();
        let mut suite: EvaluationSuite = serde_json::from_slice(bytes).unwrap();
        validate_suite(&suite, bytes, &lock).unwrap();
        suite.purpose = SuitePurpose::Holdout;
        assert!(validate_suite(&suite, bytes, &lock).is_err());
        suite.purpose = SuitePurpose::Calibration;
        assert!(validate_suite(&suite, b"changed", &lock).is_err());
        let mut changed = lock.clone();
        changed["source_observation_contract"] = json!(OBSERVATION_CONTRACT);
        assert!(validate_suite(&suite, bytes, &changed).is_err());
        for (path, bytes) in SOURCES {
            assert_eq!(sha(bytes), lock["sources"][path]);
        }
        assert_eq!(sha(PROOF), lock["pcm_proof_sha256"]);
    }

    #[test]
    fn missing_cache_does_not_create_directory_or_run_backend() {
        let root =
            std::env::temp_dir().join(format!("rhythm-map-missing-replay-{}", std::process::id()));
        assert!(!root.exists());
        let key = ObservationCacheKey::new(
            "a".repeat(64),
            "b".repeat(64),
            "old".into(),
            "default".into(),
        )
        .unwrap();
        assert!(
            frozen_cache(
                &root,
                &key,
                DecodedAudioIdentity::new(22050, 1).unwrap(),
                &"c".repeat(64)
            )
            .is_err()
        );
        assert!(!root.exists());
    }

    #[test]
    fn cache_identity_shape_schema_and_payload_changes_fail_closed() {
        let root =
            std::env::temp_dir().join(format!("rhythm-map-pinned-replay-{}", std::process::id()));
        assert!(!root.exists());
        let key = ObservationCacheKey::new(
            "a".repeat(64),
            "b".repeat(64),
            "old".into(),
            "default".into(),
        )
        .unwrap();
        let shape = DecodedAudioIdentity::new(22050, 22050).unwrap();
        let raw: RhythmObservations = serde_json::from_value(json!({
            "duration_s": 1.0, "beats": [], "beat_candidates": [], "activity": [],
            "onsets": [], "harmonic_changes": [], "activations": null,
            "source": {"backend": "fixture", "model": "fixture", "version": null, "frame_rate_hz": 50.0}
        })).unwrap();
        ObservationCache::new(&root)
            .unwrap()
            .store(&key, shape, &raw)
            .unwrap();
        let address = sha(&serde_json::to_vec(&key).unwrap());
        let path = root
            .join("v1")
            .join(&address[..2])
            .join(format!("{address}.json"));
        let bytes = fs::read(&path).unwrap();
        let digest = sha(&bytes);
        assert_eq!(frozen_cache(&root, &key, shape, &digest).unwrap(), raw);
        assert!(
            frozen_cache(
                &root,
                &key,
                DecodedAudioIdentity::new(22050, 1).unwrap(),
                &digest
            )
            .is_err()
        );
        assert!(frozen_cache(&root, &key, shape, &"0".repeat(64)).is_err());
        let other = ObservationCacheKey::new(
            "a".repeat(64),
            "b".repeat(64),
            "new".into(),
            "default".into(),
        )
        .unwrap();
        assert!(frozen_cache(&root, &other, shape, &digest).is_err());
        for field in ["schema_version", "key", "observations"] {
            let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
            match field {
                "schema_version" => changed[field] = json!(99),
                "key" => changed[field]["decoder_policy"] = json!("changed"),
                _ => changed[field]["duration_s"] = json!(2.0),
            }
            let changed_bytes = serde_json::to_vec(&changed).unwrap();
            fs::write(&path, &changed_bytes).unwrap();
            assert!(frozen_cache(&root, &key, shape, &digest).is_err());
            if field != "observations" {
                assert!(frozen_cache(&root, &key, shape, &sha(&changed_bytes)).is_err());
            }
        }
        fs::remove_dir_all(root).unwrap();
    }
}
