//! Cache-only calibration evidence; no inference, cache writes, or recovery policy.

use std::{fs, path::Path};

use anyhow::{Context, Result, ensure};
use rhythm_map_beat_this::{OBSERVATION_CONTRACT, decode_audio};
use rhythm_map_core::{BackendError, Engine, RhythmObservationBackend, RhythmObservations};
use rhythm_map_models::verify_model_pack;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    AcceptanceThresholds, BottleneckEvaluation, CaseEvaluation, CaseInput, EvaluationSuite,
    ExternalAudioResolver, GeneratedTruth, SuitePurpose, evaluate_analysis,
    metrics::match_event_pairs,
    observation_cache::{DecodedAudioIdentity, ObservationCache, ObservationCacheKey},
    runner::{load_case_truth, load_suite},
};

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

struct Replay(Option<RhythmObservations>);

impl RhythmObservationBackend for Replay {
    fn observe_mono(&mut self, _: &[f32], _: u32) -> Result<RhythmObservations, BackendError> {
        self.0
            .take()
            .ok_or_else(|| BackendError::new("replay consumed"))
    }
}

fn evidence(
    raw: RhythmObservations,
    pcm: &[f32],
    rate: u32,
    truth: &GeneratedTruth,
    thresholds: &AcceptanceThresholds,
    expected: &CaseEvaluation,
) -> Result<Value> {
    ensure!(
        !pcm.is_empty() && pcm.iter().all(|x| x.is_finite()),
        "invalid PCM"
    );
    let mut engine = Engine::new(Replay(Some(raw)));
    let observations = engine.observe_pcm(pcm, rate, 1)?;
    let analysis = engine.analyze_observations(&observations)?;
    let scored = evaluate_analysis(&truth.id, &analysis, truth, thresholds);
    ensure!(&scored == expected, "frozen score changed: {}", truth.id);
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
    Ok(
        json!({"id": truth.id, "truth_times_s": truth_times, "raw_truth_pairs": pairs,
        "beat_tolerance_s": thresholds.beat_tolerance_ms / 1000.0,
        "observations": observations, "selected_score": scored, "score_replay_exact": true,
        "pcm_sha256": pcm_sha(pcm), "sample_count": pcm.len(), "sample_rate": rate}),
    )
}

fn locked_suite(suite: &EvaluationSuite, bytes: &[u8], lock: &Value) -> Result<()> {
    ensure!(
        suite.purpose == SuitePurpose::Calibration,
        "rejects holdout/regression before I/O"
    );
    ensure!(
        suite.id == lock["suite_id"] && sha(bytes) == lock["suite_sha256"],
        "only the frozen calibration suite is permitted"
    );
    ensure!(suite.cases.len() == 15, "incomplete suite");
    ensure!(
        suite
            .cases
            .iter()
            .all(|c| c.provenance.commercial_evaluation_allowed),
        "commercial evaluation permission required"
    );
    Ok(())
}

fn probe_evidence(
    path: &Path,
    audit: &Value,
    calibration: &Value,
    truth: &GeneratedTruth,
    thresholds: &AcceptanceThresholds,
) -> Result<Value> {
    ensure!(
        sha(&fs::read(path)?) == audit["after_trace_sha256"],
        "probe trace identity changed"
    );
    let trace: Value = serde_json::from_slice(&fs::read(path)?)?;
    let pcm: Vec<f32> = serde_json::from_value(trace["mono_samples"].clone())?;
    ensure!(
        trace["purpose"] == "calibration_parity_private" && trace["case_id"] == truth.id,
        "invalid probe role/id"
    );
    ensure!(
        trace["sample_rate"] == 22050 && trace["decoded_sample_count"] == pcm.len(),
        "probe must cover the complete recording"
    );
    ensure!(
        pcm_sha(&pcm) == calibration["candidate_pcm_sha256"],
        "probe PCM changed"
    );
    let observations: RhythmObservations = serde_json::from_value(trace["observations"].clone())?;
    let expected = serde_json::from_value(calibration["candidate"].clone())?;
    let mut value = evidence(observations, &pcm, 22050, truth, thresholds, &expected)?;
    value["source_trace_sha256"] = audit["after_trace_sha256"].clone();
    value["observation_contract"] = trace["observation_contract"].clone();
    Ok(value)
}

fn frozen_inputs(
    lock: &Value,
    baseline_path: &Path,
) -> Result<(Value, Value, BottleneckEvaluation)> {
    let calibration_bytes =
        include_bytes!("../../../evaluation/parity/reference-resampler-calibration-v1.json");
    let audit_bytes =
        include_bytes!("../../../evaluation/parity/resampler-regression-event-v1.json");
    ensure!(
        sha(calibration_bytes) == lock["calibration_report_sha256"]
            && sha(audit_bytes) == lock["event_audit_sha256"],
        "historical evidence changed"
    );
    let calibration: Value = serde_json::from_slice(calibration_bytes)?;
    let calibration = calibration["suites"][0].clone();
    let audit: Value = serde_json::from_slice(audit_bytes)?;
    let bytes = fs::read(baseline_path)?;
    ensure!(
        sha(&bytes) == calibration["baseline_report_sha256"],
        "baseline identity changed"
    );
    Ok((calibration, audit, serde_json::from_slice(&bytes)?))
}

/// Export the locked 15-case cache replay and one diagnostic probe for evidence analysis.
///
/// The result contains dense observations and must be saved outside Git. This
/// path loads no neural model and never creates or changes a cache entry.
///
/// # Errors
///
/// Rejects changed identities, missing caches, tempo-only truth, non-calibration
/// input, or any failure to reproduce the frozen selected scores.
pub fn export_cached_candidate_evidence(
    suite_path: &Path,
    model_pack: &Path,
    model_root: &Path,
    audio_directory: &Path,
    cache_directory: &Path,
    baseline_path: &Path,
    probe_trace: &Path,
) -> Result<Value> {
    let lock_bytes = include_bytes!("../../../evaluation/parity/candidate-evidence-lock-v1.json");
    let lock: Value = serde_json::from_slice(lock_bytes)?;
    let (suite, root) = load_suite(suite_path)?;
    locked_suite(&suite, &fs::read(suite_path)?, &lock)?;
    let (calibration, audit, baseline) = frozen_inputs(&lock, baseline_path)?;
    let pack = verify_model_pack(model_pack, model_root)?;
    ensure!(
        pack.manifest_sha256() == baseline.model_pack.manifest_sha256
            && baseline.observation_cache_contract.as_deref() == Some(OBSERVATION_CONTRACT),
        "model or observation contract changed"
    );
    let cache = ObservationCache::new(cache_directory)?;
    let resolver = ExternalAudioResolver::new(audio_directory)?;
    let mut cases = Vec::new();
    let mut probe = None;
    for (index, case) in suite.cases.iter().enumerate() {
        eprintln!(
            "candidate evidence {}/{}: {} (cache only)",
            index + 1,
            suite.cases.len(),
            case.id
        );
        let truth = load_case_truth(case, &root)?;
        ensure!(
            !truth.beats.is_empty(),
            "tempo-only truth cannot label candidate evidence"
        );
        let CaseInput::External {
            audio,
            truth: truth_file,
        } = &case.input
        else {
            anyhow::bail!("expected external audio");
        };
        let previous = &calibration["cases"][index];
        ensure!(
            previous["id"] == case.id
                && previous["truth_sha256"] == sha(&fs::read(root.join(truth_file))?),
            "truth changed"
        );
        let decoded = decode_audio(resolver.resolve(audio)?)?;
        ensure!(
            pcm_sha(&decoded.samples) == previous["current_pcm_sha256"],
            "shipping PCM changed"
        );
        let key = ObservationCacheKey::new(
            audio.sha256.clone(),
            pack.manifest_sha256().to_owned(),
            OBSERVATION_CONTRACT.into(),
            "upstream-default".into(),
        )?;
        let raw = cache
            .load(
                &key,
                DecodedAudioIdentity::new(decoded.sample_rate, decoded.samples.len())?,
            )?
            .context("missing verified cache; this command never runs inference")?;
        let original = &baseline.cases[index];
        ensure!(original.id == case.id, "baseline order changed");
        let raw_events: Value = serde_json::to_value(&raw.beats)?;
        ensure!(
            raw_events == serde_json::to_value(&original.observations.raw_beats)?,
            "cached raw events changed"
        );
        let thresholds = case.thresholds.as_ref().unwrap_or(&suite.thresholds);
        let mut value = evidence(
            raw,
            &decoded.samples,
            decoded.sample_rate,
            &truth,
            thresholds,
            &original.end_to_end,
        )?;
        value["audio_sha256"] = json!(audio.sha256);
        value["truth_sha256"] = previous["truth_sha256"].clone();
        value["tags"] = json!(case.tags);
        cases.push(value);
        if case.id == lock["probe_case_id"] {
            probe = Some(probe_evidence(
                probe_trace,
                &audit,
                previous,
                &truth,
                thresholds,
            )?);
        }
    }
    ensure!(probe.is_some(), "missing fixed probe");
    Ok(
        json!({"schema_version": 1, "purpose": "private_calibration_candidate_evidence",
        "lock_sha256": sha(lock_bytes), "suite_sha256": sha(&fs::read(suite_path)?),
        "source_sha256": sha(include_bytes!("candidate_evidence.rs")),
        "cache_source_sha256": sha(include_bytes!("observation_cache.rs")),
        "engine_source_sha256": sha(include_bytes!("../../rhythm-map-core/src/engine.rs")),
        "estimator_source_sha256": sha(include_bytes!("../../rhythm-map-core/src/estimator.rs")),
        "model_manifest_sha256": pack.manifest_sha256(), "observation_contract": OBSERVATION_CONTRACT,
        "cache_hits": cases.len(), "neural_inferences": 0, "cache_writes": 0,
        "cases": cases, "probe": probe}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_holdout_and_changed_suite_before_cache_or_audio_io() {
        let bytes = include_bytes!("../../../evaluation/suites/artbeat-v1.json");
        let lock = serde_json::from_slice(include_bytes!(
            "../../../evaluation/parity/candidate-evidence-lock-v1.json"
        ))
        .unwrap();
        let mut suite: EvaluationSuite = serde_json::from_slice(bytes).unwrap();
        assert!(locked_suite(&suite, bytes, &lock).is_ok());
        suite.purpose = SuitePurpose::Holdout;
        assert!(locked_suite(&suite, bytes, &lock).is_err());
        suite.purpose = SuitePurpose::Calibration;
        assert!(locked_suite(&suite, b"changed", &lock).is_err());
    }
}
