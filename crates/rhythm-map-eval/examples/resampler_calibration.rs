//! Full, paired calibration for the one frozen resampler candidate.
//! No product strategy, holdout access, or training; candidate results stay outside the shipping cache.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rhythm_map_beat_this::{
    BeatThisBackend, OBSERVATION_CONTRACT,
    preprocessing_diagnostics::{decode_native, prepare_mono},
};
use rhythm_map_core::{Engine, TempoMapEstimator};
use rhythm_map_eval::{
    BackendEvaluationOptions, BottleneckEvaluation, CaseInput, EvaluationSuite,
    ExternalAudioResolver, GeneratedTruth, SuitePurpose, evaluate_analysis,
    evaluate_backend_suite_with_options,
};
use rhythm_map_models::{ModelArtifactRole, verify_model_pack};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
    time::Instant,
};

#[path = "support/reference_resampler.rs"]
mod reference_resampler;
mod support;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    audio_dir: PathBuf,
    #[arg(long, default_value = "models/beat-this-full-v1.json")]
    model_pack: PathBuf,
    #[arg(long)]
    model_dir: PathBuf,
    /// The pinned shipping-v2 report from the preceding paired calibration.
    #[arg(long)]
    baseline: PathBuf,
    /// Existing shipping observation cache; candidate observations are never stored here.
    #[arg(long)]
    observation_cache: PathBuf,
    /// New directory for per-case progress and the final aggregate report.
    #[arg(long)]
    output_dir: PathBuf,
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

fn write_new(path: &std::path::Path, value: &impl serde::Serialize) -> Result<()> {
    let mut file = File::create_new(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn locked_suite(suite: &EvaluationSuite, bytes: &[u8], baseline_bytes: &[u8]) -> Result<()> {
    suite.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        suite.purpose == SuitePurpose::Calibration,
        "candidate run rejects regression and holdout"
    );
    let expected_suite = match suite.id.as_str() {
        "artbeat-v1" => "21f3d44bacbfe9c50dfbc889990c563d44e406d56558492627402d21e5a7e81b",
        "fsld-tempo-v1" => "f6956821a09b06084a822ff5ab25b129b3898476950138f38be541be2fe65683",
        _ => anyhow::bail!("only the two frozen 15-case calibration suites are allowed"),
    };
    ensure!(
        sha(bytes) == expected_suite && suite.cases.len() == 15,
        "suite changed from the predeclared calibration"
    );
    let lock: Value = serde_json::from_str(include_str!(
        "../../../evaluation/parity/resampling-v2-calibration.json"
    ))?;
    let previous = lock["suites"]
        .as_array()
        .context("invalid baseline lock")?
        .iter()
        .find(|entry| entry["suite_id"] == suite.id)
        .context("missing baseline identity")?;
    ensure!(
        previous["after_report_sha256"] == sha(baseline_bytes),
        "shipping baseline hash mismatch"
    );
    Ok(())
}

fn shipping_replay(args: &Args, baseline: &BottleneckEvaluation) -> Result<BottleneckEvaluation> {
    // Re-run the shipping engine/estimator from verified cached raw observations,
    // and require exact old scores/oracles before comparing any candidate.
    let replay = evaluate_backend_suite_with_options(
        &args.suite,
        &args.model_pack,
        &args.model_dir,
        BackendEvaluationOptions {
            audio_directory: Some(&args.audio_dir),
            observation_cache_directory: Some(&args.observation_cache),
            ..Default::default()
        },
    )?;
    ensure!(
        replay.cases.len() == baseline.cases.len(),
        "baseline replay case count changed"
    );
    for (before, current) in baseline.cases.iter().zip(&replay.cases) {
        ensure!(
            before.id == current.id
                && before.end_to_end == current.end_to_end
                && before.oracle == current.oracle
                && before.audio_sha256 == current.audio_sha256,
            "shipping replay changed: {}",
            before.id
        );
        ensure!(
            current.observation_cache_hit == Some(true),
            "shipping replay expected an existing verified cache entry"
        );
    }
    Ok(replay)
}

struct Input {
    path: PathBuf,
    audio_sha: String,
    truth: GeneratedTruth,
    truth_sha: String,
}

fn calibration_inputs(args: &Args, suite: &EvaluationSuite) -> Result<Vec<Input>> {
    let resolver = ExternalAudioResolver::new(&args.audio_dir)?;
    let root = args.suite.parent().context("missing suite directory")?;
    let mut inputs = Vec::new();
    for case in &suite.cases {
        let audio = support::calibration_audio(suite, &case.id)?;
        let path = resolver.resolve(&audio)?;
        let CaseInput::External { truth, .. } = &case.input else {
            unreachable!()
        };
        let truth_bytes = fs::read(root.join(truth))?;
        let truth: GeneratedTruth = serde_json::from_slice(&truth_bytes)?;
        truth.validate().map_err(anyhow::Error::msg)?;
        ensure!(truth.id == case.id, "truth ID mismatch");
        inputs.push(Input {
            path,
            audio_sha: audio.sha256,
            truth,
            truth_sha: sha(&truth_bytes),
        });
    }
    Ok(inputs)
}

fn score_oracle(
    id: &str,
    truth: &GeneratedTruth,
    thresholds: &rhythm_map_eval::AcceptanceThresholds,
) -> Result<Option<rhythm_map_eval::CaseEvaluation>> {
    if truth.beats.is_empty() {
        return Ok(None);
    }
    Ok(Some(evaluate_analysis(
        id,
        &TempoMapEstimator::default().estimate(&truth.ideal_observations())?,
        truth,
        thresholds,
    )))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output_dir.exists(), "output directory must be new");
    let suite_bytes = fs::read(&args.suite)?;
    let baseline_bytes = fs::read(&args.baseline)?;
    let suite: EvaluationSuite = serde_json::from_slice(&suite_bytes)?;
    locked_suite(&suite, &suite_bytes, &baseline_bytes)?;
    let baseline: BottleneckEvaluation = serde_json::from_slice(&baseline_bytes)?;
    ensure!(
        baseline.observation_cache_contract.as_deref() == Some(OBSERVATION_CONTRACT),
        "baseline must use the current shipping contract"
    );
    let inputs = calibration_inputs(&args, &suite)?;
    let pack = verify_model_pack(&args.model_pack, &args.model_dir)?;
    ensure!(
        pack.manifest_sha256() == baseline.model_pack.manifest_sha256
            && pack.manifest().backend == "beat-this-rten",
        "baseline model mismatch"
    );
    let replay = shipping_replay(&args, &baseline)?;
    let backend = BeatThisBackend::load(
        pack.path_for(ModelArtifactRole::MelFrontend)
            .context("missing mel model")?,
        pack.path_for(ModelArtifactRole::BeatModel)
            .context("missing beat model")?,
    )?;
    let mut engine = Engine::new(backend);
    fs::create_dir(&args.output_dir)?;
    write_new(&args.output_dir.join("shipping-replay.json"), &replay)?;
    let mut results = Vec::new();
    for (
        index,
        (
            case,
            Input {
                path,
                audio_sha,
                truth,
                truth_sha,
            },
        ),
    ) in suite.cases.iter().zip(&inputs).enumerate()
    {
        eprintln!(
            "candidate calibration {}/{}: {}",
            index + 1,
            suite.cases.len(),
            case.id
        );
        let native = decode_native(path)?;
        let started = Instant::now();
        let current_pcm = prepare_mono(&native.samples, native.sample_rate)?;
        let current_resample_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let pcm = reference_resampler::resample(&native.samples, native.sample_rate)?;
        let candidate_resample_ms = started.elapsed().as_secs_f64() * 1000.0;
        ensure!(current_pcm.len() == pcm.len(), "candidate changed duration");
        let started = Instant::now();
        let observations = engine.observe_pcm(&pcm, 22050, 1)?;
        let analysis = engine.analyze_observations(&observations)?;
        let model_and_analysis_ms = started.elapsed().as_secs_f64() * 1000.0;
        let thresholds = case.thresholds.as_ref().unwrap_or(&suite.thresholds);
        let scored = evaluate_analysis(&case.id, &analysis, truth, thresholds);
        let oracle = score_oracle(&case.id, truth, thresholds)?;
        ensure!(oracle == replay.cases[index].oracle, "oracle changed");
        let result = json!({"id": case.id, "audio_sha256": audio_sha, "truth_sha256": truth_sha,
            "source_sample_rate": native.sample_rate, "source_sample_count": native.samples.len(),
            "model_sample_count": pcm.len(), "current_pcm_sha256": pcm_sha(&current_pcm), "candidate_pcm_sha256": pcm_sha(&pcm),
            "baseline": replay.cases[index].end_to_end, "candidate": scored, "oracle_unchanged": true,
            "current_resample_ms": current_resample_ms, "candidate_resample_ms": candidate_resample_ms,
            "model_and_analysis_ms": model_and_analysis_ms, "raw_beat_count": observations.beats.len(),
            "selected_beat_count": analysis.beats.len(), "warnings": analysis.warnings});
        write_new(&args.output_dir.join(format!("{}.json", case.id)), &result)?;
        results.push(result);
    }
    let report = json!({"schema_version": 1, "purpose": "paired_resampler_calibration_not_release_acceptance",
        "suite_id": suite.id, "suite_purpose": suite.purpose, "suite_sha256": sha(&suite_bytes),
        "baseline_report_sha256": sha(&baseline_bytes), "baseline_replay_exact": true,
        "baseline_cache_hits": replay.cases.len(), "candidate_cache_hits": 0,
        "model_manifest_sha256": pack.manifest_sha256(), "shipping_observation_contract": OBSERVATION_CONTRACT,
        "candidate": reference_resampler::ID, "candidate_observation_contract": format!("{OBSERVATION_CONTRACT}+{}", reference_resampler::ID),
        "coefficient_budget_bytes": reference_resampler::COEFFICIENT_BUDGET_BYTES,
        "candidate_source_sha256": sha(include_bytes!("support/reference_resampler.rs")),
        "runner_source_sha256": sha(include_bytes!("resampler_calibration.rs")),
        "adapter_source_sha256": sha(include_bytes!("../../rhythm-map-beat-this/src/lib.rs")),
        "audio_preprocessing_sha256": sha(include_bytes!("../../rhythm-map-beat-this/src/audio.rs")),
        "core_engine_sha256": sha(include_bytes!("../../rhythm-map-core/src/engine.rs")),
        "core_estimator_sha256": sha(include_bytes!("../../rhythm-map-core/src/estimator.rs")),
        "metrics_source_sha256": sha(include_bytes!("../src/metrics.rs")),
        "cases": results, "not_checked": ["holdout", "platform_parity", "stable_performance_benchmark"],
        "timing_note": "sequential per-track wall time including resampler initialization; baseline neural timings are cached and not comparable"});
    write_new(&args.output_dir.join("report.json"), &report)?;
    eprintln!(
        "Complete paired calibration: {} cases; no candidate was promoted",
        suite.cases.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_holdout_and_changed_suite_before_audio_or_model_io() {
        let bytes = include_bytes!("../../../evaluation/suites/artbeat-v1.json");
        let mut suite: EvaluationSuite = serde_json::from_slice(bytes).unwrap();
        assert!(locked_suite(&suite, bytes, b"not the baseline").is_err());
        suite.purpose = SuitePurpose::Holdout;
        assert!(locked_suite(&suite, bytes, b"").is_err());
        suite.purpose = SuitePurpose::Calibration;
        assert!(locked_suite(&suite, b"changed", b"").is_err());
    }
}
