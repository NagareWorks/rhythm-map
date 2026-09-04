//! Read-only calibration replay. Never imports audio, runs a model, or adopts paths.
use anyhow::{Result, ensure};
use clap::Parser;
use rhythm_map_core::{RhythmObservations, analyze_observations};
use rhythm_map_eval::generate_active_region_candidates;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    evidence: PathBuf,
    #[arg(long)]
    evidence_sha256: String,
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    baseline_sha256: String,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize)]
struct Evidence {
    purpose: String,
    cases: Vec<Case>,
    // Deliberately omit the ARTBeat auxiliary probe.
}

#[derive(Deserialize)]
struct Case {
    id: String,
    observations: RhythmObservations,
}

fn read_locked(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes)) == expected,
        "input SHA-256 mismatch"
    );
    Ok(bytes)
}

fn private_destination(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("parent required"))?
        .canonicalize()?;
    ensure!(
        parent.ancestors().all(|p| !p.join(".git").exists()),
        "dense output must be outside Git"
    );
    ensure!(!path.exists(), "output must be new");
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    private_destination(&args.output)?;
    let started = Instant::now();
    let evidence: Evidence =
        serde_json::from_slice(&read_locked(&args.evidence, &args.evidence_sha256)?)?;
    ensure!(
        evidence.purpose == "private_calibration_candidate_evidence"
            || evidence.purpose == "private_rubato_cache_replay_evidence",
        "only calibration evidence is accepted"
    );
    let baseline: Value =
        serde_json::from_slice(&read_locked(&args.baseline, &args.baseline_sha256)?)?;
    let histories = baseline["cases"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("baseline cases missing"))?;
    ensure!(
        histories.len() == evidence.cases.len(),
        "baseline/evidence case count mismatch"
    );
    let mut ids = std::collections::HashSet::new();
    let mut rows = Vec::new();
    for case in evidence.cases {
        ensure!(ids.insert(case.id.clone()), "duplicate evidence case");
        let matching = histories
            .iter()
            .filter(|h| h["id"].as_str() == Some(&case.id))
            .collect::<Vec<_>>();
        ensure!(matching.len() == 1, "baseline case missing or duplicated");
        let history = matching[0]["pulse_hypothesis_coverage"]["hypotheses"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("baseline hypotheses missing"))?;
        let selected = history
            .iter()
            .filter(|h| h["id"] == "selected")
            .collect::<Vec<_>>();
        ensure!(
            selected.len() == 1,
            "baseline primary missing or duplicated"
        );
        let times: Vec<f64> = serde_json::from_value(selected[0]["beat_times_s"].clone())?;
        let primary_started = Instant::now();
        let analysis = analyze_observations(&case.observations)?;
        ensure!(
            analysis.beats.iter().map(|b| b.time_s).collect::<Vec<_>>() == times,
            "historical primary timestamps changed"
        );
        let analysis_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&analysis)?));
        let primary_elapsed_s = primary_started.elapsed().as_secs_f64();
        let generation_started = Instant::now();
        let generated = generate_active_region_candidates(&case.observations, &times)?;
        let generation_elapsed_s = generation_started.elapsed().as_secs_f64();
        eprintln!(
            "{}: {} components, {:.3}s generator",
            case.id,
            generated.proposals.len(),
            generation_elapsed_s
        );
        rows.push(json!({"id":case.id,"generated":generated,
            "primary_analysis_sha256":analysis_sha256,"primary_replay_exact":true,
            "primary_elapsed_s":primary_elapsed_s,"generation_elapsed_s":generation_elapsed_s}));
    }
    let report = json!({"schema_version":1,"purpose":"private_calibration_active_region_rust_replay",
        "evidence_sha256":args.evidence_sha256,"baseline_sha256":args.baseline_sha256,
        "generator_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../src/active_regions.rs"))),
        "estimator_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../../rhythm-map-core/src/estimator.rs"))),
        "inference_run":false,"adoption_enabled":false,"extra_probe_excluded":true,
        "elapsed_s":started.elapsed().as_secs_f64(),"cases":rows});
    let mut writer = BufWriter::new(fs::File::create_new(args.output)?);
    serde_json::to_writer(&mut writer, &report)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_dense_output_in_git() {
        assert!(
            private_destination(&Path::new(env!("CARGO_MANIFEST_DIR")).join("not-written.json"))
                .is_err()
        );
    }
}
