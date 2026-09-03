//! Strict cache-only evidence export; output belongs on a private data drive.

use anyhow::{Result, ensure};
use clap::Parser;
use std::{fs::File, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long, default_value = "models/beat-this-full-v1.json")]
    model_pack: PathBuf,
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    audio_dir: PathBuf,
    #[arg(long)]
    observation_cache: PathBuf,
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    probe_trace: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "output must be new");
    let report = rhythm_map_eval::export_cached_candidate_evidence(
        &args.suite,
        &args.model_pack,
        &args.model_dir,
        &args.audio_dir,
        &args.observation_cache,
        &args.baseline,
        &args.probe_trace,
    )?;
    let mut file = File::create_new(args.output)?;
    serde_json::to_writer(&mut file, &report)?;
    file.write_all(b"\n")?;
    eprintln!("Export complete: 15 cache replays and 1 frozen probe; no inference or cache write");
    Ok(())
}
