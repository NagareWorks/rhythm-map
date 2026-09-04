//! Locked, read-only historical observation replay; dense evidence stays private.
use anyhow::{Result, ensure};
use clap::Parser;
use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

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
    output: PathBuf,
    #[arg(long)]
    private_evidence: PathBuf,
}

fn private_destination(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("parent required"))?
        .canonicalize()?;
    ensure!(
        parent.ancestors().all(|p| !p.join(".git").exists()),
        "private evidence must be outside Git"
    );
    ensure!(!path.exists(), "private output must be new");
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    private_destination(&args.private_evidence)?;
    ensure!(
        !args.output.exists() && args.output != args.private_evidence,
        "summary must be a distinct new file"
    );
    let (summary, evidence) = rhythm_map_eval::replay_rubato_cache(
        &args.suite,
        &args.model_pack,
        &args.model_dir,
        &args.audio_dir,
        &args.observation_cache,
        &args.baseline,
    )?;
    let mut private = BufWriter::new(fs::File::create_new(&args.private_evidence)?);
    serde_json::to_writer(&mut private, &evidence)?;
    writeln!(private)?;
    private.flush()?;
    let mut public = BufWriter::new(fs::File::create_new(&args.output)?);
    serde_json::to_writer_pretty(&mut public, &summary)?;
    writeln!(public)?;
    public.flush()?;
    eprintln!("25 exact historical replays; no inference, cache write or relabeling");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_private_evidence_inside_repository() {
        let output = Path::new(env!("CARGO_MANIFEST_DIR")).join("not-written.private.json");
        assert!(private_destination(&output).is_err());
        assert!(!output.exists());
    }
}
