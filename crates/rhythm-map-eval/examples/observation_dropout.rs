//! Controlled estimator audit; no model or audio access and no product options.
use anyhow::{Result, ensure};
use clap::Parser;
use rhythm_map_eval::audit_observation_dropout;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refusing to overwrite a report");
    let report = audit_observation_dropout(&args.suite)?;
    let mut writer = BufWriter::new(File::create_new(args.output)?);
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}
