//! Analyze an audio file with a verified Beat This model pack.

use std::{fs, path::PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use rhythm_map_beat_this::{BeatThisBackend, decode_audio};
use rhythm_map_core::Engine;
use rhythm_map_models::{ModelArtifactRole, verify_model_pack};

#[derive(Debug, Parser)]
#[command(about = "Analyze one audio file into Rhythm Map JSON")]
struct Args {
    /// Input WAV, MP3, FLAC, or OGG file.
    input: PathBuf,
    /// Checked-in model-pack manifest.
    #[arg(long, default_value = "models/beat-this-full-v1.json")]
    model_pack: PathBuf,
    /// Directory containing the model artifacts named by the manifest.
    #[arg(long)]
    model_dir: PathBuf,
    /// Write JSON to this path instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let pack = verify_model_pack(&args.model_pack, &args.model_dir)
        .context("model-pack verification failed")?;
    let mel_model = pack
        .path_for(ModelArtifactRole::MelFrontend)
        .ok_or_else(|| anyhow!("verified pack has no mel_frontend artifact"))?;
    let beat_model = pack
        .path_for(ModelArtifactRole::BeatModel)
        .ok_or_else(|| anyhow!("verified pack has no beat_model artifact"))?;

    let model_name = pack.manifest().id.clone();
    let model_version = Some(format!("manifest-sha256:{}", pack.manifest_sha256()));
    let audio = decode_audio(&args.input)?;
    let backend = BeatThisBackend::load_with_model_identity(
        mel_model,
        beat_model,
        model_name,
        model_version,
    )?;
    let mut engine = Engine::new(backend);
    let analysis = engine
        .analyze_pcm(&audio.samples, audio.sample_rate, 1)
        .context("timing analysis failed")?;
    let json = serde_json::to_string_pretty(&analysis)?;

    if let Some(path) = args.output {
        fs::write(&path, json).with_context(|| format!("failed to write {}", path.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}
