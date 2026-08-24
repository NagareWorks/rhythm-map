//! Command-line entry point for end-to-end audio timing analysis.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use rhythm_map_beat_this::{BeatThisBackend, decode_audio};
use rhythm_map_core::Engine;

#[derive(Debug, Parser)]
#[command(version, about = "Analyze audio into a confidence-aware tempo map")]
struct Args {
    /// Input WAV, MP3, FLAC, or OGG file.
    input: PathBuf,
    /// Beat This log-mel frontend ONNX model.
    #[arg(long)]
    mel_model: PathBuf,
    /// Beat This beat/downbeat ONNX model.
    #[arg(long)]
    beat_model: PathBuf,
    /// Write JSON to this path instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Emit compact rather than pretty JSON.
    #[arg(long)]
    compact: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let audio = decode_audio(&args.input)?;
    let backend = BeatThisBackend::load(&args.mel_model, &args.beat_model)?;
    let mut engine = Engine::new(backend);
    let analysis = engine
        .analyze_pcm(&audio.samples, audio.sample_rate, 1)
        .context("timing analysis failed")?;
    let json = if args.compact {
        serde_json::to_string(&analysis)?
    } else {
        serde_json::to_string_pretty(&analysis)?
    };

    if let Some(path) = args.output {
        fs::write(&path, json).with_context(|| format!("failed to write {}", path.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_cli_has_no_musical_strategy_switch() {
        let result = Args::try_parse_from([
            "rhythm-map",
            "song.wav",
            "--mel-model",
            "mel.onnx",
            "--beat-model",
            "beat.onnx",
            "--decoder-policy",
            "viterbi",
        ]);

        assert!(result.is_err());
    }
}
