//! Command-line entry point for end-to-end audio timing analysis.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use rhythm_map_beat_this::{BeatThisBackend, decode_audio};
use rhythm_map_core::Engine;
use rhythm_map_models::{
    BEAT_THIS_FULL_MANIFEST, ModelArtifactRole, ModelPackCache, ModelPackManifest,
    VerifiedModelPack, verify_model_pack_bytes,
};
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Analyze audio into a confidence-aware tempo map",
    subcommand_negates_reqs = true,
    args_conflicts_with_subcommands = true
)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
    /// Input WAV, MP3, FLAC, or OGG file.
    #[arg(required = true)]
    input: Option<PathBuf>,
    #[command(flatten)]
    pack: PackArgs,
    /// Legacy explicit frontend path; still verified against the selected manifest.
    #[arg(long, requires = "beat_model", conflicts_with_all = ["model_dir", "cache_dir"])]
    mel_model: Option<PathBuf>,
    /// Legacy explicit beat-model path; still verified against the selected manifest.
    #[arg(long, requires = "mel_model", conflicts_with_all = ["model_dir", "cache_dir"])]
    beat_model: Option<PathBuf>,
    /// Write JSON to this path instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Emit compact rather than pretty JSON.
    #[arg(long)]
    compact: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Explicit model acquisition and offline integrity checks.
    Models {
        #[command(subcommand)]
        command: ModelCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Download a pack once, verifying every byte before publishing it to cache.
    Fetch(CacheArgs),
    /// Verify installed or explicitly provided model files without network access.
    Verify(PackArgs),
}

#[derive(Debug, clap::Args)]
struct CacheArgs {
    /// Trusted custom manifest; defaults to the built-in pinned Beat This pack.
    #[arg(long)]
    model_pack: Option<PathBuf>,
    /// Cache root (or `RHYTHM_MAP_CACHE_DIR`; otherwise the OS user cache).
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
struct PackArgs {
    #[command(flatten)]
    cache: CacheArgs,
    /// Existing artifact directory instead of the managed cache; always offline.
    #[arg(long, conflicts_with = "cache_dir")]
    model_dir: Option<PathBuf>,
}

impl CacheArgs {
    fn manifest(&self) -> Result<Vec<u8>> {
        self.model_pack.as_ref().map_or_else(
            || Ok(BEAT_THIS_FULL_MANIFEST.to_vec()),
            |path| fs::read(path).with_context(|| format!("reading manifest {}", path.display())),
        )
    }

    fn cache(&self) -> Result<ModelPackCache> {
        let root = self.cache_dir.clone().map_or_else(default_cache_dir, Ok)?;
        Ok(ModelPackCache::new(root))
    }
}

impl PackArgs {
    fn verify(&self, manifest: &[u8]) -> Result<VerifiedModelPack> {
        if let Some(root) = &self.model_dir {
            return Ok(verify_model_pack_bytes(manifest, root)?);
        }
        self.cache.cache()?.verify(manifest).context(
            "model cache unavailable or invalid; run `rhythm-map models fetch` with the same \
             --cache-dir/--model-pack (or use --model-dir for existing files). \
             A corrupt existing pack is not overwritten; inspect the reported path",
        )
    }
}

fn default_cache_dir() -> Result<PathBuf> {
    fn absolute_env(name: &str) -> Result<Option<PathBuf>> {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(|value| {
                let path = PathBuf::from(value);
                if !path.is_absolute() {
                    bail!("{name} must be an absolute path");
                }
                Ok(path)
            })
            .transpose()
    }
    if let Some(root) = absolute_env("RHYTHM_MAP_CACHE_DIR")? {
        return Ok(root);
    }
    let root = if cfg!(target_os = "windows") {
        absolute_env("LOCALAPPDATA")?.map(|p| p.join("rhythm-map/cache"))
    } else if cfg!(target_os = "macos") {
        absolute_env("HOME")?.map(|p| p.join("Library/Caches/rhythm-map"))
    } else if let Some(root) = absolute_env("XDG_CACHE_HOME")? {
        Some(root.join("rhythm-map"))
    } else {
        absolute_env("HOME")?.map(|p| p.join(".cache/rhythm-map"))
    };
    root.ok_or_else(|| {
        anyhow!("cannot determine user cache; supply --cache-dir or RHYTHM_MAP_CACHE_DIR")
    })
}

fn model_command(command: ModelCommand) -> Result<()> {
    let pack = match command {
        ModelCommand::Fetch(args) => {
            let manifest = args.manifest()?;
            let cache = args.cache()?;
            eprintln!(
                "Acquiring verified model pack in {} (existing valid packs are reused)",
                cache.directory_for(&manifest)?.display()
            );
            cache.fetch(&manifest)?
        }
        ModelCommand::Verify(args) => args.verify(&args.cache.manifest()?)?,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "id": pack.manifest().id,
            "version": pack.manifest().version,
            "manifest_sha256": pack.manifest_sha256(),
            "model_dir": pack.root(),
            "license": pack.manifest().source.license,
        }))?
    );
    Ok(())
}

fn backend(args: &Args) -> Result<BeatThisBackend> {
    let bytes = args.pack.cache.manifest()?;
    let manifest: ModelPackManifest = serde_json::from_slice(&bytes)?;
    manifest.validate()?;
    if manifest.backend != "beat-this-rten" {
        bail!("the analysis CLI requires a beat-this-rten pack");
    }
    let (mel, beat) = if let (Some(mel), Some(beat)) = (&args.mel_model, &args.beat_model) {
        for (role, path) in [
            (ModelArtifactRole::MelFrontend, mel),
            (ModelArtifactRole::BeatModel, beat),
        ] {
            manifest
                .artifacts
                .iter()
                .find(|a| a.role == role)
                .expect("required manifest role")
                .verify_file(path)?;
        }
        (mel.clone(), beat.clone())
    } else {
        let pack = args.pack.verify(&bytes)?;
        (
            pack.path_for(ModelArtifactRole::MelFrontend)
                .expect("required manifest role"),
            pack.path_for(ModelArtifactRole::BeatModel)
                .expect("required manifest role"),
        )
    };
    Ok(BeatThisBackend::load_with_model_identity(
        mel,
        beat,
        manifest.id,
        Some(format!("manifest-sha256:{:x}", Sha256::digest(&bytes))),
    )?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(Command::Models { command }) = args.command {
        return model_command(command);
    }
    let backend = backend(&args)?;
    let audio = decode_audio(args.input.as_ref().expect("required CLI input"))?;
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
    fn cli_shape_is_consistent() {
        use clap::CommandFactory;
        Args::command().debug_assert();
    }

    #[test]
    fn default_analysis_and_explicit_acquisition_need_no_model_paths() {
        assert!(Args::try_parse_from(["rhythm-map", "song.wav"]).is_ok());
        assert!(Args::try_parse_from(["rhythm-map", "models", "fetch"]).is_ok());
        assert!(Args::try_parse_from(["rhythm-map", "models", "verify"]).is_ok());
        assert!(Args::try_parse_from(["rhythm-map"]).is_err());
    }

    #[test]
    fn incomplete_or_conflicting_model_locations_are_rejected() {
        assert!(
            Args::try_parse_from(["rhythm-map", "song.wav", "--mel-model", "mel.onnx"]).is_err()
        );
        assert!(
            Args::try_parse_from([
                "rhythm-map",
                "song.wav",
                "--model-dir",
                "models",
                "--cache-dir",
                "cache"
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from(["rhythm-map", "models", "fetch", "--model-dir", "models"])
                .is_err()
        );
    }

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
