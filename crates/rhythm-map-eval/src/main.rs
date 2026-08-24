//! Command-line orchestration for reproducible Rhythm Map evaluation.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rhythm_map_eval::{
    evaluate_backend_suite, evaluate_backend_suite_with_audio_directory,
    evaluate_backend_suite_with_policies, evaluate_core_suite,
    evaluate_decoder_recoverability_with_audio_directory,
    evaluate_decoder_sweep_with_audio_directory,
    evaluate_named_decoder_policy_with_audio_directory, fetch_public_dataset, import_artbeat_truth,
    inspect_audio_asset, render_suite, score_prediction_directory, standard_decoder_policies,
};
use rhythm_map_models::verify_model_pack;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Reproducible evaluation and fixture tooling for Rhythm Map")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Gate the core estimator with ideal observations from generated or external truth.
    Eval {
        /// Evaluation suite manifest.
        #[arg(long, default_value = "evaluation/suites/generated-v1.json")]
        suite: PathBuf,
        /// Optional JSON report destination.
        #[arg(long)]
        report: Option<PathBuf>,
        /// Emit failures without returning a non-zero exit code.
        #[arg(long)]
        no_fail: bool,
    },
    /// Compare ideal observations with the Beat This end-to-end audio path.
    EvalBackend {
        /// Evaluation suite manifest.
        #[arg(long, default_value = "evaluation/suites/generated-v1.json")]
        suite: PathBuf,
        /// Versioned model-pack manifest.
        #[arg(long, default_value = "models/beat-this-full-v1.json")]
        model_pack: PathBuf,
        /// Directory containing the model files named by the manifest.
        #[arg(long)]
        model_dir: PathBuf,
        /// Directory containing content-addressed external evaluation audio.
        #[arg(long)]
        audio_dir: Option<PathBuf>,
        /// Stable decoder policy ID; omitted for the immutable upstream default.
        #[arg(long, requires = "audio_dir")]
        decoder_policy: Option<String>,
        /// Stable deterministic estimator policy ID; omitted for the shipping default.
        #[arg(long, requires = "audio_dir")]
        estimator_policy: Option<String>,
        /// Optional JSON report destination.
        #[arg(long)]
        report: Option<PathBuf>,
        /// Emit failures without returning a non-zero exit code.
        #[arg(long)]
        no_fail: bool,
    },
    /// Verify model-pack provenance, sizes, and SHA-256 digests.
    ModelVerify {
        /// Versioned model-pack manifest.
        #[arg(long, default_value = "models/beat-this-full-v1.json")]
        model_pack: PathBuf,
        /// Directory containing the model files named by the manifest.
        #[arg(long)]
        model_dir: PathBuf,
    },
    /// Print the immutable digest and decoded shape needed by an external case.
    AudioInspect {
        /// Local audio file to hash and decode; the path is not written to the report.
        #[arg(long)]
        input: PathBuf,
    },
    /// Recover official ARTBeaT beat truth from a locked Ground Truth SVG.
    ArtbeatTruth {
        /// Stable evaluation case identifier.
        #[arg(long)]
        id: String,
        /// Official ARTBeaT sonified-figure SVG containing Ground Truth lines.
        #[arg(long)]
        annotation: PathBuf,
        /// Matching encoded audio, used only for identity and decoded duration.
        #[arg(long)]
        audio: PathBuf,
        /// Generated truth JSON destination.
        #[arg(long)]
        output: PathBuf,
    },
    /// Fetch and verify a public evaluation dataset outside the Git checkout.
    DatasetFetch {
        /// Versioned public-dataset lock manifest.
        #[arg(long, default_value = "evaluation/datasets/artbeat-v1.json")]
        manifest: PathBuf,
        /// Explicit destination directory, normally outside the Git checkout.
        #[arg(long)]
        output: PathBuf,
        /// Also fetch immutable annotation-source artifacts used to audit truth.
        #[arg(long)]
        with_annotations: bool,
    },
    /// Compare Beat This logit peak decoders without repeating model inference.
    DecoderSweep {
        /// Evaluation suite containing independent beat truth.
        #[arg(long, default_value = "evaluation/suites/artbeat-v1.json")]
        suite: PathBuf,
        /// Versioned model-pack manifest.
        #[arg(long, default_value = "models/beat-this-full-v1.json")]
        model_pack: PathBuf,
        /// Directory containing the model files named by the manifest.
        #[arg(long)]
        model_dir: PathBuf,
        /// Directory containing content-addressed external evaluation audio.
        #[arg(long)]
        audio_dir: PathBuf,
        /// Optional JSON report destination.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Evaluate one preselected decoder policy without exposing a policy sweep.
    DecoderEval {
        /// Evaluation suite; this is the only decoder command permitted for holdout suites.
        #[arg(long)]
        suite: PathBuf,
        /// Stable ID from the standard decoder policy registry.
        #[arg(long)]
        policy: String,
        /// Versioned model-pack manifest.
        #[arg(long, default_value = "models/beat-this-full-v1.json")]
        model_pack: PathBuf,
        /// Directory containing the model files named by the manifest.
        #[arg(long)]
        model_dir: PathBuf,
        /// Directory containing content-addressed external evaluation audio.
        #[arg(long)]
        audio_dir: PathBuf,
        /// Optional JSON report destination.
        #[arg(long)]
        report: Option<PathBuf>,
        /// Emit failed beat gates without returning a non-zero exit code.
        #[arg(long)]
        no_fail: bool,
    },
    /// Inspect model evidence around truth beats missed by the upstream decoder.
    DecoderRecoverability {
        /// Evaluation suite containing independent beat truth.
        #[arg(long, default_value = "evaluation/suites/artbeat-v1.json")]
        suite: PathBuf,
        /// Versioned model-pack manifest.
        #[arg(long, default_value = "models/beat-this-full-v1.json")]
        model_pack: PathBuf,
        /// Directory containing the model files named by the manifest.
        #[arg(long)]
        model_dir: PathBuf,
        /// Directory containing content-addressed external evaluation audio.
        #[arg(long)]
        audio_dir: PathBuf,
        /// Optional JSON report destination.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Render deterministic synthetic WAVs and truth JSON for backend evaluation.
    Render {
        /// Evaluation suite manifest.
        #[arg(long, default_value = "evaluation/suites/generated-v1.json")]
        suite: PathBuf,
        /// Explicit output directory, normally outside the Git checkout.
        #[arg(long)]
        output: PathBuf,
    },
    /// Score end-to-end Analysis JSON files produced by any product surface.
    Score {
        /// Evaluation suite manifest.
        #[arg(long, default_value = "evaluation/suites/generated-v1.json")]
        suite: PathBuf,
        /// Directory containing one `<case-id>.json` Analysis per case.
        #[arg(long)]
        predictions: PathBuf,
        /// Optional JSON report destination.
        #[arg(long)]
        report: Option<PathBuf>,
        /// Emit failures without returning a non-zero exit code.
        #[arg(long)]
        no_fail: bool,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Eval {
            suite,
            report,
            no_fail,
        } => emit_report(&evaluate_core_suite(&suite)?, report, no_fail),
        Command::EvalBackend {
            suite,
            model_pack,
            model_dir,
            audio_dir,
            decoder_policy,
            estimator_policy,
            report,
            no_fail,
        } => run_backend_eval(
            &suite,
            &model_pack,
            &model_dir,
            audio_dir.as_deref(),
            decoder_policy.as_deref(),
            estimator_policy.as_deref(),
            report,
            no_fail,
        ),
        Command::ModelVerify {
            model_pack,
            model_dir,
        } => run_model_verify(&model_pack, &model_dir),
        Command::AudioInspect { input } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&inspect_audio_asset(input)?)?
            );
            Ok(())
        }
        Command::ArtbeatTruth {
            id,
            annotation,
            audio,
            output,
        } => {
            let imported = import_artbeat_truth(id, annotation, audio)?;
            write_json_file(&output, &imported.truth)?;
            println!("{}", serde_json::to_string_pretty(&imported)?);
            Ok(())
        }
        Command::DatasetFetch {
            manifest,
            output,
            with_annotations,
        } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&fetch_public_dataset(
                    &manifest,
                    &output,
                    with_annotations,
                )?)?
            );
            Ok(())
        }
        Command::DecoderSweep {
            suite,
            model_pack,
            model_dir,
            audio_dir,
            report,
        } => run_decoder_sweep(&suite, &model_pack, &model_dir, &audio_dir, report),
        Command::DecoderEval {
            suite,
            policy,
            model_pack,
            model_dir,
            audio_dir,
            report,
            no_fail,
        } => run_decoder_eval(
            &suite,
            &policy,
            &model_pack,
            &model_dir,
            &audio_dir,
            report,
            no_fail,
        ),
        Command::DecoderRecoverability {
            suite,
            model_pack,
            model_dir,
            audio_dir,
            report,
        } => run_decoder_recoverability(&suite, &model_pack, &model_dir, &audio_dir, report),
        Command::Render { suite, output } => {
            for path in render_suite(&suite, &output)? {
                println!("{}", path.display());
            }
            Ok(())
        }
        Command::Score {
            suite,
            predictions,
            report,
            no_fail,
        } => emit_report(
            &score_prediction_directory(&suite, &predictions)?,
            report,
            no_fail,
        ),
    }
}

fn run_model_verify(model_pack: &Path, model_dir: &Path) -> Result<()> {
    let verified = verify_model_pack(model_pack, model_dir)?;
    let output = serde_json::json!({
        "schema_version": 1,
        "verified": true,
        "id": verified.manifest().id,
        "version": verified.manifest().version,
        "backend": verified.manifest().backend,
        "manifest_sha256": verified.manifest_sha256(),
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_backend_eval(
    suite: &Path,
    model_pack: &Path,
    model_dir: &Path,
    audio_dir: Option<&Path>,
    decoder_policy: Option<&str>,
    estimator_policy: Option<&str>,
    report: Option<PathBuf>,
    no_fail: bool,
) -> Result<()> {
    let result = if let Some(audio_dir) = audio_dir
        && (decoder_policy.is_some() || estimator_policy.is_some())
    {
        evaluate_backend_suite_with_policies(
            suite,
            model_pack,
            model_dir,
            audio_dir,
            decoder_policy,
            estimator_policy,
        )?
    } else if let Some(audio_dir) = audio_dir {
        evaluate_backend_suite_with_audio_directory(suite, model_pack, model_dir, audio_dir)?
    } else {
        evaluate_backend_suite(suite, model_pack, model_dir)?
    };
    emit_report(&result, report, no_fail)
}

fn run_decoder_sweep(
    suite: &Path,
    model_pack: &Path,
    model_dir: &Path,
    audio_dir: &Path,
    report: Option<PathBuf>,
) -> Result<()> {
    emit_json_report(
        &evaluate_decoder_sweep_with_audio_directory(
            suite,
            model_pack,
            model_dir,
            audio_dir,
            &standard_decoder_policies(),
        )?,
        report,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_decoder_eval(
    suite: &Path,
    policy: &str,
    model_pack: &Path,
    model_dir: &Path,
    audio_dir: &Path,
    report: Option<PathBuf>,
    no_fail: bool,
) -> Result<()> {
    emit_report(
        &evaluate_named_decoder_policy_with_audio_directory(
            suite, model_pack, model_dir, audio_dir, policy,
        )?,
        report,
        no_fail,
    )
}

fn run_decoder_recoverability(
    suite: &Path,
    model_pack: &Path,
    model_dir: &Path,
    audio_dir: &Path,
    report: Option<PathBuf>,
) -> Result<()> {
    emit_json_report(
        &evaluate_decoder_recoverability_with_audio_directory(
            suite, model_pack, model_dir, audio_dir,
        )?,
        report,
    )
}

fn emit_report<T>(report: &T, destination: Option<PathBuf>, no_fail: bool) -> Result<()>
where
    T: EvaluationOutcome + Serialize,
{
    emit_json_report(report, destination)?;
    if !report.passed() && !no_fail {
        bail!("evaluation acceptance budgets failed");
    }
    Ok(())
}

fn emit_json_report<T>(report: &T, destination: Option<PathBuf>) -> Result<()>
where
    T: Serialize,
{
    let json = serde_json::to_string_pretty(report)?;
    if let Some(path) = destination {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&path, format!("{json}\n"))
            .with_context(|| format!("writing {}", path.display()))?;
    }
    println!("{json}");
    Ok(())
}

fn write_json_file<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(value)?;
    fs::write(path, format!("{json}\n")).with_context(|| format!("writing {}", path.display()))
}

trait EvaluationOutcome {
    fn passed(&self) -> bool;
}

impl EvaluationOutcome for rhythm_map_eval::SuiteEvaluation {
    fn passed(&self) -> bool {
        self.passed
    }
}

impl EvaluationOutcome for rhythm_map_eval::BottleneckEvaluation {
    fn passed(&self) -> bool {
        self.passed
    }
}

impl EvaluationOutcome for rhythm_map_eval::DecoderPolicyEvaluation {
    fn passed(&self) -> bool {
        self.passed
    }
}
