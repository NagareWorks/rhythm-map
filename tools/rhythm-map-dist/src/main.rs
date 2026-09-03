//! Assemble auditable native and browser distribution directories.

use std::fmt::Write as _;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use sha2::{Digest, Sha256};

const PACKAGE_SCHEMA_VERSION: u32 = 1;
const PRODUCT_CAPABILITY: &str = "rhythm/time-map";

#[derive(Debug, Parser)]
#[command(about = "Assemble verifiable Rhythm Map distribution directories")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the distribution version inherited from the workspace.
    Version,
    /// Require a release tag to match the workspace version exactly.
    VerifyTag {
        /// Git tag in the form `vMAJOR.MINOR.PATCH`.
        #[arg(long)]
        tag: String,
    },
    /// Assemble a CLI and C ABI package from an existing native release build.
    Native(PackageArgs),
    /// Assemble a browser package from existing wasm-bindgen output.
    Wasm(PackageArgs),
}

#[derive(Debug, Args)]
struct PackageArgs {
    /// Repository root containing LICENSE, NOTICE, examples, include, and models.
    #[arg(long, default_value = ".")]
    repository_root: PathBuf,
    /// Directory containing already-built native or wasm-bindgen artifacts.
    #[arg(long)]
    input_dir: PathBuf,
    /// New, empty package directory to create.
    #[arg(long)]
    output_dir: PathBuf,
    /// Rust target triple represented by this package.
    #[arg(long)]
    target: String,
    /// Full immutable Git commit recorded in the package manifest.
    #[arg(long)]
    git_commit: String,
}

#[derive(Debug, Serialize)]
struct DistributionManifest {
    schema_version: u32,
    package: &'static str,
    version: &'static str,
    target: String,
    git_commit: String,
    capabilities: [&'static str; 1],
    model_artifacts_included: bool,
    entry_points: EntryPoints,
    files: Vec<PackagedFile>,
}

#[derive(Debug, Serialize)]
struct EntryPoints {
    cli: Option<String>,
    c_abi: Option<String>,
    wasm_module: Option<String>,
}

#[derive(Debug, Serialize)]
struct PackagedFile {
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, PartialEq, Eq)]
struct NativeEntryNames {
    executable: &'static str,
    shared_library: &'static str,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Version => println!("{}", env!("CARGO_PKG_VERSION")),
        Command::VerifyTag { tag } => verify_tag(&tag)?,
        Command::Native(args) => package_native(&args)?,
        Command::Wasm(args) => package_wasm(&args)?,
    }
    Ok(())
}

fn verify_tag(tag: &str) -> Result<()> {
    let expected = format!("v{}", env!("CARGO_PKG_VERSION"));
    if tag != expected {
        bail!("release tag {tag:?} does not match workspace version {expected:?}");
    }
    Ok(())
}

fn package_native(args: &PackageArgs) -> Result<()> {
    validate_args(args)?;
    create_new_output(&args.output_dir)?;

    let names = native_entry_names(&args.target)?;

    copy_required(
        &args.input_dir.join(names.executable),
        &args.output_dir.join("bin").join(names.executable),
    )?;
    copy_required(
        &args.input_dir.join(names.shared_library),
        &args.output_dir.join("lib").join(names.shared_library),
    )?;
    copy_repository_file(args, "include/rhythm_map.h", "include/rhythm_map.h")?;
    copy_repository_file(
        args,
        "models/beat-this-full-v1.json",
        "models/beat-this-full-v1.json",
    )?;
    copy_repository_file(args, "LICENSE", "LICENSE")?;
    copy_repository_file(args, "NOTICE", "NOTICE")?;
    copy_repository_file(
        args,
        "licenses/beat-this-rs-MIT.txt",
        "licenses/beat-this-rs-MIT.txt",
    )?;
    copy_repository_file(
        args,
        "examples/05-distribution/verify_package.py",
        "verify_package.py",
    )?;
    copy_required(
        &args
            .repository_root
            .join("tools/rhythm-map-dist/assets/native-README.md"),
        &args.output_dir.join("README.md"),
    )?;

    finalize_package(
        args,
        "rhythm-map-native",
        EntryPoints {
            cli: Some(format!("bin/{}", names.executable)),
            c_abi: Some(format!("lib/{}", names.shared_library)),
            wasm_module: None,
        },
    )
}

fn native_entry_names(target: &str) -> Result<NativeEntryNames> {
    if target.contains("windows") {
        Ok(NativeEntryNames {
            executable: "rhythm-map.exe",
            shared_library: "rhythm_map_ffi.dll",
        })
    } else if target.contains("apple") {
        Ok(NativeEntryNames {
            executable: "rhythm-map",
            shared_library: "librhythm_map_ffi.dylib",
        })
    } else if target.contains("linux") {
        Ok(NativeEntryNames {
            executable: "rhythm-map",
            shared_library: "librhythm_map_ffi.so",
        })
    } else {
        bail!("unsupported native distribution target: {target}");
    }
}

fn package_wasm(args: &PackageArgs) -> Result<()> {
    validate_args(args)?;
    if args.target != "wasm32-unknown-unknown" {
        bail!("browser package target must be wasm32-unknown-unknown");
    }
    create_new_output(&args.output_dir)?;

    for required in ["rhythm_map.js", "rhythm_map_bg.wasm"] {
        copy_required(
            &args.input_dir.join(required),
            &args.output_dir.join("pkg").join(required),
        )?;
    }
    for optional in ["rhythm_map.d.ts", "rhythm_map_bg.wasm.d.ts"] {
        copy_optional(
            &args.input_dir.join(optional),
            &args.output_dir.join("pkg").join(optional),
        )?;
    }
    for demo_file in [
        "index.html",
        "main.js",
        "observations.json",
        "package.json",
        "smoke.mjs",
        "styles.css",
    ] {
        copy_repository_file(
            args,
            &format!("examples/04-browser-wasm/{demo_file}"),
            demo_file,
        )?;
    }
    copy_repository_file(args, "LICENSE", "LICENSE")?;
    copy_repository_file(args, "NOTICE", "NOTICE")?;
    copy_repository_file(
        args,
        "examples/05-distribution/verify_package.py",
        "verify_package.py",
    )?;
    copy_required(
        &args
            .repository_root
            .join("tools/rhythm-map-dist/assets/wasm-README.md"),
        &args.output_dir.join("README.md"),
    )?;

    finalize_package(
        args,
        "rhythm-map-browser-wasm",
        EntryPoints {
            cli: None,
            c_abi: None,
            wasm_module: Some("pkg/rhythm_map.js".to_string()),
        },
    )
}

fn validate_args(args: &PackageArgs) -> Result<()> {
    if args.target.trim().is_empty() {
        bail!("target must not be empty");
    }
    validate_git_commit(&args.git_commit)?;
    if !args.repository_root.is_dir() {
        bail!(
            "repository root does not exist: {}",
            args.repository_root.display()
        );
    }
    if !args.input_dir.is_dir() {
        bail!(
            "input directory does not exist: {}",
            args.input_dir.display()
        );
    }
    validate_repository_identity(&args.repository_root, &args.git_commit)?;
    Ok(())
}

fn validate_git_commit(value: &str) -> Result<()> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("git commit must be a full 40-character lowercase hexadecimal SHA");
    }
    Ok(())
}

fn validate_repository_identity(repository: &Path, expected_commit: &str) -> Result<()> {
    let head = git_output(repository, &["rev-parse", "HEAD"])?;
    if head.trim() != expected_commit {
        bail!(
            "repository HEAD {} does not match requested package commit {}",
            head.trim(),
            expected_commit
        );
    }
    let status = git_output(
        repository,
        &["status", "--porcelain", "--untracked-files=all"],
    )?;
    if !status.trim().is_empty() {
        bail!("repository worktree is not clean; refusing to package an uncommitted source state");
    }
    Ok(())
}

fn git_output(repository: &Path, arguments: &[&str]) -> Result<String> {
    let output = ProcessCommand::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .with_context(|| format!("failed to run git in {}", repository.display()))?;
    if !output.status.success() {
        bail!(
            "git {} failed in {}: {}",
            arguments.join(" "),
            repository.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout).context("git output was not valid UTF-8")
}

fn create_new_output(output: &Path) -> Result<()> {
    if output.exists() {
        bail!(
            "output directory already exists; refusing to merge stale package files: {}",
            output.display()
        );
    }
    fs::create_dir_all(output)
        .with_context(|| format!("failed to create package directory {}", output.display()))
}

fn copy_repository_file(args: &PackageArgs, source: &str, destination: &str) -> Result<()> {
    copy_required(
        &args.repository_root.join(source),
        &args.output_dir.join(destination),
    )
}

fn copy_required(source: &Path, destination: &Path) -> Result<()> {
    if !source.is_file() {
        bail!("required package input is missing: {}", source.display());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::copy(source, destination).with_context(|| {
        format!(
            "failed to copy {} to {}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn copy_optional(source: &Path, destination: &Path) -> Result<()> {
    if source.exists() {
        copy_required(source, destination)?;
    }
    Ok(())
}

fn finalize_package(
    args: &PackageArgs,
    package: &'static str,
    entry_points: EntryPoints,
) -> Result<()> {
    let files = collect_packaged_files(&args.output_dir)?;
    let manifest = DistributionManifest {
        schema_version: PACKAGE_SCHEMA_VERSION,
        package,
        version: env!("CARGO_PKG_VERSION"),
        target: args.target.clone(),
        git_commit: args.git_commit.clone(),
        capabilities: [PRODUCT_CAPABILITY],
        model_artifacts_included: false,
        entry_points,
        files,
    };
    let manifest_path = args.output_dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;

    let checksummed_files = collect_packaged_files(&args.output_dir)?;
    let mut sums = String::new();
    for file in checksummed_files {
        writeln!(&mut sums, "{}  {}", file.sha256, file.path)?;
    }
    let sums_path = args.output_dir.join("SHA256SUMS");
    fs::write(&sums_path, sums)
        .with_context(|| format!("failed to write {}", sums_path.display()))?;
    Ok(())
}

fn collect_packaged_files(root: &Path) -> Result<Vec<PackagedFile>> {
    let mut paths = Vec::new();
    collect_file_paths(root, root, &mut paths)?;
    paths.sort();
    paths
        .into_iter()
        .map(|relative| {
            let absolute = root.join(&relative);
            let metadata = fs::metadata(&absolute)
                .with_context(|| format!("failed to inspect {}", absolute.display()))?;
            Ok(PackagedFile {
                path: portable_path(&relative),
                size_bytes: metadata.len(),
                sha256: sha256_file(&absolute)?,
            })
        })
        .collect()
}

fn collect_file_paths(root: &Path, directory: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("failed to read {}", directory.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            bail!(
                "package output must not contain symlinks: {}",
                entry.path().display()
            );
        }
        if file_type.is_dir() {
            collect_file_paths(root, &entry.path(), output)?;
        } else if file_type.is_file() && entry.file_name() != "SHA256SUMS" {
            output.push(entry.path().strip_prefix(root)?.to_path_buf());
        }
    }
    Ok(())
}

fn portable_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open {} for hashing", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapted_source_license_is_retained_in_crate_and_native_distribution() {
        assert_eq!(
            include_str!("../../../licenses/beat-this-rs-MIT.txt"),
            include_str!("../../../crates/rhythm-map-beat-this/LICENSE-MIT"),
        );
    }

    #[test]
    fn release_tag_must_match_workspace_version() {
        assert!(verify_tag(&format!("v{}", env!("CARGO_PKG_VERSION"))).is_ok());
        assert!(verify_tag("v999.0.0").is_err());
    }

    #[test]
    fn package_identity_requires_a_full_lowercase_sha() {
        assert!(validate_git_commit(&"a".repeat(40)).is_ok());
        assert!(validate_git_commit(&"A".repeat(40)).is_err());
        assert!(validate_git_commit("abc").is_err());
    }

    #[test]
    fn native_entry_points_follow_each_supported_platform() {
        assert_eq!(
            native_entry_names("x86_64-pc-windows-msvc").unwrap(),
            NativeEntryNames {
                executable: "rhythm-map.exe",
                shared_library: "rhythm_map_ffi.dll",
            }
        );
        assert_eq!(
            native_entry_names("aarch64-apple-darwin").unwrap(),
            NativeEntryNames {
                executable: "rhythm-map",
                shared_library: "librhythm_map_ffi.dylib",
            }
        );
        assert_eq!(
            native_entry_names("x86_64-unknown-linux-gnu").unwrap(),
            NativeEntryNames {
                executable: "rhythm-map",
                shared_library: "librhythm_map_ffi.so",
            }
        );
        assert!(native_entry_names("wasm32-unknown-unknown").is_err());
    }

    #[test]
    fn metadata_is_content_addressed_and_portable() {
        let root =
            std::env::temp_dir().join(format!("rhythm-map-dist-test-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }
        fs::create_dir_all(root.join("nested")).unwrap();
        fs::write(root.join("nested/example.txt"), b"rhythm-map").unwrap();

        let files = collect_packaged_files(&root).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "nested/example.txt");
        assert_eq!(files[0].size_bytes, 10);
        assert_eq!(
            files[0].sha256,
            "b2b474daa259679452607f12649d2b58a1af76ee3fc343a279247218ba7e6bc2"
        );

        fs::remove_dir_all(root).unwrap();
    }
}
