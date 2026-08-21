use std::{
    collections::HashSet,
    fs::{self, File},
    io,
    path::{Component, Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use ureq::Agent;

use crate::dataset::sha256_file;

const DOWNLOAD_ATTEMPTS: usize = 3;

/// Purpose of one immutable artifact in a public-dataset lock.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PublicDatasetAssetRole {
    /// Audio consumed by end-to-end evaluation.
    Audio,
    /// Upstream artifact retained to audit checked-in reference truth.
    AnnotationSource,
}

/// One content-addressed artifact distributed by a public dataset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PublicDatasetAsset {
    /// Safe relative destination below the explicit output directory.
    pub path: String,
    /// Direct HTTPS source URL.
    pub url: String,
    /// Lowercase SHA-256 of the exact downloaded bytes.
    pub sha256: String,
    /// Exact encoded file size.
    pub size_bytes: u64,
    /// Whether the artifact is evaluation audio or an annotation audit source.
    pub role: PublicDatasetAssetRole,
}

/// Versioned source, rights, and immutable artifacts for a public dataset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PublicDatasetLock {
    /// Lock schema version.
    pub schema_version: u32,
    /// Stable dataset identifier.
    pub id: String,
    /// Upstream dataset version represented by this lock.
    pub version: String,
    /// Canonical human-readable dataset page.
    pub homepage_url: String,
    /// SPDX identifier for the locked artifacts.
    pub license: String,
    /// Credit that must accompany redistributed or derived material.
    pub attribution: String,
    /// Immutable artifacts selected for this evaluation slice.
    pub assets: Vec<PublicDatasetAsset>,
}

impl PublicDatasetLock {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            bail!(
                "unsupported public-dataset lock schema {}",
                self.schema_version
            );
        }
        if self.id.trim().is_empty()
            || self.version.trim().is_empty()
            || self.license.trim().is_empty()
            || self.attribution.trim().is_empty()
            || self.assets.is_empty()
        {
            bail!("dataset identity, rights, attribution, and assets are required");
        }
        if !self.homepage_url.starts_with("https://") {
            bail!("dataset homepage must use HTTPS");
        }
        let mut paths = HashSet::new();
        for asset in &self.assets {
            let path = Path::new(&asset.path);
            if asset.path.trim().is_empty()
                || path.is_absolute()
                || !path
                    .components()
                    .all(|component| matches!(component, Component::Normal(_)))
            {
                bail!(
                    "dataset asset path must stay below the output directory: {}",
                    asset.path
                );
            }
            if !paths.insert(asset.path.as_str()) {
                bail!("duplicate dataset asset path: {}", asset.path);
            }
            if !asset.url.starts_with("https://") {
                bail!("dataset asset URL must use HTTPS: {}", asset.url);
            }
            if asset.size_bytes == 0 || !is_lower_sha256(&asset.sha256) {
                bail!(
                    "dataset asset {} has an invalid size or SHA-256",
                    asset.path
                );
            }
        }
        Ok(())
    }
}

/// Whether a fetched dataset asset was already valid or downloaded now.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DatasetFetchStatus {
    /// The destination already matched the locked size and digest.
    Reused,
    /// New bytes were downloaded and verified before installation.
    Downloaded,
}

/// Verified result for one selected public-dataset artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetFetchAsset {
    /// Relative path copied from the lock.
    pub path: String,
    /// Immutable artifact role.
    pub role: PublicDatasetAssetRole,
    /// Locked and verified SHA-256.
    pub sha256: String,
    /// Reused or downloaded status.
    pub status: DatasetFetchStatus,
}

/// Machine-readable result of fetching one public-dataset slice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetFetchReport {
    /// Report schema version.
    pub schema_version: u32,
    /// Stable dataset identifier.
    pub dataset_id: String,
    /// Locked upstream version.
    pub dataset_version: String,
    /// Successfully verified selected artifacts.
    pub assets: Vec<DatasetFetchAsset>,
}

/// Fetch and verify a public dataset below an explicit directory.
///
/// Audio is always selected. Annotation-source artifacts are fetched only when
/// `include_annotations` is true. Existing files are trusted only after both
/// their exact byte length and SHA-256 match the lock.
///
/// # Errors
///
/// Returns an error for an invalid lock, unsafe output path, network failure,
/// or content that does not match the immutable identity in the lock.
pub fn fetch_public_dataset(
    lock_path: &Path,
    output_directory: &Path,
    include_annotations: bool,
) -> Result<DatasetFetchReport> {
    let bytes = fs::read(lock_path)
        .with_context(|| format!("reading dataset lock {}", lock_path.display()))?;
    let lock: PublicDatasetLock = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing dataset lock {}", lock_path.display()))?;
    lock.validate()?;
    fs::create_dir_all(output_directory)
        .with_context(|| format!("creating dataset output {}", output_directory.display()))?;
    let root = output_directory
        .canonicalize()
        .with_context(|| format!("resolving dataset output {}", output_directory.display()))?;
    let agent = Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(180)))
        .user_agent("NagareWorks-rhythm-map-eval/0.1")
        .build()
        .new_agent();
    let mut results = Vec::new();
    for asset in &lock.assets {
        if asset.role == PublicDatasetAssetRole::AnnotationSource && !include_annotations {
            continue;
        }
        let destination = safe_destination(&root, &asset.path)?;
        let status = if verify_asset(&destination, asset)? {
            DatasetFetchStatus::Reused
        } else {
            eprintln!("fetching {}", asset.path);
            download_verified(&agent, asset, &destination)?;
            DatasetFetchStatus::Downloaded
        };
        results.push(DatasetFetchAsset {
            path: asset.path.clone(),
            role: asset.role,
            sha256: asset.sha256.clone(),
            status,
        });
    }
    Ok(DatasetFetchReport {
        schema_version: 1,
        dataset_id: lock.id,
        dataset_version: lock.version,
        assets: results,
    })
}

fn safe_destination(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    let parent = relative
        .parent()
        .context("dataset asset destination has no parent")?;
    let mut resolved_parent = root.to_path_buf();
    for component in parent.components() {
        let Component::Normal(name) = component else {
            bail!("dataset asset destination escapes the output directory");
        };
        let candidate = resolved_parent.join(name);
        match fs::symlink_metadata(&candidate) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                bail!(
                    "dataset directory may not be a symbolic link: {}",
                    candidate.display()
                );
            }
            Ok(metadata) if !metadata.is_dir() => {
                bail!(
                    "dataset directory path is not a directory: {}",
                    candidate.display()
                );
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                fs::create_dir(&candidate).with_context(|| {
                    format!("creating dataset directory {}", candidate.display())
                })?;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("reading dataset directory {}", candidate.display()));
            }
        }
        resolved_parent = candidate
            .canonicalize()
            .with_context(|| format!("resolving dataset directory {}", candidate.display()))?;
        if !resolved_parent.starts_with(root) {
            bail!("dataset asset destination escapes the output directory");
        }
    }
    Ok(resolved_parent.join(
        relative
            .file_name()
            .context("dataset asset destination has no filename")?,
    ))
}

fn verify_asset(path: &Path, asset: &PublicDatasetAsset) -> Result<bool> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    if metadata.file_type().is_symlink() {
        bail!(
            "dataset destination may not be a symbolic link: {}",
            path.display()
        );
    }
    if !metadata.is_file() || metadata.len() != asset.size_bytes {
        return Ok(false);
    }
    Ok(sha256_file(path)? == asset.sha256)
}

fn download_verified(agent: &Agent, asset: &PublicDatasetAsset, destination: &Path) -> Result<()> {
    let filename = destination
        .file_name()
        .and_then(|value| value.to_str())
        .context("dataset asset filename is not UTF-8")?;
    let temporary = destination.with_file_name(format!(".{filename}.part"));
    reject_symlink(&temporary)?;
    let mut last_error = None;
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        if temporary.exists() {
            fs::remove_file(&temporary)
                .with_context(|| format!("removing stale download {}", temporary.display()))?;
        }
        let result = download_once(agent, asset, &temporary).and_then(|()| {
            if verify_asset(&temporary, asset)? {
                Ok(())
            } else {
                bail!("downloaded bytes did not match locked size and SHA-256")
            }
        });
        match result {
            Ok(()) => {
                replace_regular_file(&temporary, destination)?;
                return Ok(());
            }
            Err(error) => {
                last_error = Some(error.context(format!(
                    "attempt {attempt}/{DOWNLOAD_ATTEMPTS} for {}",
                    asset.path
                )));
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("dataset download failed")))
}

fn download_once(agent: &Agent, asset: &PublicDatasetAsset, temporary: &Path) -> Result<()> {
    let mut response = agent
        .get(&asset.url)
        .call()
        .with_context(|| format!("requesting {}", asset.url))?;
    let mut file = File::create(temporary)
        .with_context(|| format!("creating download {}", temporary.display()))?;
    io::copy(&mut response.body_mut().as_reader(), &mut file)
        .with_context(|| format!("downloading {}", asset.url))?;
    file.sync_all()
        .with_context(|| format!("syncing download {}", temporary.display()))?;
    Ok(())
}

fn replace_regular_file(temporary: &Path, destination: &Path) -> Result<()> {
    reject_symlink(destination)?;
    if destination.exists() {
        fs::remove_file(destination)
            .with_context(|| format!("replacing invalid asset {}", destination.display()))?;
    }
    fs::rename(temporary, destination).with_context(|| {
        format!(
            "installing verified asset {} as {}",
            temporary.display(),
            destination.display()
        )
    })
}

fn reject_symlink(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            bail!(
                "dataset path may not be a symbolic link: {}",
                path.display()
            )
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use sha2::{Digest, Sha256};

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    fn temporary_directory() -> PathBuf {
        let suffix = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rhythm-map-public-dataset-{}-{suffix}",
            std::process::id()
        ))
    }

    fn asset(bytes: &[u8]) -> PublicDatasetAsset {
        PublicDatasetAsset {
            path: "audio/example.mp3".to_string(),
            url: "https://example.invalid/example.mp3".to_string(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            size_bytes: bytes.len() as u64,
            role: PublicDatasetAssetRole::Audio,
        }
    }

    #[test]
    fn existing_asset_requires_size_and_digest() {
        let root = temporary_directory();
        fs::create_dir_all(root.join("audio")).unwrap();
        let expected = asset(b"expected bytes");
        let path = root.join(&expected.path);
        fs::write(&path, b"same byte count!").unwrap();
        assert!(!verify_asset(&path, &expected).unwrap());
        fs::write(&path, b"expected bytes").unwrap();
        assert!(verify_asset(&path, &expected).unwrap());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn valid_existing_asset_is_reused_without_network() {
        let root = temporary_directory();
        let output = root.join("output");
        fs::create_dir_all(output.join("audio")).unwrap();
        let expected = asset(b"expected bytes");
        fs::write(output.join(&expected.path), b"expected bytes").unwrap();
        let lock = PublicDatasetLock {
            schema_version: 1,
            id: "fixture".to_string(),
            version: "1".to_string(),
            homepage_url: "https://example.invalid".to_string(),
            license: "CC-BY-4.0".to_string(),
            attribution: "Fixture author".to_string(),
            assets: vec![expected],
        };
        let lock_path = root.join("lock.json");
        fs::write(&lock_path, serde_json::to_vec(&lock).unwrap()).unwrap();
        let report = fetch_public_dataset(&lock_path, &output, false).unwrap();
        assert_eq!(report.assets[0].status, DatasetFetchStatus::Reused);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn lock_rejects_paths_outside_the_output_root() {
        let mut escaped = asset(b"expected bytes");
        escaped.path = "../example.mp3".to_string();
        let lock = PublicDatasetLock {
            schema_version: 1,
            id: "fixture".to_string(),
            version: "1".to_string(),
            homepage_url: "https://example.invalid".to_string(),
            license: "CC-BY-4.0".to_string(),
            attribution: "Fixture author".to_string(),
            assets: vec![escaped],
        };

        assert!(lock.validate().is_err());
    }
}
