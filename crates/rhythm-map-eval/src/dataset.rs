use std::{
    cell::RefCell,
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use rhythm_map_beat_this::decode_audio;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::AudioReference;

/// Verified identity and decoded shape of one external evaluation asset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioAssetInspection {
    /// Report schema version.
    pub schema_version: u32,
    /// SHA-256 of the exact encoded file bytes.
    pub sha256: String,
    /// Encoded file size.
    pub size_bytes: u64,
    /// Decoded mono sample rate used by the Beat This adapter.
    pub decoded_sample_rate_hz: u32,
    /// Decoded audio duration.
    pub duration_s: f64,
}

/// Content-addressed resolver for public or private audio kept outside Git.
#[derive(Debug, Clone)]
pub struct ExternalAudioResolver {
    root: PathBuf,
    candidates: Vec<PathBuf>,
    digests: RefCell<HashMap<PathBuf, String>>,
}

impl ExternalAudioResolver {
    /// Bind a resolver to an explicit local audio directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the root does not exist or is not a directory.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let requested = root.as_ref();
        let root = requested
            .canonicalize()
            .with_context(|| format!("resolving audio directory {}", requested.display()))?;
        if !root.is_dir() {
            bail!("audio directory {} is not a directory", root.display());
        }
        let mut candidates = Vec::new();
        collect_audio_files(&root, &mut candidates)?;
        candidates.sort();
        Ok(Self {
            root,
            candidates,
            digests: RefCell::new(HashMap::new()),
        })
    }

    /// Resolve one audio reference by immutable file digest.
    ///
    /// The optional filename is tried first but is not authoritative. If it is
    /// absent or stale, supported audio files below the resolver root are
    /// searched deterministically for the declared digest.
    ///
    /// # Errors
    ///
    /// Returns an error when no file below the root has the expected digest.
    pub fn resolve(&self, reference: &AudioReference) -> Result<PathBuf> {
        if let Some(hint) = &reference.local_file_hint {
            let candidate = self.root.join(hint);
            if self.matches_reference(&candidate, reference)? {
                return Ok(candidate);
            }
        }

        for candidate in &self.candidates {
            if self.matches_reference(candidate, reference)? {
                return Ok(candidate.clone());
            }
        }
        bail!(
            "no supported audio below {} matches SHA-256 {}",
            self.root.display(),
            reference.sha256
        )
    }

    fn matches_reference(&self, path: &Path, reference: &AudioReference) -> Result<bool> {
        if !path.is_file() || !is_supported_audio_path(path) {
            return Ok(false);
        }
        let resolved = path
            .canonicalize()
            .with_context(|| format!("resolving audio candidate {}", path.display()))?;
        if !resolved.starts_with(&self.root) {
            return Ok(false);
        }
        let digest = if let Some(digest) = self.digests.borrow().get(&resolved) {
            digest.clone()
        } else {
            let digest = sha256_file(&resolved)?;
            self.digests.borrow_mut().insert(resolved, digest.clone());
            digest
        };
        Ok(digest == reference.sha256)
    }
}

/// Hash and decode one local audio file for manifest authoring.
///
/// # Errors
///
/// Returns an error when the file cannot be read or decoded.
pub fn inspect_audio_asset(path: impl AsRef<Path>) -> Result<AudioAssetInspection> {
    let path = path.as_ref();
    let metadata = fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
    if !metadata.is_file() {
        bail!("audio asset {} is not a file", path.display());
    }
    let sha256 = sha256_file(path)?;
    let decoded = decode_audio(path).with_context(|| format!("decoding {}", path.display()))?;
    let duration_s = usize_to_f64(decoded.samples.len()) / f64::from(decoded.sample_rate);
    Ok(AudioAssetInspection {
        schema_version: 1,
        sha256,
        size_bytes: metadata.len(),
        decoded_sample_rate_hz: decoded.sample_rate,
        duration_s,
    })
}

fn collect_audio_files(directory: &Path, result: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("reading audio directory {}", directory.display()))?
    {
        let entry =
            entry.with_context(|| format!("reading entry below {}", directory.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("reading file type for {}", entry.path().display()))?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            collect_audio_files(&entry.path(), result)?;
        } else if file_type.is_file() && is_supported_audio_path(&entry.path()) {
            result.push(entry.path());
        }
    }
    Ok(())
}

fn is_supported_audio_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "aac" | "aif" | "aiff" | "flac" | "m4a" | "mp3" | "ogg" | "opus" | "wav"
            )
        })
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hashing {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    fn temporary_directory() -> PathBuf {
        let suffix = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rhythm-map-eval-dataset-{}-{suffix}",
            std::process::id()
        ))
    }

    #[test]
    fn stale_hint_falls_back_to_content_addressed_search() {
        let root = temporary_directory();
        let nested = root.join("nested");
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join("stale.wav"), b"wrong bytes").unwrap();
        let expected = nested.join("renamed.flac");
        fs::write(&expected, b"expected bytes").unwrap();
        let reference = AudioReference {
            sha256: sha256_file(&expected).unwrap(),
            local_file_hint: Some("stale.wav".to_string()),
        };

        let resolved = ExternalAudioResolver::new(&root)
            .unwrap()
            .resolve(&reference)
            .unwrap();
        assert_eq!(resolved, expected.canonicalize().unwrap());
        fs::remove_dir_all(root).unwrap();
    }
}
