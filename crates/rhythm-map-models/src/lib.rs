//! Versioned model-pack provenance and local artifact verification.

use std::{
    collections::HashSet,
    fs::File,
    io::{self, Read},
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Current serialized model-pack manifest version.
pub const MODEL_PACK_SCHEMA_VERSION: u32 = 1;

/// Complete provenance and compatibility contract for one model pack.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelPackManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Stable pack identifier.
    pub id: String,
    /// Immutable human-readable pack version.
    pub version: String,
    /// Observation backend that consumes the pack.
    pub backend: String,
    /// Upstream code and conversion provenance.
    pub source: ModelSource,
    /// Audio and activation contract expected by the backend.
    pub feature_contract: FeatureContract,
    /// Files required to load the pack.
    pub artifacts: Vec<ModelArtifact>,
}

/// Upstream repository and conversion provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSource {
    /// Canonical upstream repository URL.
    pub repository: String,
    /// Immutable upstream commit or release identity.
    pub revision: String,
    /// SPDX identifier for upstream code and model weights.
    pub license: String,
    /// Model conversion provenance, if the distributed files were converted.
    #[serde(default)]
    pub conversion: Option<ModelConversion>,
}

/// Reproducible conversion information for derived model files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelConversion {
    /// Conversion tool or script identity.
    pub tool: String,
    /// Command documented by the upstream maintainer.
    pub command: String,
    /// Original checkpoint identifier.
    pub source_checkpoint: String,
}

/// Backend feature contract that must remain stable across model files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureContract {
    /// PCM sample rate expected by the log-mel front end.
    pub sample_rate_hz: u32,
    /// Number of frequency bands emitted by the audio frontend.
    ///
    /// The serialized name is retained for schema-v1 compatibility even when
    /// a backend uses a non-mel logarithmic filterbank.
    pub mel_bands: u32,
    /// Complete per-frame model input width, including derived features.
    #[serde(default)]
    pub input_feature_count: Option<u32>,
    /// Analysis window length in PCM samples, when fixed.
    #[serde(default)]
    pub window_size_samples: Option<u32>,
    /// Analysis hop length in PCM samples, when fixed.
    #[serde(default)]
    pub hop_size_samples: Option<u32>,
    /// Stable description of the frontend feature family.
    #[serde(default)]
    pub feature_kind: Option<String>,
    /// Activation frame rate emitted by the model.
    pub frame_rate_hz: f64,
    /// Maximum inference chunk length in seconds, when fixed.
    #[serde(default)]
    pub chunk_duration_s: Option<f64>,
}

/// Semantic role of a model artifact.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ModelArtifactRole {
    /// Audio-to-log-mel frontend graph.
    MelFrontend,
    /// Beat and downbeat prediction graph.
    BeatModel,
    /// Self-contained rhythm activation graph with a native Rust frontend.
    RhythmModel,
}

/// Content-addressed file in a model pack.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelArtifact {
    /// Artifact purpose.
    pub role: ModelArtifactRole,
    /// Relative filename below the chosen model directory.
    pub file: String,
    /// Lowercase SHA-256 of the exact bytes.
    pub sha256: String,
    /// Exact expected file size.
    pub size_bytes: u64,
    /// Immutable or content-addressed upstream download URL.
    pub download_url: String,
}

/// A manifest whose local artifacts have passed size and SHA-256 checks.
#[derive(Debug, Clone)]
pub struct VerifiedModelPack {
    manifest: ModelPackManifest,
    manifest_sha256: String,
    root: PathBuf,
}

impl VerifiedModelPack {
    /// Verified manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ModelPackManifest {
        &self.manifest
    }

    /// SHA-256 of the serialized manifest file that was verified.
    #[must_use]
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    /// Resolve a verified artifact by semantic role.
    #[must_use]
    pub fn path_for(&self, role: ModelArtifactRole) -> Option<PathBuf> {
        self.manifest
            .artifacts
            .iter()
            .find(|artifact| artifact.role == role)
            .map(|artifact| self.root.join(&artifact.file))
    }
}

/// Model manifest or artifact verification failure.
#[derive(Debug, Error)]
pub enum ModelPackError {
    /// Manifest JSON could not be decoded.
    #[error("invalid model-pack manifest {path}: {source}")]
    InvalidJson {
        /// Manifest path.
        path: PathBuf,
        /// JSON parser error.
        #[source]
        source: serde_json::Error,
    },
    /// Manifest or artifact file could not be read.
    #[error("failed to read {path}: {source}")]
    Io {
        /// Failed path.
        path: PathBuf,
        /// Filesystem error.
        #[source]
        source: io::Error,
    },
    /// Manifest violates the schema contract.
    #[error("invalid model-pack manifest: {0}")]
    InvalidManifest(String),
    /// Artifact size does not match the manifest.
    #[error("model artifact {path} has size {actual}, expected {expected}")]
    SizeMismatch {
        /// Artifact path.
        path: PathBuf,
        /// Expected byte count.
        expected: u64,
        /// Actual byte count.
        actual: u64,
    },
    /// Artifact digest does not match the manifest.
    #[error("model artifact {path} has SHA-256 {actual}, expected {expected}")]
    DigestMismatch {
        /// Artifact path.
        path: PathBuf,
        /// Expected lowercase digest.
        expected: String,
        /// Actual lowercase digest.
        actual: String,
    },
    /// An artifact resolved through a link outside the selected model root.
    #[error("model artifact {path} resolves outside model root {root}")]
    ArtifactOutsideRoot {
        /// Canonical artifact path.
        path: PathBuf,
        /// Canonical selected model root.
        root: PathBuf,
    },
}

impl ModelPackManifest {
    /// Validate structure, provenance, feature values, roles, and paths.
    ///
    /// # Errors
    ///
    /// Returns [`ModelPackError::InvalidManifest`] for an unsafe or incomplete
    /// manifest.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> Result<(), ModelPackError> {
        if self.schema_version != MODEL_PACK_SCHEMA_VERSION {
            return Err(ModelPackError::InvalidManifest(format!(
                "unsupported schema version {}",
                self.schema_version
            )));
        }
        for (name, value) in [
            ("id", self.id.as_str()),
            ("version", self.version.as_str()),
            ("backend", self.backend.as_str()),
            ("source repository", self.source.repository.as_str()),
            ("source revision", self.source.revision.as_str()),
            ("source license", self.source.license.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ModelPackError::InvalidManifest(format!(
                    "{name} must not be empty"
                )));
            }
        }
        if !self.source.repository.starts_with("https://") {
            return Err(ModelPackError::InvalidManifest(
                "source repository must use HTTPS".to_string(),
            ));
        }
        if let Some(conversion) = &self.source.conversion {
            for (name, value) in [
                ("conversion tool", conversion.tool.as_str()),
                ("conversion command", conversion.command.as_str()),
                ("source checkpoint", conversion.source_checkpoint.as_str()),
            ] {
                if value.trim().is_empty() {
                    return Err(ModelPackError::InvalidManifest(format!(
                        "{name} must not be empty"
                    )));
                }
            }
        }
        if self.feature_contract.sample_rate_hz == 0
            || self.feature_contract.mel_bands == 0
            || self
                .feature_contract
                .input_feature_count
                .is_some_and(|value| value == 0)
            || self
                .feature_contract
                .window_size_samples
                .is_some_and(|value| value == 0)
            || self
                .feature_contract
                .hop_size_samples
                .is_some_and(|value| value == 0)
            || !self.feature_contract.frame_rate_hz.is_finite()
            || self.feature_contract.frame_rate_hz <= 0.0
            || self
                .feature_contract
                .chunk_duration_s
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(ModelPackError::InvalidManifest(
                "feature contract values must be finite and positive".to_string(),
            ));
        }
        if self
            .feature_contract
            .feature_kind
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(ModelPackError::InvalidManifest(
                "feature kind must not be empty when present".to_string(),
            ));
        }
        if self.artifacts.is_empty() {
            return Err(ModelPackError::InvalidManifest(
                "at least one model artifact is required".to_string(),
            ));
        }
        let mut roles = HashSet::new();
        for artifact in &self.artifacts {
            if !roles.insert(artifact.role) {
                return Err(ModelPackError::InvalidManifest(format!(
                    "duplicate artifact role {:?}",
                    artifact.role
                )));
            }
            validate_relative_file(&artifact.file)?;
            if artifact.size_bytes == 0 {
                return Err(ModelPackError::InvalidManifest(format!(
                    "artifact {} has zero size",
                    artifact.file
                )));
            }
            if !is_lower_sha256(&artifact.sha256) {
                return Err(ModelPackError::InvalidManifest(format!(
                    "artifact {} has an invalid SHA-256",
                    artifact.file
                )));
            }
            if !artifact.download_url.starts_with("https://") {
                return Err(ModelPackError::InvalidManifest(format!(
                    "artifact {} download URL must use HTTPS",
                    artifact.file
                )));
            }
        }
        let required_roles: &[ModelArtifactRole] = match self.backend.as_str() {
            "beat-this-rten" => &[ModelArtifactRole::MelFrontend, ModelArtifactRole::BeatModel],
            "beatnet-rten-experimental" => &[ModelArtifactRole::RhythmModel],
            backend => {
                return Err(ModelPackError::InvalidManifest(format!(
                    "unsupported model-pack backend {backend:?}"
                )));
            }
        };
        for &required in required_roles {
            if !roles.contains(&required) {
                return Err(ModelPackError::InvalidManifest(format!(
                    "missing required artifact role {required:?}"
                )));
            }
        }
        Ok(())
    }
}

/// Load a manifest and verify every artifact below an explicit local root.
///
/// # Errors
///
/// Returns [`ModelPackError`] for invalid JSON, unsafe manifests, missing
/// artifacts, or size and digest mismatches.
pub fn verify_model_pack(
    manifest_path: &Path,
    artifact_root: &Path,
) -> Result<VerifiedModelPack, ModelPackError> {
    let bytes = std::fs::read(manifest_path).map_err(|source| ModelPackError::Io {
        path: manifest_path.to_path_buf(),
        source,
    })?;
    let manifest: ModelPackManifest =
        serde_json::from_slice(&bytes).map_err(|source| ModelPackError::InvalidJson {
            path: manifest_path.to_path_buf(),
            source,
        })?;
    manifest.validate()?;
    let canonical_root =
        std::fs::canonicalize(artifact_root).map_err(|source| ModelPackError::Io {
            path: artifact_root.to_path_buf(),
            source,
        })?;
    for artifact in &manifest.artifacts {
        let requested_path = canonical_root.join(&artifact.file);
        let path = std::fs::canonicalize(&requested_path).map_err(|source| ModelPackError::Io {
            path: requested_path,
            source,
        })?;
        if !path.starts_with(&canonical_root) {
            return Err(ModelPackError::ArtifactOutsideRoot {
                path,
                root: canonical_root,
            });
        }
        let metadata = std::fs::metadata(&path).map_err(|source| ModelPackError::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.len() != artifact.size_bytes {
            return Err(ModelPackError::SizeMismatch {
                path,
                expected: artifact.size_bytes,
                actual: metadata.len(),
            });
        }
        let actual = sha256_file(&path)?;
        if actual != artifact.sha256 {
            return Err(ModelPackError::DigestMismatch {
                path,
                expected: artifact.sha256.clone(),
                actual,
            });
        }
    }
    Ok(VerifiedModelPack {
        manifest,
        manifest_sha256: sha256_bytes(&bytes),
        root: canonical_root,
    })
}

fn validate_relative_file(value: &str) -> Result<(), ModelPackError> {
    let path = Path::new(value);
    if value.trim().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(ModelPackError::InvalidManifest(format!(
            "artifact path {value:?} must stay below the model root"
        )));
    }
    Ok(())
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_file(path: &Path) -> Result<String, ModelPackError> {
    let mut file = File::open(path).map_err(|source| ModelPackError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|source| ModelPackError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn rejects_parent_path() {
        assert!(matches!(
            validate_relative_file("../model.onnx"),
            Err(ModelPackError::InvalidManifest(_))
        ));
    }

    #[test]
    fn checked_in_full_pack_is_schema_valid() {
        let manifest: ModelPackManifest =
            serde_json::from_str(include_str!("../../../models/beat-this-full-v1.json")).unwrap();
        manifest.validate().unwrap();
    }

    #[test]
    fn checked_in_beatnet_pack_is_schema_valid() {
        let manifest: ModelPackManifest =
            serde_json::from_str(include_str!("../../../models/beatnet-v1.json")).unwrap();
        manifest.validate().unwrap();
    }

    #[test]
    fn checked_in_small_pack_is_schema_valid() {
        let manifest: ModelPackManifest =
            serde_json::from_str(include_str!("../../../models/beat-this-small-v1.json")).unwrap();
        manifest.validate().unwrap();
    }
}
