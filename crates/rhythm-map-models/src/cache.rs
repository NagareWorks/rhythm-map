use std::{
    fs, io,
    path::{Path, PathBuf},
};

use crate::{
    ModelPackError, ModelPackManifest, VerifiedModelPack, sha256_bytes, verify_model_pack_bytes,
};

/// Caller-owned cache of complete, manifest-addressed model packs.
///
/// Verification never downloads. Acquisition is available only with `download`.
/// Treat the cache directory and custom manifests as trusted local inputs; hashes
/// establish byte identity, not authorship, licensing, or safety of a model graph.
#[derive(Debug, Clone)]
pub struct ModelPackCache {
    root: PathBuf,
}

impl ModelPackCache {
    /// Select an explicit cache root. This does not touch the filesystem.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Resolve a pack directory from the exact trusted manifest bytes.
    ///
    /// # Errors
    /// Rejects an invalid manifest before filesystem or network access.
    pub fn directory_for(&self, manifest: &[u8]) -> Result<PathBuf, ModelPackError> {
        parse(manifest)?;
        Ok(self.root.join("model-packs").join(sha256_bytes(manifest)))
    }

    /// Verify an installed pack against trusted manifest bytes, without network IO.
    ///
    /// # Errors
    /// Missing, modified, linked, or incomplete cache entries are rejected.
    pub fn verify(&self, manifest: &[u8]) -> Result<VerifiedModelPack, ModelPackError> {
        let parsed = parse(manifest)?;
        let directory = self.directory_for(manifest)?;
        reject_link(&self.root.join("model-packs"))?;
        reject_link(&directory)?;
        let manifest_path = directory.join("manifest.json");
        reject_link(&manifest_path)?;
        let stored = fs::read(&manifest_path).map_err(|e| io_error(&manifest_path, e))?;
        if stored != manifest {
            return Err(ModelPackError::Cache(format!(
                "cached manifest differs from trusted manifest: {}",
                manifest_path.display()
            )));
        }
        let artifacts = directory.join("artifacts");
        reject_link(&artifacts)?;
        for artifact in &parsed.artifacts {
            let mut path = artifacts.clone();
            for part in artifact.file.split('/') {
                path.push(part);
                reject_link(&path)?;
            }
        }
        verify_model_pack_bytes(manifest, &artifacts)
    }

    /// Explicitly acquire a trusted manifest's pack over HTTPS, or reuse a fully
    /// verified local entry. Nothing is downloaded by the analysis engine.
    ///
    /// Downloads are streamed with an exact-size bound, staged privately, and
    /// published only after every digest passes. A corrupt existing entry is an
    /// error, never silently overwritten. Retry after failure starts a fresh stage.
    ///
    /// # Errors
    /// Returns manifest, network, IO, integrity, or publication failures.
    #[cfg(feature = "download")]
    pub fn fetch(&self, manifest: &[u8]) -> Result<VerifiedModelPack, ModelPackError> {
        use std::time::Duration;
        let agent = ureq::Agent::config_builder()
            .https_only(true)
            .timeout_global(Some(Duration::from_secs(900)))
            .timeout_connect(Some(Duration::from_secs(30)))
            .user_agent("NagareWorks-rhythm-map-models/0.1")
            .build()
            .new_agent();
        self.fetch_with(manifest, |artifact| {
            let response = agent.get(&artifact.download_url).call().map_err(|error| {
                ModelPackError::Cache(format!("downloading {}: {error}", artifact.file))
            })?;
            Ok(Box::new(response.into_body().into_reader()))
        })
    }

    #[cfg(any(feature = "download", test))]
    fn fetch_with(
        &self,
        manifest: &[u8],
        mut open: impl FnMut(&crate::ModelArtifact) -> Result<Box<dyn io::Read>, ModelPackError>,
    ) -> Result<VerifiedModelPack, ModelPackError> {
        use io::Read;
        let parsed = parse(manifest)?;
        let directory = self.directory_for(manifest)?;
        let parent = self.root.join("model-packs");
        fs::create_dir_all(&self.root).map_err(|e| io_error(&self.root, e))?;
        match fs::create_dir(&parent) {
            Ok(()) => (),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => reject_link(&parent)?,
            Err(e) => return Err(io_error(&parent, e)),
        }
        match fs::symlink_metadata(&directory) {
            Ok(_) => return self.verify(manifest),
            Err(e) if e.kind() == io::ErrorKind::NotFound => (),
            Err(e) => return Err(io_error(&directory, e)),
        }
        let stage = tempfile::Builder::new()
            .prefix(".download-")
            .tempdir_in(&parent)
            .map_err(|e| io_error(&parent, e))?;
        let artifacts = stage.path().join("artifacts");
        fs::create_dir(&artifacts).map_err(|e| io_error(&artifacts, e))?;
        for artifact in &parsed.artifacts {
            let path = artifacts.join(&artifact.file);
            let parent = path.parent().expect("artifact below staging root");
            fs::create_dir_all(parent).map_err(|e| io_error(parent, e))?;
            let limit = artifact
                .size_bytes
                .checked_add(1)
                .ok_or_else(|| ModelPackError::Cache("artifact size overflow".into()))?;
            let mut input = open(artifact)?.take(limit);
            let mut output = fs::File::create_new(&path).map_err(|e| io_error(&path, e))?;
            io::copy(&mut input, &mut output).map_err(|e| io_error(&path, e))?;
            output.sync_all().map_err(|e| io_error(&path, e))?;
            drop(output);
            artifact.verify_file(&path)?;
        }
        let manifest_path = stage.path().join("manifest.json");
        fs::write(&manifest_path, manifest).map_err(|e| io_error(&manifest_path, e))?;
        verify_model_pack_bytes(manifest, &artifacts)?;
        // Do not replace an entry (even an empty/incomplete one) that appeared
        // while this download was in flight. Concurrent successful publications
        // are immutable, nonempty directories, so rename cannot replace them.
        if fs::symlink_metadata(&directory).is_ok() {
            return self.verify(manifest);
        }
        if let Err(error) = fs::rename(stage.path(), &directory) {
            // Another caller may have published this exact pack while downloading.
            // Its content must pass the same trusted manifest, not merely exist.
            if directory.exists() {
                return self.verify(manifest);
            }
            return Err(io_error(&directory, error));
        }
        self.verify(manifest)
    }
}

fn parse(bytes: &[u8]) -> Result<ModelPackManifest, ModelPackError> {
    let manifest: ModelPackManifest =
        serde_json::from_slice(bytes).map_err(|source| ModelPackError::InvalidJson {
            path: "<manifest bytes>".into(),
            source,
        })?;
    manifest.validate()?;
    Ok(manifest)
}

fn reject_link(path: &Path) -> Result<(), ModelPackError> {
    let metadata = fs::symlink_metadata(path).map_err(|e| io_error(path, e))?;
    #[cfg(windows)]
    let is_link = {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0 // FILE_ATTRIBUTE_REPARSE_POINT, including junctions.
    };
    #[cfg(not(windows))]
    let is_link = metadata.file_type().is_symlink();
    if is_link {
        return Err(ModelPackError::Cache(format!(
            "cache entry must not be a link: {}",
            path.display()
        )));
    }
    Ok(())
}

fn io_error(path: &Path, source: io::Error) -> ModelPackError {
    ModelPackError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture() -> Vec<u8> {
        let mut manifest: ModelPackManifest =
            serde_json::from_slice(crate::BEAT_THIS_FULL_MANIFEST).unwrap();
        for artifact in &mut manifest.artifacts {
            artifact.size_bytes = 3;
            artifact.sha256 = sha256_bytes(b"abc");
        }
        serde_json::to_vec(&manifest).unwrap()
    }

    #[allow(clippy::unnecessary_wraps)] // Same signature as the fallible network opener.
    fn good(_: &crate::ModelArtifact) -> Result<Box<dyn io::Read>, ModelPackError> {
        Ok(Box::new(Cursor::new(b"abc")))
    }

    #[test]
    fn complete_pack_is_reused_without_network() {
        let root = tempfile::tempdir().unwrap();
        let cache = ModelPackCache::new(root.path());
        let bytes = fixture();
        cache.fetch_with(&bytes, good).unwrap();
        let pack = cache
            .fetch_with(&bytes, |_| panic!("cache hit must be offline"))
            .unwrap();
        assert_eq!(pack.manifest_sha256(), sha256_bytes(&bytes));
        assert!(pack.root().ends_with("artifacts"));
    }

    #[test]
    fn truncated_oversized_or_wrong_digest_is_never_published() {
        for payload in [b"ab".as_slice(), b"abcd", b"bad"] {
            let root = tempfile::tempdir().unwrap();
            let cache = ModelPackCache::new(root.path());
            let bytes = fixture();
            assert!(
                cache
                    .fetch_with(&bytes, |_| Ok(Box::new(Cursor::new(payload.to_vec()))))
                    .is_err()
            );
            assert!(!cache.directory_for(&bytes).unwrap().exists());
            assert_eq!(
                fs::read_dir(root.path().join("model-packs"))
                    .unwrap()
                    .count(),
                0
            );
            cache.fetch_with(&bytes, good).unwrap();
        }
    }

    #[test]
    fn second_file_failure_leaves_no_partial_pack() {
        let root = tempfile::tempdir().unwrap();
        let cache = ModelPackCache::new(root.path());
        let bytes = fixture();
        let mut calls = 0;
        assert!(
            cache
                .fetch_with(&bytes, |artifact| {
                    calls += 1;
                    if calls == 2 {
                        return Err(ModelPackError::Cache("connection failed".into()));
                    }
                    good(artifact)
                })
                .is_err()
        );
        assert_eq!(
            fs::read_dir(root.path().join("model-packs"))
                .unwrap()
                .count(),
            0
        );
    }

    #[test]
    fn unbounded_and_interrupted_streams_do_not_publish() {
        struct Interrupted;
        impl io::Read for Interrupted {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(
                    io::ErrorKind::ConnectionReset,
                    "test interruption",
                ))
            }
        }
        let root = tempfile::tempdir().unwrap();
        let cache = ModelPackCache::new(root.path());
        let bytes = fixture();
        assert!(
            cache
                .fetch_with(&bytes, |_| Ok(Box::new(io::repeat(b'a'))))
                .is_err()
        );
        assert!(
            cache
                .fetch_with(&bytes, |_| Ok(Box::new(Interrupted)))
                .is_err()
        );
        assert_eq!(
            fs::read_dir(root.path().join("model-packs"))
                .unwrap()
                .count(),
            0
        );
    }

    #[cfg(unix)]
    #[test]
    fn linked_cache_entries_are_not_followed() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let cache = ModelPackCache::new(root.path());
        let bytes = fixture();
        let pack = cache.fetch_with(&bytes, good).unwrap();
        let path = pack.path_for(crate::ModelArtifactRole::BeatModel).unwrap();
        let outside = root.path().join("outside");
        fs::rename(&path, &outside).unwrap();
        symlink(&outside, &path).unwrap();
        assert!(cache.verify(&bytes).is_err());
        assert!(
            cache
                .fetch_with(&bytes, |_| panic!("must not download"))
                .is_err()
        );
    }

    #[test]
    fn corrupt_cache_is_not_overwritten_or_downloaded() {
        let root = tempfile::tempdir().unwrap();
        let cache = ModelPackCache::new(root.path());
        let bytes = fixture();
        let pack = cache.fetch_with(&bytes, good).unwrap();
        let file = pack.path_for(crate::ModelArtifactRole::BeatModel).unwrap();
        fs::write(&file, b"bad").unwrap();
        assert!(
            cache
                .fetch_with(&bytes, |_| panic!("corrupt cache must not download"))
                .is_err()
        );
        assert_eq!(fs::read(file).unwrap(), b"bad");
    }

    #[test]
    fn cached_manifest_cannot_redefine_integrity() {
        let root = tempfile::tempdir().unwrap();
        let cache = ModelPackCache::new(root.path());
        let bytes = fixture();
        cache.fetch_with(&bytes, good).unwrap();
        fs::write(
            cache.directory_for(&bytes).unwrap().join("manifest.json"),
            b"{}",
        )
        .unwrap();
        assert!(cache.verify(&bytes).is_err());
    }

    #[test]
    fn invalid_manifest_fails_before_io_and_missing_verify_does_not_create_cache() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("absent");
        let cache = ModelPackCache::new(&path);
        assert!(
            cache
                .fetch_with(b"{}", |_| panic!("invalid manifest must not download"))
                .is_err()
        );
        assert!(cache.verify(&fixture()).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn manifest_changes_get_distinct_cache_entries() {
        let root = tempfile::tempdir().unwrap();
        let cache = ModelPackCache::new(root.path());
        let bytes = fixture();
        let mut other = bytes.clone();
        other.push(b' ');
        assert_ne!(
            cache.directory_for(&bytes).unwrap(),
            cache.directory_for(&other).unwrap()
        );
    }

    #[test]
    fn concurrent_fetches_publish_one_valid_pack() {
        let root = tempfile::tempdir().unwrap();
        let cache = ModelPackCache::new(root.path());
        let barrier = std::sync::Barrier::new(2);
        let bytes = fixture();
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    let mut first = true;
                    cache
                        .fetch_with(&bytes, |artifact| {
                            if first {
                                first = false;
                                barrier.wait();
                            }
                            good(artifact)
                        })
                        .unwrap();
                });
            }
        });
        cache.verify(&bytes).unwrap();
        assert_eq!(
            fs::read_dir(root.path().join("model-packs"))
                .unwrap()
                .count(),
            1
        );
    }
}
