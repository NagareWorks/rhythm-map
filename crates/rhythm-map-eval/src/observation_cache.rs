use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use rhythm_map_core::RhythmObservations;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const OBSERVATION_CACHE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ObservationCacheKey {
    schema_version: u32,
    audio_sha256: String,
    model_manifest_sha256: String,
    backend_contract: String,
    decoder_policy: String,
}

impl ObservationCacheKey {
    pub(crate) fn new(
        audio_sha256: String,
        model_manifest_sha256: String,
        backend_contract: String,
        decoder_policy: String,
    ) -> Result<Self> {
        for (label, digest) in [
            ("audio", &audio_sha256),
            ("model manifest", &model_manifest_sha256),
        ] {
            if !is_lower_sha256(digest) {
                bail!("observation cache {label} SHA-256 must be 64 lowercase hex characters");
            }
        }
        if backend_contract.is_empty() {
            bail!("observation cache backend contract must not be empty");
        }
        if decoder_policy.is_empty() {
            bail!("observation cache decoder policy must not be empty");
        }
        Ok(Self {
            schema_version: OBSERVATION_CACHE_SCHEMA_VERSION,
            audio_sha256,
            model_manifest_sha256,
            backend_contract,
            decoder_policy,
        })
    }

    fn digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(self).context("serializing observation cache key")?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct DecodedAudioIdentity {
    sample_rate_hz: u32,
    mono_sample_count: u64,
}

impl DecodedAudioIdentity {
    pub(crate) fn new(sample_rate_hz: u32, mono_sample_count: usize) -> Result<Self> {
        if sample_rate_hz == 0 {
            bail!("observation cache sample rate must be non-zero");
        }
        Ok(Self {
            sample_rate_hz,
            mono_sample_count: u64::try_from(mono_sample_count)
                .context("decoded sample count does not fit u64")?,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ObservationCacheEntry {
    schema_version: u32,
    key: ObservationCacheKey,
    decoded_audio: DecodedAudioIdentity,
    observations: RhythmObservations,
}

#[derive(Debug, Clone)]
pub(crate) struct ObservationCache {
    root: PathBuf,
}

impl ObservationCache {
    pub(crate) fn new(root: &Path) -> Result<Self> {
        if root.as_os_str().is_empty() {
            bail!("observation cache directory must not be empty");
        }
        Ok(Self {
            root: root.to_path_buf(),
        })
    }

    pub(crate) fn load(
        &self,
        key: &ObservationCacheKey,
        decoded_audio: DecodedAudioIdentity,
    ) -> Result<Option<RhythmObservations>> {
        let path = self.entry_path(key)?;
        if !path
            .try_exists()
            .with_context(|| format!("checking observation cache entry {}", path.display()))?
        {
            return Ok(None);
        }
        let bytes = fs::read(&path)
            .with_context(|| format!("reading observation cache entry {}", path.display()))?;
        let entry = serde_json::from_slice::<ObservationCacheEntry>(&bytes)
            .with_context(|| format!("decoding observation cache entry {}", path.display()))?;
        if entry.schema_version != OBSERVATION_CACHE_SCHEMA_VERSION {
            bail!(
                "observation cache entry {} has schema {}, expected {}",
                path.display(),
                entry.schema_version,
                OBSERVATION_CACHE_SCHEMA_VERSION
            );
        }
        if entry.key != *key {
            bail!(
                "observation cache entry {} does not match its content-addressed key",
                path.display()
            );
        }
        if entry.decoded_audio != decoded_audio {
            bail!(
                "observation cache entry {} does not match decoded audio shape",
                path.display()
            );
        }
        Ok(Some(entry.observations))
    }

    pub(crate) fn store(
        &self,
        key: &ObservationCacheKey,
        decoded_audio: DecodedAudioIdentity,
        observations: &RhythmObservations,
    ) -> Result<()> {
        let path = self.entry_path(key)?;
        if path
            .try_exists()
            .with_context(|| format!("checking observation cache entry {}", path.display()))?
        {
            return Ok(());
        }
        let parent = path
            .parent()
            .context("observation cache entry has no parent directory")?;
        fs::create_dir_all(parent).with_context(|| {
            format!("creating observation cache directory {}", parent.display())
        })?;
        let digest = key.digest()?;
        let temporary = parent.join(format!(".{digest}.{}.tmp", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .with_context(|| {
                format!(
                    "creating temporary observation cache entry {}",
                    temporary.display()
                )
            })?;
        let entry = ObservationCacheEntry {
            schema_version: OBSERVATION_CACHE_SCHEMA_VERSION,
            key: key.clone(),
            decoded_audio,
            observations: observations.clone(),
        };
        serde_json::to_writer_pretty(&mut file, &entry)
            .context("serializing observation cache entry")?;
        file.write_all(b"\n")
            .context("terminating observation cache entry")?;
        file.sync_all()
            .context("syncing temporary observation cache entry")?;
        drop(file);
        if let Err(error) = fs::rename(&temporary, &path) {
            if path.try_exists().unwrap_or(false) {
                fs::remove_file(&temporary).with_context(|| {
                    format!(
                        "removing raced temporary observation cache entry {}",
                        temporary.display()
                    )
                })?;
            } else {
                return Err(error).with_context(|| {
                    format!("publishing observation cache entry {}", path.display())
                });
            }
        }
        Ok(())
    }

    fn entry_path(&self, key: &ObservationCacheKey) -> Result<PathBuf> {
        let digest = key.digest()?;
        Ok(self
            .root
            .join(format!("v{OBSERVATION_CACHE_SCHEMA_VERSION}"))
            .join(&digest[..2])
            .join(format!("{digest}.json")))
    }
}

pub(crate) fn decoded_pcm_sha256(samples: &[f32], sample_rate_hz: u32) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"rhythm-map-decoded-mono-pcm-v1\0");
    hasher.update(sample_rate_hz.to_le_bytes());
    hasher.update(
        u64::try_from(samples.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    for sample in samples {
        hasher.update(sample.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use rhythm_map_core::{ModelInfo, ObservedBeat, RhythmObservations};

    use super::{DecodedAudioIdentity, ObservationCache, ObservationCacheKey, decoded_pcm_sha256};

    fn observations() -> RhythmObservations {
        RhythmObservations {
            duration_s: 1.0,
            beats: vec![ObservedBeat {
                time_s: 0.420_000_076_293_945,
                confidence: 0.999_902_304_067_608_7,
                downbeat_confidence: 0.4,
            }],
            beat_candidates: Vec::new(),
            activations: None,
            activity: Vec::new(),
            onsets: Vec::new(),
            harmonic_changes: Vec::new(),
            source: ModelInfo {
                backend: "test".to_string(),
                model: "fixture".to_string(),
                version: Some("1".to_string()),
                frame_rate_hz: Some(50.0),
            },
        }
    }

    fn unique_cache_root() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rhythm-map-observation-cache-test-{}-{nonce}",
            std::process::id()
        ))
    }

    fn key(policy: &str) -> ObservationCacheKey {
        ObservationCacheKey::new(
            "a".repeat(64),
            "b".repeat(64),
            "test-observation-v1".to_string(),
            policy.to_string(),
        )
        .unwrap()
    }

    #[test]
    fn content_addressed_entry_round_trips_exact_observations() {
        let root = unique_cache_root();
        let cache = ObservationCache::new(&root).unwrap();
        let identity = DecodedAudioIdentity::new(22_050, 22_050).unwrap();
        let expected = observations();

        assert!(cache.load(&key("upstream"), identity).unwrap().is_none());
        cache.store(&key("upstream"), identity, &expected).unwrap();
        assert_eq!(
            cache.load(&key("upstream"), identity).unwrap(),
            Some(expected)
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn decoder_policy_is_part_of_the_cache_address() {
        let root = unique_cache_root();
        let cache = ObservationCache::new(&root).unwrap();
        let identity = DecodedAudioIdentity::new(22_050, 22_050).unwrap();
        cache
            .store(&key("upstream"), identity, &observations())
            .unwrap();

        assert!(cache.load(&key("candidate"), identity).unwrap().is_none());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn every_inference_input_identity_is_part_of_the_cache_address() {
        let root = unique_cache_root();
        let cache = ObservationCache::new(&root).unwrap();
        let identity = DecodedAudioIdentity::new(22_050, 22_050).unwrap();
        let baseline = key("upstream");
        cache.store(&baseline, identity, &observations()).unwrap();
        let alternatives = [
            ObservationCacheKey::new(
                "c".repeat(64),
                "b".repeat(64),
                "test-observation-v1".to_string(),
                "upstream".to_string(),
            )
            .unwrap(),
            ObservationCacheKey::new(
                "a".repeat(64),
                "c".repeat(64),
                "test-observation-v1".to_string(),
                "upstream".to_string(),
            )
            .unwrap(),
            ObservationCacheKey::new(
                "a".repeat(64),
                "b".repeat(64),
                "test-observation-v2".to_string(),
                "upstream".to_string(),
            )
            .unwrap(),
        ];

        for alternative in alternatives {
            assert!(cache.load(&alternative, identity).unwrap().is_none());
        }

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn decoded_pcm_identity_is_stable_and_shape_sensitive() {
        let samples = [0.0, -0.0, 0.25, -0.5];
        assert_eq!(
            decoded_pcm_sha256(&samples, 44_100),
            decoded_pcm_sha256(&samples, 44_100)
        );
        assert_ne!(
            decoded_pcm_sha256(&samples, 44_100),
            decoded_pcm_sha256(&samples, 48_000)
        );
        assert_ne!(
            decoded_pcm_sha256(&samples, 44_100),
            decoded_pcm_sha256(&samples[..3], 44_100)
        );
    }
}
