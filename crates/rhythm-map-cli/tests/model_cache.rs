//! Network-free subprocess coverage for the packaged CLI model workflow.

use std::{fs, process::Command};

use rhythm_map_models::{BEAT_THIS_FULL_MANIFEST, ModelPackCache, ModelPackManifest};

#[test]
fn cli_verifies_custom_pack_and_reuses_cache_offline() {
    let temp = tempfile::tempdir().unwrap();
    let mut manifest: ModelPackManifest = serde_json::from_slice(BEAT_THIS_FULL_MANIFEST).unwrap();
    for artifact in &mut manifest.artifacts {
        artifact.size_bytes = 3;
        artifact.sha256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into();
        artifact.download_url = "https://network-must-not-be-used.invalid/model".into();
    }
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let manifest_path = temp.path().join("trusted.json");
    fs::write(&manifest_path, &bytes).unwrap();
    let cache = ModelPackCache::new(temp.path());
    let directory = cache.directory_for(&bytes).unwrap();
    let artifacts = directory.join("artifacts");
    fs::create_dir_all(&artifacts).unwrap();
    fs::write(directory.join("manifest.json"), &bytes).unwrap();
    for artifact in &manifest.artifacts {
        fs::write(artifacts.join(&artifact.file), b"abc").unwrap();
    }
    for operation in ["verify", "fetch"] {
        let output = Command::new(env!("CARGO_BIN_EXE_rhythm-map"))
            .args(["models", operation, "--model-pack"])
            .arg(&manifest_path)
            .arg("--cache-dir")
            .arg(temp.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["id"], manifest.id);
        assert_eq!(
            fs::canonicalize(result["model_dir"].as_str().unwrap()).unwrap(),
            fs::canonicalize(&artifacts).unwrap()
        );
    }
    fs::write(artifacts.join(&manifest.artifacts[0].file), b"bad").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rhythm-map"))
        .args(["models", "fetch", "--model-pack"])
        .arg(&manifest_path)
        .arg("--cache-dir")
        .arg(temp.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("SHA-256"));
}

#[test]
fn missing_cache_fails_offline_with_setup_guidance() {
    let temp = tempfile::tempdir().unwrap();
    let cache = temp.path().join("absent");
    let output = Command::new(env!("CARGO_BIN_EXE_rhythm-map"))
        .arg("missing.wav")
        .env("RHYTHM_MAP_CACHE_DIR", &cache)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("models fetch"));
    assert!(!cache.exists());
}
