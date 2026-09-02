//! Explicit model setup followed by offline verification; no inference tuning.

use std::path::PathBuf;

use anyhow::{Context, Result};
use rhythm_map_models::{BEAT_THIS_FULL_MANIFEST, ModelPackCache};

fn main() -> Result<()> {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: 06-model-cache <cache-directory>")?;
    let cache = ModelPackCache::new(root);

    // This is an explicit setup step and may access the network. Only the
    // `download` feature enables it; core analysis and FFI remain offline.
    cache.fetch(BEAT_THIS_FULL_MANIFEST)?;

    // Later application runs need only this call. Reuse rechecks every file,
    // and a missing or modified cache fails without downloading a replacement.
    let pack = cache.verify(BEAT_THIS_FULL_MANIFEST)?;
    println!("{} {}", pack.manifest().id, pack.manifest_sha256());
    println!("model directory: {}", pack.root().display());
    Ok(())
}
