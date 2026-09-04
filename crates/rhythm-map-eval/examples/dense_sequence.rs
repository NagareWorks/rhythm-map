//! Private JSON-lines evaluation runner; never a production decoder selection.
use anyhow::Result;
use rhythm_map_core::{RhythmObservations, analyze_observations};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::{self, BufRead};
use std::time::Instant;

#[path = "support/dense_sequence.rs"]
mod sequence;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    frames: sequence::Frames,
    baseline_observations: Option<RhythmObservations>,
}

fn main() -> Result<()> {
    for line in io::stdin().lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;
        let baseline = request
            .baseline_observations
            .as_ref()
            .map(analyze_observations)
            .transpose()?;
        let start = Instant::now();
        // No baseline events, case ID, annotation or output path crosses this call.
        let decoded = sequence::decode(&request.frames)?;
        println!(
            "{}",
            json!({"schema_version":1,
            "purpose":"private_frozen_dense_clock_prediction", "automatic_candidate":true,
            "production_output_changed":false, "frame_rate_hz":50,
            "decoder_source_sha256":format!("{:x}", Sha256::digest(include_bytes!("support/dense_sequence.rs"))),
            "runner_source_sha256":format!("{:x}", Sha256::digest(include_bytes!("dense_sequence.rs"))),
            "elapsed_s":start.elapsed().as_secs_f64(), "decoded":decoded, "baseline":baseline})
        );
    }
    Ok(())
}
