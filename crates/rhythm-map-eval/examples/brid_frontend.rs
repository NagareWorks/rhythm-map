//! Full-record BRID frontend export; frozen research inputs, not a product API.
use anyhow::{Context, Result, ensure};
use beat_this::{Model, RtenRuntime, Runtime, Tensor};
use clap::Parser;
use rhythm_map_beat_this::{OBSERVATION_CONTRACT, decode_audio};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

const REFERENCES: &[u8] = include_bytes!("../../../evaluation/datasets/brid-reference-v1.json");
const REFERENCE_SHA: &str = "ad70e2ed3a005c1d349368e25be405abb5f6a779f1efcf22d2f495291aa34a6a";
const MEL_SHA: &str = "fdd59e65c515331308e4c8841edf99972deca646bdf6197744c2a5b7755e3de9";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    audio_dir: PathBuf,
    /// Exact `mel_spectrogram.onnx` from the existing full model pack.
    #[arg(long)]
    mel_model: PathBuf,
    /// Fresh private directory. Binary PCM/mel must stay outside Git.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize)]
struct References {
    records: Vec<Record>,
}

#[derive(Deserialize)]
struct Record {
    recording_id: String,
    audio_sha256: String,
    frames: usize,
    duration_s: f64,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn records() -> Result<Vec<Record>> {
    ensure!(
        digest(REFERENCES) == REFERENCE_SHA,
        "reference identity drift"
    );
    let refs: References = serde_json::from_slice(REFERENCES)?;
    ensure!(refs.records.len() == 93, "full population required");
    for (index, row) in refs.records.iter().enumerate() {
        ensure!(
            row.recording_id == format!("{:04}", index + 1),
            "ordered population drift"
        );
    }
    Ok(refs.records)
}

fn check_extent(
    samples: usize,
    rate: u32,
    shape: &[usize],
    values: &[f32],
    row: &Record,
) -> Result<()> {
    // Both sides are exact nonnegative integer sample counts, not approximate durations.
    ensure!(
        rate == 22050
            && f64::from(u32::try_from(samples)?).to_bits()
                == (row.duration_s * 22050.0).round().to_bits(),
        "shipping decoder extent changed"
    );
    ensure!(
        row.frames == 1 + samples / 441 && shape == [1, row.frames, 128],
        "frontend frame geometry changed; do not crop or pad references"
    );
    ensure!(
        values.len() == row.frames * 128 && values.iter().all(|v| v.is_finite()),
        "invalid mel values"
    );
    Ok(())
}

fn save_f32(path: &Path, values: &[f32]) -> Result<String> {
    let mut writer = BufWriter::new(fs::File::create_new(path)?);
    let mut hash = Sha256::new();
    for value in values {
        let bytes = value.to_le_bytes();
        writer.write_all(&bytes)?;
        hash.update(bytes);
    }
    writer.flush()?;
    Ok(format!("{:x}", hash.finalize()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let population = records()?;
    ensure!(
        digest(&fs::read(&args.mel_model)?) == MEL_SHA,
        "frontend model changed"
    );
    let mut model = RtenRuntime.load_model(&args.mel_model)?;
    fs::create_dir(&args.output).context("fresh private output directory required")?;
    let mut rows = Vec::new();
    for row in population {
        let id = &row.recording_id;
        let path = args.audio_dir.join("audio").join(format!("{id}.wav"));
        ensure!(
            digest(&fs::read(&path)?) == row.audio_sha256,
            "audio identity changed: {id}"
        );
        let audio = decode_audio(&path)?;
        ensure!(audio.samples.iter().all(|v| v.is_finite()), "nonfinite PCM");
        let input = Tensor {
            shape: vec![1, audio.samples.len()],
            data: audio.samples,
        };
        // Same named model call as beat-this 1.0.0 MelExtractor (private upstream
        // module). No alternate resampler, frontend math, truncation or NN heads.
        let mut outputs = model.run(&[("audio_pcm", &input)])?;
        let mel = outputs
            .remove("mel_spectrogram")
            .context("missing mel output")?;
        check_extent(
            input.data.len(),
            audio.sample_rate,
            &mel.shape,
            &mel.data,
            &row,
        )?;
        let pcm_sha = save_f32(&args.output.join(format!("{id}.pcm.f32")), &input.data)?;
        let mel_sha = save_f32(&args.output.join(format!("{id}.mel.f32")), &mel.data)?;
        rows.push(json!({"recording_id": id, "audio_sha256": row.audio_sha256,
            "samples": input.data.len(), "frames": row.frames,
            "pcm_sha256": pcm_sha, "mel_sha256": mel_sha}));
        eprintln!(
            "Frontend {id}: {} samples, {} frames",
            input.data.len(),
            row.frames
        );
    }
    let report = json!({"schema": "rhythm-map.brid-frontend.v1", "records": rows,
        "reference_sha256": REFERENCE_SHA, "mel_model_sha256": MEL_SHA,
        "observation_contract": OBSERVATION_CONTRACT,
        "exporter_sha256": digest(include_bytes!("brid_frontend.rs")),
        "audio_preprocessing_sha256": digest(include_bytes!("../../rhythm-map-beat-this/src/audio.rs")),
        "sample_rate": 22050, "frame_rate": 50, "full_recordings": true,
        "encoder_calls": 0, "optimizer_steps": 0});
    let mut writer = BufWriter::new(fs::File::create_new(args.output.join("report.json"))?);
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_population_and_extent_are_fixed() {
        let rows = records().unwrap();
        assert_eq!(rows.len(), 93);
        assert_eq!(rows.iter().map(|r| r.frames).sum::<usize>(), 133_247);
        let row = &rows[0];
        let samples = 793_800; // The frozen first record is exactly 36 seconds.
        let mut values = vec![0.; row.frames * 128];
        assert!(check_extent(samples, 22050, &[1, row.frames, 128], &values, row).is_ok());
        assert!(check_extent(samples - 1, 22050, &[1, row.frames, 128], &values, row).is_err());
        assert!(check_extent(samples, 44100, &[1, row.frames, 128], &values, row).is_err());
        assert!(check_extent(samples, 22050, &[1, row.frames - 1, 128], &values, row).is_err());
        values[0] = f32::NAN;
        assert!(check_extent(samples, 22050, &[1, row.frames, 128], &values, row).is_err());
    }
}
