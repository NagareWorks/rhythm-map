//! Private, model-free view of the unchanged shipping activity/silence contract.
//! Single diagnostic sentinels query membership only; they are never beat evidence.
use anyhow::{Result, ensure};
use clap::Parser;
use rhythm_map_core::{
    BackendError, Engine, EstimatorOptions, ModelInfo, ObservedBeat, RhythmObservationBackend,
    RhythmObservations, analyze_observations,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

const RATE: u32 = 24_000;
const MAX_SAMPLES: usize = 30 * 24_000;

#[derive(Parser)]
struct Args {
    /// Complete authored little-endian mono f32 PCM, not a decoded music file.
    #[arg(long)]
    pcm: PathBuf,
    #[arg(long)]
    pcm_sha256: String,
    #[arg(long)]
    output: PathBuf,
}

struct EmptyBackend;

fn empty_observations(duration_s: f64) -> RhythmObservations {
    RhythmObservations {
        duration_s,
        beats: Vec::new(),
        beat_candidates: Vec::new(),
        activations: None,
        activity: Vec::new(),
        onsets: Vec::new(),
        harmonic_changes: Vec::new(),
        source: ModelInfo {
            backend: "authored-activity-audit".into(),
            model: "no-model-no-beat-evidence".into(),
            version: None,
            frame_rate_hz: None,
        },
    }
}

impl RhythmObservationBackend for EmptyBackend {
    fn observe_mono(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<RhythmObservations, BackendError> {
        let count =
            u32::try_from(samples.len()).map_err(|_| BackendError::new("input too long"))?;
        Ok(empty_observations(
            f64::from(count) / f64::from(sample_rate),
        ))
    }
}

fn observe(samples: &[f32]) -> Result<RhythmObservations> {
    ensure!(
        (1025..=MAX_SAMPLES).contains(&samples.len()),
        "bounded complete PCM required"
    );
    ensure!(
        samples.iter().all(|x| x.is_finite() && x.abs() <= 1.0),
        "invalid PCM samples"
    );
    Ok(Engine::new(EmptyBackend).observe_pcm(samples, RATE, 1)?)
}

fn retained_at(observations: &RhythmObservations, time_s: f64) -> Result<Option<bool>> {
    ensure!(
        time_s.is_finite() && time_s >= 0.0 && time_s <= observations.duration_s,
        "invalid query"
    );
    ensure!(
        observations.beats.is_empty() && observations.beat_candidates.is_empty(),
        "not an empty audit backend"
    );
    // The estimator treats an absent envelope as no silence constraint. This
    // research composition instead retains explicit unknown availability.
    if !activity_covers(observations, time_s) {
        return Ok(None);
    }
    let mut probe = observations.clone();
    probe.beats.push(ObservedBeat {
        time_s,
        confidence: 1.0,
        downbeat_confidence: 0.0,
    });
    let analysis = analyze_observations(&probe)?;
    ensure!(
        analysis.beats.len() <= 1
            && analysis.global_bpm.is_none()
            && analysis.tempo_curve.is_empty(),
        "sentinel must not become a tempo inference"
    );
    let retained = !analysis.beats.is_empty();
    if retained {
        ensure!(
            analysis.beats[0].time_s.to_bits() == time_s.to_bits(),
            "sentinel moved"
        );
    } else {
        ensure!(
            analysis
                .warnings
                .iter()
                .any(|w| w == "low_activity_beats_rejected"),
            "sentinel removed for an unexpected reason"
        );
    }
    Ok(Some(retained))
}

fn activity_covers(observations: &RhythmObservations, time_s: f64) -> bool {
    let Some(first) = observations.activity.first() else {
        return false;
    };
    let last = observations.activity.last().expect("nonempty activity");
    let hop = if observations.activity.len() == 1 {
        observations.duration_s.max(0.001)
    } else {
        let mut hops = observations
            .activity
            .windows(2)
            .map(|p| p[1].time_s - p[0].time_s)
            .collect::<Vec<_>>();
        hops.sort_by(f64::total_cmp);
        let mid = hops.len() / 2;
        if hops.len().is_multiple_of(2) {
            f64::midpoint(hops[mid - 1], hops[mid])
        } else {
            hops[mid]
        }
    };
    // This exporter only takes the engine's complete uniform envelope. A
    // missing cell is unknown, not permission to extrapolate the tail or gaps.
    observations.activity.iter().all(|p| p.time_s.is_finite())
        && hop.is_finite()
        && hop > 0.0
        && observations.activity.windows(2).all(|p| {
            let gap = p[1].time_s - p[0].time_s;
            gap > 0.0 && (gap - hop).abs() <= 1e-12
        })
        && time_s >= (first.time_s - hop * 0.5).max(0.0)
        && time_s <= (last.time_s + hop * 0.5).min(observations.duration_s)
}

fn source_hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn private_parent(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("parent required"))?
        .canonicalize()?;
    ensure!(
        parent.ancestors().all(|p| !p.join(".git").exists()),
        "private output must be outside Git"
    );
    #[cfg(windows)]
    {
        let system = std::env::var("SystemDrive").unwrap_or_else(|_| "C:".into());
        let normalized = parent
            .to_string_lossy()
            .to_ascii_lowercase()
            .replace("\\\\?\\", "");
        ensure!(
            !normalized.starts_with(&system.to_ascii_lowercase()),
            "private output must be off system drive"
        );
    }
    ensure!(!path.exists(), "output must be fresh");
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    private_parent(&args.output)?;
    let metadata = fs::symlink_metadata(&args.pcm)?;
    ensure!(
        metadata.file_type().is_file() && metadata.len() <= u64::try_from(MAX_SAMPLES * 4)?,
        "invalid PCM file"
    );
    let mut bytes = Vec::new();
    File::open(&args.pcm)?
        .take(u64::try_from(MAX_SAMPLES * 4 + 1)?)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= MAX_SAMPLES * 4 && bytes.len().is_multiple_of(4),
        "invalid PCM byte count"
    );
    ensure!(source_hash(&bytes) == args.pcm_sha256, "PCM hash mismatch");
    let samples = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    let observations = observe(&samples)?;
    let count = (samples.len() / 240).div_ceil(4);
    let retained = (0..count)
        .map(|i| retained_at(&observations, f64::from(u32::try_from(i)?) / 25.0))
        .collect::<Result<Vec<_>>>()?;
    let defaults = EstimatorOptions::default();
    let report = json!({
        "schema": "rhythm-map.native-activity.v1", "samples": samples.len(), "sample_rate": RATE,
        "pcm_sha256": args.pcm_sha256, "activity": observations.activity,
        "native_token_retained": retained,
        "method": "one-sentinel-membership-query-per-native-center-not-beat-evidence",
        "silence_threshold_db": defaults.silence_threshold_db, "minimum_silence_s": defaults.minimum_silence_s,
        "source_sha256": {
            "engine": source_hash(include_bytes!("../../rhythm-map-core/src/engine.rs")),
            "estimator": source_hash(include_bytes!("../../rhythm-map-core/src/estimator.rs")),
            "types": source_hash(include_bytes!("../../rhythm-map-core/src/types.rs")),
            "cargo_lock": source_hash(include_bytes!("../../../Cargo.lock")),
            "exporter": source_hash(include_bytes!("activity_availability.rs"))
        },
        "pretrained_inference": false, "production_change": false
    });
    let mut output = BufWriter::new(
        File::options()
            .create_new(true)
            .write(true)
            .open(args.output)?,
    );
    serde_json::to_writer_pretty(&mut output, &report)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhythm_map_core::AudioActivityPoint;

    #[test]
    fn all_zero_pcm_is_rejected_without_a_tempo_result() {
        let observations = observe(&vec![0.0; 48_000]).unwrap();
        assert!(
            observations
                .activity
                .iter()
                .all(|p| p.relative_db == -120.0)
        );
        for t in [0.0, 0.5, 1.0, 1.96] {
            assert_eq!(retained_at(&observations, t).unwrap(), Some(false));
        }
        assert_eq!(retained_at(&observations, 2.0).unwrap(), None);
    }

    #[test]
    fn uniform_level_and_quiet_gain_do_not_imply_rhythm_or_silence() {
        for amplitude in [0.125, 0.125 / 1_048_576.0] {
            let observations = observe(&vec![amplitude; 48_000]).unwrap();
            assert!(observations.activity.iter().all(|p| p.relative_db == 0.0));
            assert_eq!(retained_at(&observations, 1.0).unwrap(), Some(true));
        }
    }

    #[test]
    fn short_rest_keeps_the_query_and_long_rest_rejects_it() {
        for (end, expected) in [(24_000, true), (40_800, false)] {
            let mut pcm = vec![0.125; 72_000];
            pcm[12_000..end].fill(0.0);
            let observations = observe(&pcm).unwrap();
            assert_eq!(retained_at(&observations, 0.75).unwrap(), Some(expected));
            assert_eq!(retained_at(&observations, 0.25).unwrap(), Some(true));
            assert_eq!(retained_at(&observations, 2.5).unwrap(), Some(true));
        }
    }

    #[test]
    fn missing_activity_is_unknown_not_an_audibility_vote() {
        assert_eq!(retained_at(&empty_observations(2.0), 1.0).unwrap(), None);
    }

    #[test]
    fn missing_cell_and_uncovered_native_tail_are_unknown() {
        let mut observations = observe(&vec![0.125; 48_000]).unwrap();
        observations.activity.remove(20);
        assert_eq!(retained_at(&observations, 1.0).unwrap(), None);
        let tail = observe(&vec![0.0; 290_160]).unwrap();
        assert_eq!(retained_at(&tail, 12.04).unwrap(), Some(false));
        assert_eq!(retained_at(&tail, 12.08).unwrap(), None);
    }

    #[test]
    fn exact_default_threshold_and_duration_boundaries_are_retained() {
        for (duration, db, expected) in [
            (0.8, -40.0, false),
            (0.799_999, -40.0, true),
            (0.8, -39.999_999, true),
        ] {
            let mut observations = empty_observations(duration);
            observations.activity.push(AudioActivityPoint {
                time_s: duration / 2.0,
                rms: 0.001,
                relative_db: db,
            });
            assert_eq!(
                retained_at(&observations, duration / 2.0).unwrap(),
                Some(expected)
            );
        }
    }

    #[test]
    fn membership_does_not_mutate_the_observation_or_create_tempo() {
        let observations = observe(&vec![0.125; 48_000]).unwrap();
        let before = observations.clone();
        assert_eq!(retained_at(&observations, 1.0).unwrap(), Some(true));
        assert_eq!(observations, before);
    }

    #[test]
    fn invalid_pcm_and_queries_are_refused() {
        for pcm in [vec![0.0; 1024], vec![f32::NAN; 1025], vec![2.0; 1025]] {
            assert!(observe(&pcm).is_err());
        }
        for time in [f64::NAN, -0.1, 2.1] {
            assert!(retained_at(&empty_observations(2.0), time).is_err());
        }
    }
}
