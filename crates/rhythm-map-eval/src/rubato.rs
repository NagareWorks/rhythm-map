use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use rhythm_map_core::TempoSegmentKind;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{GeneratedTruth, TruthBeat, TruthTempoSegment, inspect_audio_asset};

const TIMESTAMP_EPSILON_S: f64 = 0.000_001;

/// One form segment recovered from a RUBATO physical-time annotation.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RubatoStructureSegment {
    /// Segment start in audio seconds.
    pub start_s: f64,
    /// Segment end in audio seconds.
    pub end_s: f64,
    /// Upstream structural label, retained verbatim.
    pub label: String,
}

/// Audit metadata emitted with truth recovered from one RUBATO performance.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RubatoTruthImport {
    /// Generated beat, downbeat, and beat-local tempo truth.
    pub truth: GeneratedTruth,
    /// Number of official beat timestamps imported.
    pub recovered_beats: usize,
    /// Number of official measure starts matched to beats as downbeats.
    pub recovered_downbeats: usize,
    /// Beat intervals retained for timing but omitted from BPM truth because
    /// they imply a tempo above the evaluation schema's 1000 BPM ceiling.
    pub omitted_tempo_intervals: usize,
    /// Terminal beat markers omitted because upstream frame quantization puts
    /// them just beyond the decoded audio boundary.
    pub omitted_out_of_bounds_beats: usize,
    /// Terminal measure markers omitted for the same audio-boundary reason.
    pub omitted_out_of_bounds_measures: usize,
    /// Form segments retained separately from tempo change-point truth.
    pub structure_segments: Vec<RubatoStructureSegment>,
    /// SHA-256 identity of the encoded audio used for duration validation.
    pub audio_sha256: String,
    /// SHA-256 identity of the beat annotation.
    pub beat_annotation_sha256: String,
    /// SHA-256 identity of the measure annotation.
    pub measure_annotation_sha256: String,
    /// SHA-256 identity of the structure annotation.
    pub structure_annotation_sha256: String,
}

#[derive(Debug, Clone, Copy)]
struct TimedMarker {
    time_s: f64,
    musical_position: f64,
}

/// Recover beat, downbeat, and beat-local tempo truth from RUBATO annotations.
///
/// The official beat and measure CSVs are the only timing labels. Measure
/// timestamps must coincide with beat timestamps and mark those beats as
/// downbeats. Structure labels are validated and returned for future section
/// evaluation, but are deliberately not relabeled as tempo change points.
/// Audio decoding supplies only exact duration and content identity.
///
/// # Errors
///
/// Returns an error for malformed annotations, non-monotonic or out-of-range
/// timestamps, measure starts without a matching beat, or undecodable audio.
pub fn import_rubato_truth(
    id: impl Into<String>,
    beat_annotation: impl AsRef<Path>,
    measure_annotation: impl AsRef<Path>,
    structure_annotation: impl AsRef<Path>,
    audio_file: impl AsRef<Path>,
) -> Result<RubatoTruthImport> {
    let (beat_bytes, beat_text) = read_utf8(beat_annotation.as_ref())?;
    let (measure_bytes, measure_text) = read_utf8(measure_annotation.as_ref())?;
    let (structure_bytes, structure_text) = read_utf8(structure_annotation.as_ref())?;
    let beat_markers = parse_timed_markers(&beat_text, "time;beat;audio_frame;mus_frame")?;
    let measure_markers = parse_timed_markers(&measure_text, "time;measure;audio_frame;mus_frame")?;
    let structure_segments = parse_structure(&structure_text)?;
    let audio = inspect_audio_asset(audio_file)?;

    if beat_markers.len() < 3 {
        bail!("RUBATO truth requires at least three beat timestamps");
    }
    validate_marker_order("beat", &beat_markers)?;
    validate_marker_order("measure", &measure_markers)?;
    if measure_markers.is_empty() {
        bail!("RUBATO truth requires at least one measure timestamp");
    }
    if beat_markers
        .last()
        .is_some_and(|marker| marker.time_s > audio.duration_s + 0.1)
        || measure_markers
            .last()
            .is_some_and(|marker| marker.time_s > audio.duration_s + 0.1)
    {
        bail!("RUBATO annotation timing falls outside the decoded audio duration");
    }
    validate_structure(&structure_segments, audio.duration_s)?;

    let omitted_out_of_bounds_beats = beat_markers
        .iter()
        .filter(|marker| marker.time_s > audio.duration_s)
        .count();
    let omitted_out_of_bounds_measures = measure_markers
        .iter()
        .filter(|marker| marker.time_s > audio.duration_s)
        .count();
    let beat_markers = beat_markers
        .into_iter()
        .filter(|marker| marker.time_s <= audio.duration_s)
        .collect::<Vec<_>>();
    let measure_markers = measure_markers
        .into_iter()
        .filter(|marker| marker.time_s <= audio.duration_s)
        .collect::<Vec<_>>();
    if beat_markers.len() < 3 {
        bail!("RUBATO truth has fewer than three in-bounds beat timestamps");
    }

    let downbeats = match_downbeats(&beat_markers, &measure_markers)?;
    let beats = beat_markers
        .iter()
        .zip(downbeats)
        .map(|(marker, downbeat)| TruthBeat {
            time_s: marker.time_s,
            downbeat,
        })
        .collect::<Vec<_>>();
    let (tempo_segments, omitted_tempo_intervals) = build_tempo_segments(&beats);
    let truth = GeneratedTruth {
        schema_version: 1,
        id: id.into(),
        duration_s: audio.duration_s,
        beats,
        tempo_segments,
        change_points: Vec::new(),
    };
    truth.validate().map_err(anyhow::Error::msg)?;

    Ok(RubatoTruthImport {
        recovered_beats: truth.beats.len(),
        recovered_downbeats: truth.beats.iter().filter(|beat| beat.downbeat).count(),
        omitted_tempo_intervals,
        omitted_out_of_bounds_beats,
        omitted_out_of_bounds_measures,
        truth,
        structure_segments,
        audio_sha256: audio.sha256,
        beat_annotation_sha256: sha256_bytes(&beat_bytes),
        measure_annotation_sha256: sha256_bytes(&measure_bytes),
        structure_annotation_sha256: sha256_bytes(&structure_bytes),
    })
}

fn match_downbeats(beats: &[TimedMarker], measures: &[TimedMarker]) -> Result<Vec<bool>> {
    let mut downbeats = vec![false; beats.len()];
    for measure in measures {
        let beat_index = beats
            .iter()
            .position(|beat| (beat.time_s - measure.time_s).abs() <= TIMESTAMP_EPSILON_S)
            .with_context(|| {
                format!(
                    "RUBATO measure {} at {:.9}s has no matching beat timestamp",
                    measure.musical_position, measure.time_s
                )
            })?;
        downbeats[beat_index] = true;
    }
    Ok(downbeats)
}

fn build_tempo_segments(beats: &[TruthBeat]) -> (Vec<TruthTempoSegment>, usize) {
    let mut omitted = 0;
    let segments = beats
        .windows(2)
        .filter_map(|pair| {
            let bpm = round_six(60.0 / (pair[1].time_s - pair[0].time_s));
            if bpm > 1_000.0 {
                omitted += 1;
                return None;
            }
            Some(TruthTempoSegment {
                start_s: pair[0].time_s,
                end_s: pair[1].time_s,
                kind: TempoSegmentKind::Constant,
                start_bpm: bpm,
                end_bpm: bpm,
            })
        })
        .collect();
    (segments, omitted)
}

fn read_utf8(path: &Path) -> Result<(Vec<u8>, String)> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = std::str::from_utf8(&bytes)
        .with_context(|| format!("{} is not UTF-8", path.display()))?
        .trim_start_matches('\u{feff}')
        .to_string();
    Ok((bytes, text))
}

fn parse_timed_markers(input: &str, expected_header: &str) -> Result<Vec<TimedMarker>> {
    let mut lines = input.lines();
    let header = lines
        .next()
        .map(str::trim_end)
        .context("empty RUBATO CSV")?;
    if header != expected_header {
        bail!("unexpected RUBATO CSV header {header:?}; expected {expected_header:?}");
    }
    lines
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            let columns = line.trim_end().split(';').collect::<Vec<_>>();
            if columns.len() != 4 {
                bail!("RUBATO CSV row {} must contain four columns", index + 2);
            }
            let time_s = parse_finite(columns[0], "physical time", index + 2)?;
            let musical_position = parse_finite(columns[1], "musical position", index + 2)?;
            let _audio_frame = parse_finite(columns[2], "audio frame", index + 2)?;
            let _musical_frame = parse_finite(columns[3], "musical frame", index + 2)?;
            Ok(TimedMarker {
                time_s,
                musical_position,
            })
        })
        .collect()
}

fn parse_structure(input: &str) -> Result<Vec<RubatoStructureSegment>> {
    let mut lines = input.lines();
    let header = lines
        .next()
        .map(str::trim_end)
        .context("empty RUBATO structure CSV")?;
    if header != "start;end;structure" {
        bail!("unexpected RUBATO structure CSV header {header:?}");
    }
    lines
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            let columns = line.trim_end().splitn(3, ';').collect::<Vec<_>>();
            if columns.len() != 3 || columns[2].trim().is_empty() {
                bail!(
                    "RUBATO structure row {} must contain start, end, and label",
                    index + 2
                );
            }
            Ok(RubatoStructureSegment {
                start_s: parse_finite(columns[0], "structure start", index + 2)?,
                end_s: parse_finite(columns[1], "structure end", index + 2)?,
                label: columns[2].trim().to_string(),
            })
        })
        .collect()
}

fn parse_finite(value: &str, label: &str, row: usize) -> Result<f64> {
    let value = value
        .parse::<f64>()
        .with_context(|| format!("invalid RUBATO {label} on row {row}"))?;
    if !value.is_finite() {
        bail!("non-finite RUBATO {label} on row {row}");
    }
    Ok(value)
}

fn validate_marker_order(label: &str, markers: &[TimedMarker]) -> Result<()> {
    if markers.iter().any(|marker| marker.time_s < 0.0)
        || markers
            .windows(2)
            .any(|pair| pair[1].time_s <= pair[0].time_s)
    {
        bail!("RUBATO {label} timestamps must be non-negative and strictly increasing");
    }
    Ok(())
}

fn validate_structure(segments: &[RubatoStructureSegment], duration_s: f64) -> Result<()> {
    if segments.is_empty() {
        bail!("RUBATO truth requires at least one structure segment");
    }
    for segment in segments {
        if segment.start_s < 0.0
            || segment.end_s <= segment.start_s
            || segment.end_s > duration_s + 0.1
        {
            bail!("RUBATO structure segment has invalid audio bounds");
        }
    }
    if segments
        .windows(2)
        .any(|pair| pair[1].start_s + TIMESTAMP_EPSILON_S < pair[0].end_s)
    {
        bail!("RUBATO structure segments must be ordered and non-overlapping");
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn round_six(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::import_rubato_truth;

    fn fixture_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("rhythm-map-rubato-{name}-{}", std::process::id()))
    }

    fn wav_bytes(duration_samples: u32) -> Vec<u8> {
        let data_bytes = duration_samples * 2;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&8_000_u32.to_le_bytes());
        bytes.extend_from_slice(&16_000_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        bytes.resize(bytes.len() + data_bytes as usize, 0);
        bytes
    }

    #[test]
    fn imports_beats_downbeats_tempo_and_separate_structure() {
        let root = fixture_root("valid");
        fs::create_dir_all(&root).unwrap();
        let beat = root.join("beat.csv");
        let measure = root.join("measure.csv");
        let structure = root.join("structure.csv");
        let audio = root.join("audio.wav");
        fs::write(
            &beat,
            "time;beat;audio_frame;mus_frame\r\n0.1;1;1;1\r\n0.6;1.25;2;2\r\n1.1;2;3;3\r\n1.6;2.25;4;4\r\n",
        )
        .unwrap();
        fs::write(
            &measure,
            "time;measure;audio_frame;mus_frame\n0.1;1;1;1\n1.1;2;3;3\n",
        )
        .unwrap();
        fs::write(&structure, "start;end;structure\n0.1;1.1;A\n1.1;1.9;B\n").unwrap();
        fs::write(&audio, wav_bytes(16_000)).unwrap();

        let imported = import_rubato_truth("fixture", &beat, &measure, &structure, &audio).unwrap();
        assert_eq!(imported.recovered_beats, 4);
        assert_eq!(imported.recovered_downbeats, 2);
        assert_eq!(imported.omitted_tempo_intervals, 0);
        assert_eq!(imported.omitted_out_of_bounds_beats, 0);
        assert_eq!(imported.omitted_out_of_bounds_measures, 0);
        assert_eq!(
            imported
                .truth
                .beats
                .iter()
                .map(|beat| beat.downbeat)
                .collect::<Vec<_>>(),
            vec![true, false, true, false]
        );
        assert!(
            imported
                .truth
                .tempo_segments
                .iter()
                .all(|segment| (segment.start_bpm - 120.0).abs() < f64::EPSILON)
        );
        assert_eq!(imported.structure_segments[1].label, "B");
        assert!(imported.truth.change_points.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_measure_without_matching_beat() {
        let root = fixture_root("unmatched-measure");
        fs::create_dir_all(&root).unwrap();
        let beat = root.join("beat.csv");
        let measure = root.join("measure.csv");
        let structure = root.join("structure.csv");
        let audio = root.join("audio.wav");
        fs::write(
            &beat,
            "time;beat;audio_frame;mus_frame\n0.1;1;1;1\n0.6;2;2;2\n1.1;3;3;3\n",
        )
        .unwrap();
        fs::write(&measure, "time;measure;audio_frame;mus_frame\n0.2;1;1;1\n").unwrap();
        fs::write(&structure, "start;end;structure\n0.1;1.1;A\n").unwrap();
        fs::write(&audio, wav_bytes(16_000)).unwrap();

        let error = import_rubato_truth("fixture", &beat, &measure, &structure, &audio)
            .unwrap_err()
            .to_string();
        assert!(error.contains("has no matching beat timestamp"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn retains_close_official_beats_without_inventing_extreme_tempo() {
        let root = fixture_root("extreme-interval");
        fs::create_dir_all(&root).unwrap();
        let beat = root.join("beat.csv");
        let measure = root.join("measure.csv");
        let structure = root.join("structure.csv");
        let audio = root.join("audio.wav");
        fs::write(
            &beat,
            "time;beat;audio_frame;mus_frame\n0.1;1;1;1\n0.11;1.5;2;2\n0.6;2;3;3\n",
        )
        .unwrap();
        fs::write(&measure, "time;measure;audio_frame;mus_frame\n0.1;1;1;1\n").unwrap();
        fs::write(&structure, "start;end;structure\n0.1;0.6;A\n").unwrap();
        fs::write(&audio, wav_bytes(8_000)).unwrap();

        let imported = import_rubato_truth("fixture", &beat, &measure, &structure, &audio).unwrap();
        assert_eq!(imported.recovered_beats, 3);
        assert_eq!(imported.omitted_tempo_intervals, 1);
        assert_eq!(imported.truth.tempo_segments.len(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn omits_frame_quantized_terminal_markers_beyond_audio() {
        let root = fixture_root("terminal-marker");
        fs::create_dir_all(&root).unwrap();
        let beat = root.join("beat.csv");
        let measure = root.join("measure.csv");
        let structure = root.join("structure.csv");
        let audio = root.join("audio.wav");
        fs::write(
            &beat,
            "time;beat;audio_frame;mus_frame\n0.1;1;1;1\n0.4;2;2;2\n0.7;3;3;3\n1.01;4;4;4\n",
        )
        .unwrap();
        fs::write(
            &measure,
            "time;measure;audio_frame;mus_frame\n0.1;1;1;1\n1.01;2;4;4\n",
        )
        .unwrap();
        fs::write(&structure, "start;end;structure\n0.1;1.01;A\n").unwrap();
        fs::write(&audio, wav_bytes(8_000)).unwrap();

        let imported = import_rubato_truth("fixture", &beat, &measure, &structure, &audio).unwrap();
        assert_eq!(imported.recovered_beats, 3);
        assert_eq!(imported.recovered_downbeats, 1);
        assert_eq!(imported.omitted_out_of_bounds_beats, 1);
        assert_eq!(imported.omitted_out_of_bounds_measures, 1);
        fs::remove_dir_all(root).unwrap();
    }
}
