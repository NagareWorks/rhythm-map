use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use regex::Regex;
use rhythm_map_core::TempoSegmentKind;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{GeneratedTruth, TruthBeat, TruthTempoSegment, inspect_audio_asset};

const SCORE_EPSILON: f64 = 0.000_01;

/// Audit metadata emitted with truth recovered from one Vienna 4x22 match file.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ViennaTruthImport {
    /// Generated evaluation truth.
    pub truth: GeneratedTruth,
    /// Musical meter used to construct the beat grid.
    pub meter: String,
    /// Beats whose score positions had directly aligned performed notes.
    pub directly_aligned_beats: usize,
    /// Beats recovered by interpolation between aligned score positions.
    pub interpolated_beats: usize,
    /// SHA-256 identity of the encoded audio used for the duration boundary.
    pub audio_sha256: String,
    /// SHA-256 identity of the match annotation used for the timing labels.
    pub match_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Meter {
    numerator: u32,
    denominator: u32,
    score_start: f64,
}

impl Meter {
    fn tactus_step(self) -> f64 {
        if self.numerator > 3 && self.numerator.is_multiple_of(3) {
            3.0
        } else {
            1.0
        }
    }

    fn display(self) -> String {
        format!("{}/{}", self.numerator, self.denominator)
    }
}

#[derive(Debug, Clone, Copy)]
struct Alignment {
    score_position: f64,
    onset_tick: f64,
}

#[derive(Debug)]
struct ParsedMatch {
    ticks_per_quarter: f64,
    microseconds_per_quarter: f64,
    meter: Meter,
    alignments: Vec<Alignment>,
}

/// Recover expressive beat/downbeat and beat-local tempo truth from a Vienna
/// 4x22 score-performance match file.
///
/// Match annotations provide the only timing labels. Notes sharing one score
/// onset are reduced to their median performance onset. A missing musical beat
/// is interpolated only when aligned score positions exist on both sides. The
/// audio is decoded solely to bind the truth to exact bytes and duration.
///
/// # Errors
///
/// Returns an error for an unsupported match file, invalid/non-monotonic timing,
/// insufficient bilateral alignment evidence, or undecodable audio.
pub fn import_vienna_truth(
    id: impl Into<String>,
    match_file: impl AsRef<Path>,
    audio_file: impl AsRef<Path>,
) -> Result<ViennaTruthImport> {
    let match_file = match_file.as_ref();
    let match_bytes =
        fs::read(match_file).with_context(|| format!("failed to read {}", match_file.display()))?;
    let match_text = std::str::from_utf8(&match_bytes)
        .with_context(|| format!("{} is not UTF-8", match_file.display()))?;
    let parsed = parse_match(match_text)?;
    let audio = inspect_audio_asset(audio_file)?;
    let (beats, directly_aligned_beats, interpolated_beats) = recover_beats(&parsed)?;

    if beats.len() < 3 {
        bail!("Vienna truth requires at least three recovered musical beats");
    }
    if beats
        .last()
        .is_some_and(|beat| beat.time_s > audio.duration_s + 0.1)
    {
        bail!("Vienna match timing falls outside the decoded audio duration");
    }

    let tempo_segments = beats
        .windows(2)
        .map(|pair| {
            let interval = pair[1].time_s - pair[0].time_s;
            let bpm = round_three(60.0 / interval);
            TruthTempoSegment {
                start_s: pair[0].time_s,
                end_s: pair[1].time_s,
                kind: TempoSegmentKind::Constant,
                start_bpm: bpm,
                end_bpm: bpm,
            }
        })
        .collect::<Vec<_>>();

    let truth = GeneratedTruth {
        schema_version: 1,
        id: id.into(),
        duration_s: audio.duration_s,
        beats,
        tempo_segments,
        // Expressive beat-to-beat rubato is tempo metadata, not evidence of a
        // discrete structural change. Vienna does not publish such labels.
        change_points: Vec::new(),
    };
    truth.validate().map_err(anyhow::Error::msg)?;

    Ok(ViennaTruthImport {
        truth,
        meter: parsed.meter.display(),
        directly_aligned_beats,
        interpolated_beats,
        audio_sha256: audio.sha256,
        match_sha256: format!("{:x}", Sha256::digest(match_bytes)),
    })
}

fn parse_match(input: &str) -> Result<ParsedMatch> {
    let clock_units = Regex::new(r"^info\(midiClockUnits,(\d+)\)\.$")?;
    let clock_rate = Regex::new(r"^info\(midiClockRate,(\d+)\)\.$")?;
    let time_signature =
        Regex::new(r"^scoreprop\(timeSignature,(\d+)/(\d+),[^,]+,[^,]+,(-?\d+(?:\.\d+)?)\)\.$")?;
    let aligned_note = Regex::new(
        r"^snote\(.*?,(-?\d+(?:\.\d+)?),-?\d+(?:\.\d+)?,\[[^]]*\]\)-note\([^,]+,[^,]+,(-?\d+),-?\d+,",
    )?;

    let mut ticks_per_quarter = None;
    let mut microseconds_per_quarter = None;
    let mut meters = Vec::new();
    let mut raw_alignments = Vec::new();

    for line in input.lines().map(str::trim) {
        if let Some(capture) = clock_units.captures(line) {
            ticks_per_quarter = Some(capture[1].parse::<f64>()?);
        } else if let Some(capture) = clock_rate.captures(line) {
            microseconds_per_quarter = Some(capture[1].parse::<f64>()?);
        } else if let Some(capture) = time_signature.captures(line) {
            meters.push(Meter {
                numerator: capture[1].parse()?,
                denominator: capture[2].parse()?,
                score_start: capture[3].parse()?,
            });
        } else if let Some(capture) = aligned_note.captures(line) {
            raw_alignments.push(Alignment {
                score_position: capture[1].parse()?,
                onset_tick: capture[2].parse()?,
            });
        }
    }

    let ticks_per_quarter = ticks_per_quarter.context("missing midiClockUnits")?;
    let microseconds_per_quarter = microseconds_per_quarter.context("missing midiClockRate")?;
    if ticks_per_quarter <= 0.0 || microseconds_per_quarter <= 0.0 {
        bail!("MIDI clock values must be positive");
    }
    if meters.len() != 1 {
        bail!("Vienna importer currently requires exactly one time signature");
    }
    let meter = meters[0];
    if meter.numerator == 0 || meter.denominator == 0 {
        bail!("time signature values must be positive");
    }
    if raw_alignments.len() < 3 {
        bail!("match file contains fewer than three aligned score onsets");
    }

    raw_alignments.sort_by(|left, right| left.score_position.total_cmp(&right.score_position));
    let alignments = median_onsets_by_score_position(&raw_alignments);
    if alignments
        .windows(2)
        .any(|pair| pair[1].onset_tick <= pair[0].onset_tick)
    {
        bail!("median score-performance alignment is not strictly monotonic");
    }

    Ok(ParsedMatch {
        ticks_per_quarter,
        microseconds_per_quarter,
        meter,
        alignments,
    })
}

fn median_onsets_by_score_position(raw: &[Alignment]) -> Vec<Alignment> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < raw.len() {
        let score_position = raw[index].score_position;
        let mut end = index + 1;
        while end < raw.len() && (raw[end].score_position - score_position).abs() <= SCORE_EPSILON {
            end += 1;
        }
        let mut ticks = raw[index..end]
            .iter()
            .map(|alignment| alignment.onset_tick)
            .collect::<Vec<_>>();
        ticks.sort_by(f64::total_cmp);
        result.push(Alignment {
            score_position,
            onset_tick: median(&ticks),
        });
        index = end;
    }
    result
}

fn recover_beats(parsed: &ParsedMatch) -> Result<(Vec<TruthBeat>, usize, usize)> {
    let minimum = parsed
        .alignments
        .first()
        .context("missing first alignment")?
        .score_position;
    let maximum = parsed
        .alignments
        .last()
        .context("missing last alignment")?
        .score_position;
    let mut targets = Vec::new();
    let step = parsed.meter.tactus_step();

    let mut pickup = parsed.meter.score_start;
    while pickup < -SCORE_EPSILON {
        if pickup >= minimum - SCORE_EPSILON {
            targets.push((pickup, false));
        }
        pickup += step;
    }

    let mut score_position = 0.0;
    while score_position <= maximum + SCORE_EPSILON {
        let measure_index = (score_position / f64::from(parsed.meter.numerator)).round();
        let downbeat = (score_position - measure_index * f64::from(parsed.meter.numerator)).abs()
            <= SCORE_EPSILON;
        targets.push((score_position, downbeat));
        score_position += step;
    }

    let seconds_per_tick =
        parsed.microseconds_per_quarter / (parsed.ticks_per_quarter * 1_000_000.0);
    let mut beats = Vec::new();
    let mut directly_aligned = 0;
    let mut interpolated = 0;
    for (target, downbeat) in targets {
        let (tick, direct) = interpolate_tick(&parsed.alignments, target)?;
        let time_s = round_six(tick * seconds_per_tick);
        if time_s < 0.0 {
            bail!("recovered beat occurs before the audio origin");
        }
        if beats
            .last()
            .is_some_and(|previous: &TruthBeat| time_s <= previous.time_s)
        {
            bail!("recovered beat timestamps are not strictly increasing");
        }
        beats.push(TruthBeat { time_s, downbeat });
        if direct {
            directly_aligned += 1;
        } else {
            interpolated += 1;
        }
    }
    Ok((beats, directly_aligned, interpolated))
}

fn interpolate_tick(alignments: &[Alignment], target: f64) -> Result<(f64, bool)> {
    let right =
        alignments.partition_point(|alignment| alignment.score_position < target - SCORE_EPSILON);
    if let Some(exact) = alignments.get(right)
        && (exact.score_position - target).abs() <= SCORE_EPSILON
    {
        return Ok((exact.onset_tick, true));
    }
    if right == 0 || right == alignments.len() {
        bail!("musical beat at score position {target} lacks bilateral alignment evidence");
    }
    let left = alignments[right - 1];
    let right = alignments[right];
    let ratio = (target - left.score_position) / (right.score_position - left.score_position);
    Ok((
        left.onset_tick + ratio * (right.onset_tick - left.onset_tick),
        false,
    ))
}

fn median(values: &[f64]) -> f64 {
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f64::midpoint(values[middle - 1], values[middle])
    } else {
        values[middle]
    }
}

fn round_six(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn round_three(value: f64) -> f64 {
    (value * 1_000.0).round() / 1_000.0
}

#[cfg(test)]
mod tests {
    use super::{parse_match, recover_beats};

    const SIMPLE: &str = r"
info(midiClockUnits,480).
info(midiClockRate,500000).
scoreprop(timeSignature,3/4,0:1,0,-1.0000).
snote(n1,[C,n],4,0:1,0,1/4,-1.0000,0.0000,[v1])-note(n1,60,0,200,80,0,0).
snote(n2,[C,n],4,1:1,0,1/4,0.0000,1.0000,[v1])-note(n2,60,480,700,80,0,0).
snote(n3,[E,n],4,1:1,0,1/4,0.0000,1.0000,[v2])-note(n3,64,500,700,80,0,0).
snote(n4,[C,n],4,1:3,0,1/4,2.0000,3.0000,[v1])-note(n4,60,1440,1600,80,0,0).
snote(n5,[C,n],4,2:1,0,1/4,3.0000,4.0000,[v1])-note(n5,60,1920,2100,80,0,0).
";

    const COMPOUND: &str = r"
info(midiClockUnits,480).
info(midiClockRate,500000).
scoreprop(timeSignature,6/8,0:1,0,-4.0000).
snote(n1,[C,n],4,0:1,0,1/8,-4.0000,-3.0000,[v1])-note(n1,60,0,200,80,0,0).
snote(n2,[C,n],4,0:4,0,1/8,-1.0000,0.0000,[v1])-note(n2,60,720,900,80,0,0).
snote(n3,[C,n],4,1:1,0,1/8,0.0000,1.0000,[v1])-note(n3,60,960,1100,80,0,0).
snote(n4,[C,n],4,1:4,0,1/8,3.0000,4.0000,[v1])-note(n4,60,1680,1800,80,0,0).
snote(n5,[C,n],4,2:1,0,1/8,6.0000,7.0000,[v1])-note(n5,60,2400,2600,80,0,0).
";

    #[test]
    fn recovers_simple_meter_pickup_and_interpolated_beat() {
        let parsed = parse_match(SIMPLE).expect("parse simple match");
        let (beats, direct, interpolated) = recover_beats(&parsed).expect("recover beats");
        assert_eq!(parsed.meter.display(), "3/4");
        assert_eq!(beats.len(), 5);
        assert_eq!(direct, 4);
        assert_eq!(interpolated, 1);
        assert!(!beats[0].downbeat);
        assert!(beats[1].downbeat);
        assert!((beats[1].time_s - 0.510_417).abs() < 0.000_001);
        assert!((beats[2].time_s - 1.005_208).abs() < 0.000_001);
        assert!(beats[4].downbeat);
    }

    #[test]
    fn groups_six_eight_into_two_tactus_beats_per_bar() {
        let parsed = parse_match(COMPOUND).expect("parse compound match");
        let (beats, direct, interpolated) = recover_beats(&parsed).expect("recover beats");
        assert_eq!(beats.len(), 5);
        assert_eq!(direct, 5);
        assert_eq!(interpolated, 0);
        assert_eq!(
            beats.iter().map(|beat| beat.downbeat).collect::<Vec<_>>(),
            vec![false, false, true, false, true]
        );
    }

    #[test]
    fn rejects_non_monotonic_alignment() {
        let invalid = SIMPLE.replace(
            "snote(n5,[C,n],4,2:1,0,1/4,3.0000,4.0000,[v1])-note(n5,60,1920",
            "snote(n5,[C,n],4,2:1,0,1/4,3.0000,4.0000,[v1])-note(n5,60,1200",
        );
        assert!(parse_match(&invalid).is_err());
    }

    #[test]
    fn parses_every_pinned_vienna_match_file() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../evaluation/datasets/sources/vienna4x22/match");
        let mut count = 0;
        for entry in std::fs::read_dir(&directory).expect("read pinned Vienna match directory") {
            let path = entry.expect("directory entry").path();
            if path.extension().and_then(std::ffi::OsStr::to_str) != Some("match") {
                continue;
            }
            let source = std::fs::read_to_string(&path).expect("read match file");
            let parsed = parse_match(&source)
                .unwrap_or_else(|error| panic!("failed to parse {}: {error:#}", path.display()));
            let (beats, _, _) = recover_beats(&parsed)
                .unwrap_or_else(|error| panic!("failed to recover {}: {error:#}", path.display()));
            assert!(beats.len() >= 3, "too few beats in {}", path.display());
            count += 1;
        }
        assert_eq!(count, 88);
    }
}
