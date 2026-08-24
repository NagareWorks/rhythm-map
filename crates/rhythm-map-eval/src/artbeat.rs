use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use regex::Regex;
use rhythm_map_core::TempoSegmentKind;
use serde::Serialize;

use crate::{GeneratedTruth, TruthBeat, TruthTempoSegment, inspect_audio_asset};

/// Audit metadata emitted alongside truth recovered from one ARTBeaT SVG.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ArtbeatTruthImport {
    /// Generated evaluation truth.
    pub truth: GeneratedTruth,
    /// Number of official beat lines recovered from the SVG.
    pub recovered_beats: usize,
    /// Median tempo calculated only from consecutive official beat timestamps.
    pub median_bpm: f64,
    /// SHA-256 identity of the encoded audio used for the duration boundary.
    pub audio_sha256: String,
}

/// Recover ARTBeaT's plotted `Ground Truth` beat lines from an immutable SVG.
///
/// The importer does not inspect audio features or model output. The SVG time
/// axis maps the selected vertical-line coordinates back to seconds; the audio
/// file supplies only its decoded duration and content identity.
///
/// # Errors
///
/// Returns an error when the SVG layout is unsupported, beat timestamps are
/// invalid, or the decoded audio cannot be inspected.
pub fn import_artbeat_truth(
    id: impl Into<String>,
    annotation: impl AsRef<Path>,
    audio: impl AsRef<Path>,
) -> Result<ArtbeatTruthImport> {
    let annotation = annotation.as_ref();
    let svg = fs::read_to_string(annotation)
        .with_context(|| format!("failed to read {}", annotation.display()))?;
    let ticks = parse_ticks(&svg)?;
    let beat_x = parse_ground_truth_x(&svg)?;
    let (slope, intercept) = axis_mapping(&ticks)?;

    let audio = inspect_audio_asset(audio)?;
    let mut times = beat_x
        .into_iter()
        .map(|x| round_six(slope.mul_add(x, intercept)))
        .collect::<Vec<_>>();
    times.dedup_by(|left, right| (*left - *right).abs() < 0.000_001);
    if times.len() < 3 {
        bail!("ARTBeaT truth must contain at least three distinct beats");
    }
    if times[0] < -0.001 || times.last().copied().unwrap_or_default() > audio.duration_s + 0.1 {
        bail!("ARTBeaT beat coordinates fall outside the decoded audio duration");
    }
    for time in &mut times {
        if time.abs() < 0.001 {
            *time = 0.0;
        }
    }

    let mut bpms = times
        .windows(2)
        .filter_map(|pair| {
            let interval = pair[1] - pair[0];
            (interval > 0.0).then_some(60.0 / interval)
        })
        .collect::<Vec<_>>();
    bpms.sort_by(f64::total_cmp);
    let median_bpm = round_three(median(&bpms));
    let truth = GeneratedTruth {
        schema_version: 1,
        id: id.into(),
        duration_s: audio.duration_s,
        beats: times
            .into_iter()
            .map(|time_s| TruthBeat {
                time_s,
                downbeat: false,
            })
            .collect(),
        tempo_segments: vec![TruthTempoSegment {
            start_s: 0.0,
            end_s: audio.duration_s,
            kind: TempoSegmentKind::Constant,
            start_bpm: median_bpm,
            end_bpm: median_bpm,
        }],
        change_points: Vec::new(),
    };
    truth.validate().map_err(anyhow::Error::msg)?;
    Ok(ArtbeatTruthImport {
        recovered_beats: truth.beats.len(),
        truth,
        median_bpm,
        audio_sha256: audio.sha256,
    })
}

fn parse_ticks(svg: &str) -> Result<Vec<(f64, f64)>> {
    let tick = Regex::new(
        r#"(?s)<g id="xtick_\d+">.*?<use[^>]* x="([0-9.]+)"[^>]*>?.*?<!--\s*(-?[0-9]+(?:\.[0-9]+)?)\s*-->"#,
    )?;
    let ticks = tick
        .captures_iter(svg)
        .filter_map(|capture| {
            Some((
                capture.get(1)?.as_str().parse().ok()?,
                capture.get(2)?.as_str().parse().ok()?,
            ))
        })
        .collect::<Vec<_>>();
    if ticks.len() < 2 {
        bail!("ARTBeaT SVG has no usable numeric time-axis ticks");
    }
    Ok(ticks)
}

fn parse_ground_truth_x(svg: &str) -> Result<Vec<f64>> {
    let line = Regex::new(
        r#"(?s)<g id="line2d_\d+">\s*<path d="M\s+([0-9.]+)\s+([0-9.]+)\s+L\s+([0-9.]+)\s+([0-9.]+)[^"]*"[^>]*style="[^"]*stroke:\s*#0000ff[^"]*"[^>]*/?>\s*</g>"#,
    )?;
    let mut groups: Vec<(f64, f64, Vec<f64>)> = Vec::new();
    for capture in line.captures_iter(svg) {
        let x1: f64 = capture[1].parse()?;
        let y1: f64 = capture[2].parse()?;
        let x2: f64 = capture[3].parse()?;
        let y2: f64 = capture[4].parse()?;
        if (x1 - x2).abs() > 0.000_001 || (y1 - y2).abs() < 1.0 {
            continue;
        }
        if let Some((_, _, values)) = groups.iter_mut().find(|(group_y1, group_y2, _)| {
            (*group_y1 - y1).abs() < 0.000_001 && (*group_y2 - y2).abs() < 0.000_001
        }) {
            values.push(x1);
        } else {
            groups.push((y1, y2, vec![x1]));
        }
    }
    let (_, _, mut beats) = groups
        .into_iter()
        .max_by_key(|(_, _, values)| values.len())
        .context("ARTBeaT SVG has no vertical blue Ground Truth line group")?;
    beats.sort_by(f64::total_cmp);
    Ok(beats)
}

fn axis_mapping(ticks: &[(f64, f64)]) -> Result<(f64, f64)> {
    let first = ticks[0];
    let second = ticks
        .iter()
        .copied()
        .find(|tick| (tick.0 - first.0).abs() > 0.000_001 && (tick.1 - first.1).abs() > 0.000_001)
        .context("ARTBeaT SVG time-axis ticks do not define a scale")?;
    let slope = (second.1 - first.1) / (second.0 - first.0);
    let intercept = first.1 - slope * first.0;
    if !slope.is_finite() || slope <= 0.0 || !intercept.is_finite() {
        bail!("ARTBeaT SVG has an invalid time-axis mapping");
    }
    Ok((slope, intercept))
}

fn median(values: &[f64]) -> f64 {
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
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
    use super::{axis_mapping, parse_ground_truth_x, parse_ticks};

    const SVG: &str = r#"
      <g id="xtick_1"><g><use x="10" y="30"/></g><g><!-- 0 --></g></g>
      <g id="xtick_2"><g><use x="30" y="30"/></g><g><!-- 2 --></g></g>
      <g id="line2d_1"><path d="M 10 20 L 10 5 " style="stroke: #0000ff"/></g>
      <g id="line2d_2"><path d="M 20 20 L 20 5 " style="stroke: #0000ff"/></g>
      <g id="line2d_3"><path d="M 30 20 L 30 5 " style="stroke: #0000ff"/></g>
      <g id="line2d_4"><path d="M 12 40 L 12 25 " style="stroke: #0000ff"/></g>
    "#;

    #[test]
    fn recovers_largest_vertical_blue_group_and_axis_scale() {
        let ticks = parse_ticks(SVG).expect("ticks");
        let beats = parse_ground_truth_x(SVG).expect("beats");
        let mapping = axis_mapping(&ticks).expect("mapping");
        assert_eq!(beats, vec![10.0, 20.0, 30.0]);
        assert!((mapping.0 - 0.1).abs() < f64::EPSILON);
        assert!((mapping.1 + 1.0).abs() < f64::EPSILON);
    }
}
