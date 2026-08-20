use rhythm_map_core::{ChangeKind, ModelInfo, ObservedBeat, RhythmObservations, TempoSegmentKind};
use serde::{Deserialize, Serialize};

/// Versioned recipe for a deterministic click-track fixture.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyntheticRecipe {
    /// Recipe schema version.
    pub schema_version: u32,
    /// Stable fixture identifier.
    pub id: String,
    /// Rendered WAV sample rate.
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    /// Number of beats in one bar for synthetic downbeat labels.
    #[serde(default = "default_beats_per_bar")]
    pub beats_per_bar: u32,
    /// Ordered timing segments.
    pub segments: Vec<RecipeSegment>,
}

/// One consecutive region of a synthetic timing recipe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecipeSegment {
    /// Segment duration in seconds.
    pub duration_s: f64,
    /// Tempo behavior in this region.
    #[serde(flatten)]
    pub shape: SegmentShape,
}

/// Analytic tempo function used to render exact beat timestamps.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SegmentShape {
    /// Fixed tempo.
    Constant {
        /// Tempo in beats per minute.
        bpm: f64,
    },
    /// Tempo varying linearly with elapsed seconds.
    LinearRamp {
        /// Tempo at the segment start.
        start_bpm: f64,
        /// Tempo at the segment end.
        end_bpm: f64,
    },
    /// No beat events; musical phase restarts after the gap.
    Silence,
}

/// Exact beat label generated from a recipe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TruthBeat {
    /// Timestamp in seconds.
    pub time_s: f64,
    /// Synthetic downbeat label.
    pub downbeat: bool,
}

/// Exact tempo region generated from a recipe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TruthTempoSegment {
    /// Region start in seconds.
    pub start_s: f64,
    /// Region end in seconds.
    pub end_s: f64,
    /// Constant or ramp shape.
    pub kind: TempoSegmentKind,
    /// Tempo at the region start.
    pub start_bpm: f64,
    /// Tempo at the region end.
    pub end_bpm: f64,
}

/// Exact transition label generated from a recipe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TruthChangePoint {
    /// Transition timestamp.
    pub time_s: f64,
    /// Expected transition type.
    pub kind: ChangeKind,
}

/// Complete deterministic ground truth for one recipe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeneratedTruth {
    /// Truth schema version.
    pub schema_version: u32,
    /// Recipe identifier.
    pub id: String,
    /// Total rendered duration.
    pub duration_s: f64,
    /// Exact beat timestamps.
    pub beats: Vec<TruthBeat>,
    /// Exact non-silent tempo functions.
    pub tempo_segments: Vec<TruthTempoSegment>,
    /// Exact timing transitions.
    pub change_points: Vec<TruthChangePoint>,
}

impl GeneratedTruth {
    /// Convert exact labels into ideal backend observations for core evaluation.
    #[must_use]
    pub fn ideal_observations(&self) -> RhythmObservations {
        RhythmObservations {
            duration_s: self.duration_s,
            beats: self
                .beats
                .iter()
                .map(|beat| ObservedBeat {
                    time_s: beat.time_s,
                    confidence: 1.0,
                    downbeat_confidence: if beat.downbeat { 1.0 } else { 0.0 },
                })
                .collect(),
            source: ModelInfo {
                backend: "evaluation".to_string(),
                model: "ideal_observations".to_string(),
                version: Some("1".to_string()),
                frame_rate_hz: None,
            },
        }
    }

    /// Return the exact tempo at a timestamp, or `None` during silence.
    #[must_use]
    pub fn tempo_at(&self, time_s: f64) -> Option<f64> {
        self.tempo_segments
            .iter()
            .find(|segment| time_s >= segment.start_s && time_s <= segment.end_s)
            .map(|segment| {
                if segment.kind == TempoSegmentKind::Constant || segment.end_s <= segment.start_s {
                    segment.start_bpm
                } else {
                    let ratio = (time_s - segment.start_s) / (segment.end_s - segment.start_s);
                    segment.start_bpm + ratio * (segment.end_bpm - segment.start_bpm)
                }
            })
    }
}

/// Generate analytic ground truth without rendering or decoding audio.
///
/// # Errors
///
/// Returns a description for invalid versions, durations, tempi, or an empty
/// recipe.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn generate_truth(recipe: &SyntheticRecipe) -> Result<GeneratedTruth, String> {
    validate_recipe(recipe)?;
    let mut beats = Vec::new();
    let mut tempo_segments = Vec::new();
    let mut starts = Vec::with_capacity(recipe.segments.len());
    let mut cursor = 0.0;
    let mut beat_index = 0_u64;

    for segment in &recipe.segments {
        starts.push(cursor);
        match segment.shape {
            SegmentShape::Constant { bpm } => {
                tempo_segments.push(TruthTempoSegment {
                    start_s: cursor,
                    end_s: cursor + segment.duration_s,
                    kind: TempoSegmentKind::Constant,
                    start_bpm: bpm,
                    end_bpm: bpm,
                });
                let count = (segment.duration_s * bpm / 60.0).floor() as u64;
                for local_index in 0..=count {
                    let local_time = 60.0 * local_index as f64 / bpm;
                    if local_time < segment.duration_s - 1e-9 {
                        push_beat(
                            &mut beats,
                            cursor + local_time,
                            beat_index,
                            recipe.beats_per_bar,
                        );
                        beat_index += 1;
                    }
                }
            }
            SegmentShape::LinearRamp { start_bpm, end_bpm } => {
                tempo_segments.push(TruthTempoSegment {
                    start_s: cursor,
                    end_s: cursor + segment.duration_s,
                    kind: TempoSegmentKind::Ramp,
                    start_bpm,
                    end_bpm,
                });
                let total_beats = segment.duration_s * (start_bpm + end_bpm) / 120.0;
                for local_index in 0..=(total_beats.floor() as u64) {
                    let local_time = invert_linear_phase(
                        local_index as f64,
                        start_bpm,
                        end_bpm,
                        segment.duration_s,
                    );
                    if local_time < segment.duration_s - 1e-9 {
                        push_beat(
                            &mut beats,
                            cursor + local_time,
                            beat_index,
                            recipe.beats_per_bar,
                        );
                        beat_index += 1;
                    }
                }
            }
            SegmentShape::Silence => {}
        }
        cursor += segment.duration_s;
    }

    let change_points = derive_change_points(recipe, &starts);
    Ok(GeneratedTruth {
        schema_version: 1,
        id: recipe.id.clone(),
        duration_s: cursor,
        beats,
        tempo_segments,
        change_points,
    })
}

fn validate_recipe(recipe: &SyntheticRecipe) -> Result<(), String> {
    if recipe.schema_version != 1 {
        return Err(format!(
            "unsupported synthetic recipe schema {}",
            recipe.schema_version
        ));
    }
    if recipe.id.trim().is_empty() || recipe.segments.is_empty() {
        return Err("recipe id and at least one segment are required".to_string());
    }
    if recipe.sample_rate < 8_000 || recipe.beats_per_bar == 0 {
        return Err("sample rate must be at least 8000 and beats_per_bar non-zero".to_string());
    }
    for segment in &recipe.segments {
        if !segment.duration_s.is_finite()
            || segment.duration_s <= 0.0
            || segment.duration_s > 3_600.0
        {
            return Err(
                "segment durations must be finite, positive, and at most one hour".to_string(),
            );
        }
        match segment.shape {
            SegmentShape::Constant { bpm } => validate_bpm(bpm)?,
            SegmentShape::LinearRamp { start_bpm, end_bpm } => {
                validate_bpm(start_bpm)?;
                validate_bpm(end_bpm)?;
            }
            SegmentShape::Silence => {}
        }
    }
    Ok(())
}

fn validate_bpm(bpm: f64) -> Result<(), String> {
    if !bpm.is_finite() || bpm <= 0.0 || bpm > 1_000.0 {
        return Err("tempo must be finite, positive, and at most 1000 BPM".to_string());
    }
    Ok(())
}

fn push_beat(beats: &mut Vec<TruthBeat>, time_s: f64, index: u64, beats_per_bar: u32) {
    if beats
        .last()
        .is_some_and(|previous| (previous.time_s - time_s).abs() < 1e-9)
    {
        return;
    }
    beats.push(TruthBeat {
        time_s,
        downbeat: index.is_multiple_of(u64::from(beats_per_bar)),
    });
}

fn invert_linear_phase(beats: f64, start_bpm: f64, end_bpm: f64, duration_s: f64) -> f64 {
    let slope = (end_bpm - start_bpm) / duration_s;
    if slope.abs() < 1e-12 {
        return 60.0 * beats / start_bpm;
    }
    (-start_bpm + (start_bpm.mul_add(start_bpm, 120.0 * slope * beats)).sqrt()) / slope
}

fn derive_change_points(recipe: &SyntheticRecipe, starts: &[f64]) -> Vec<TruthChangePoint> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < recipe.segments.len() {
        if matches!(recipe.segments[index].shape, SegmentShape::Silence) {
            let first = index;
            while index + 1 < recipe.segments.len()
                && matches!(recipe.segments[index + 1].shape, SegmentShape::Silence)
            {
                index += 1;
            }
            let end = starts[index] + recipe.segments[index].duration_s;
            result.push(TruthChangePoint {
                time_s: (starts[first] + end) * 0.5,
                kind: ChangeKind::RhythmDiscontinuity,
            });
            index += 1;
            continue;
        }
        if index > 0 && !matches!(recipe.segments[index - 1].shape, SegmentShape::Silence) {
            let before = &recipe.segments[index - 1].shape;
            let after = &recipe.segments[index].shape;
            let before_bpm = end_bpm(before);
            let after_bpm = start_bpm(after);
            let kind = if (after_bpm / before_bpm).ln().abs() > 0.01_f64.ln_1p() {
                Some(ChangeKind::TempoJump)
            } else if matches!(before, SegmentShape::LinearRamp { .. })
                != matches!(after, SegmentShape::LinearRamp { .. })
            {
                Some(ChangeKind::RampBoundary)
            } else {
                None
            };
            if let Some(kind) = kind {
                result.push(TruthChangePoint {
                    time_s: starts[index],
                    kind,
                });
            }
        }
        index += 1;
    }
    result
}

fn start_bpm(shape: &SegmentShape) -> f64 {
    match *shape {
        SegmentShape::Constant { bpm } => bpm,
        SegmentShape::LinearRamp { start_bpm, .. } => start_bpm,
        SegmentShape::Silence => 0.0,
    }
}

fn end_bpm(shape: &SegmentShape) -> f64 {
    match *shape {
        SegmentShape::Constant { bpm } => bpm,
        SegmentShape::LinearRamp { end_bpm, .. } => end_bpm,
        SegmentShape::Silence => 0.0,
    }
}

const fn default_sample_rate() -> u32 {
    44_100
}

const fn default_beats_per_bar() -> u32 {
    4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_ramp_has_analytic_beats_and_boundaries() {
        let recipe = SyntheticRecipe {
            schema_version: 1,
            id: "ramp".to_string(),
            sample_rate: 44_100,
            beats_per_bar: 4,
            segments: vec![
                RecipeSegment {
                    duration_s: 10.0,
                    shape: SegmentShape::Constant { bpm: 96.0 },
                },
                RecipeSegment {
                    duration_s: 10.0,
                    shape: SegmentShape::LinearRamp {
                        start_bpm: 96.0,
                        end_bpm: 144.0,
                    },
                },
                RecipeSegment {
                    duration_s: 10.0,
                    shape: SegmentShape::Constant { bpm: 144.0 },
                },
            ],
        };
        let truth = generate_truth(&recipe).unwrap();
        assert_eq!(truth.beats.len(), 60);
        assert_eq!(truth.change_points.len(), 2);
        assert_eq!(truth.change_points[0].kind, ChangeKind::RampBoundary);
        assert!((truth.tempo_at(15.0).unwrap() - 120.0).abs() < 1e-9);
    }
}
