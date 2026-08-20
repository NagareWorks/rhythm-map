use serde::{Deserialize, Serialize};

/// Current serialized analysis schema version.
pub const ANALYSIS_SCHEMA_VERSION: u32 = 1;

/// Identity and timing contract of an observation backend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    /// Backend implementation name.
    pub backend: String,
    /// Human-readable model identity.
    pub model: String,
    /// Optional immutable model version or checksum.
    pub version: Option<String>,
    /// Frame rate of model activations, when available.
    pub frame_rate_hz: Option<f64>,
}

/// A beat observation produced by a model backend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObservedBeat {
    /// Time from the beginning of the decoded audio.
    pub time_s: f64,
    /// Beat confidence in `[0, 1]`.
    pub confidence: f64,
    /// Downbeat confidence in `[0, 1]`.
    pub downbeat_confidence: f64,
}

/// Deterministic short-time audio activity measurement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioActivityPoint {
    /// Center of the analysis window in seconds.
    pub time_s: f64,
    /// Root-mean-square signal level before normalization.
    pub rms: f64,
    /// Signal level in decibels relative to the loudest window.
    pub relative_db: f64,
}

/// Backend-neutral observations consumed by the timing estimator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RhythmObservations {
    /// Audio duration in seconds.
    pub duration_s: f64,
    /// Sorted beat observations.
    pub beats: Vec<ObservedBeat>,
    /// Optional deterministic activity envelope derived from decoded PCM.
    #[serde(default)]
    pub activity: Vec<AudioActivityPoint>,
    /// Source model metadata.
    pub source: ModelInfo,
}

/// Beat event enriched by the timing analysis layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeatEvent {
    /// Beat timestamp in seconds.
    pub time_s: f64,
    /// Observation confidence.
    pub confidence: f64,
    /// Whether the event is classified as a downbeat.
    pub downbeat: bool,
    /// Downbeat confidence.
    pub downbeat_confidence: f64,
}

/// A sampled point on the regularized tempo curve.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TempoPoint {
    /// Timestamp in seconds.
    pub time_s: f64,
    /// Regularized tempo in beats per minute.
    pub bpm: f64,
    /// Local curve confidence in `[0, 1]`.
    pub confidence: f64,
}

/// Shape of a tempo segment.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TempoSegmentKind {
    /// Tempo is stable within the configured tolerance.
    Constant,
    /// Tempo changes continuously across the segment.
    Ramp,
}

/// Piecewise representation of the tempo curve.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TempoSegment {
    /// Inclusive segment start.
    pub start_s: f64,
    /// Exclusive segment end, except for the final segment.
    pub end_s: f64,
    /// Segment shape.
    pub kind: TempoSegmentKind,
    /// Tempo at the segment start.
    pub start_bpm: f64,
    /// Tempo at the segment end.
    pub end_bpm: f64,
    /// Aggregate segment confidence.
    pub confidence: f64,
}

/// Type of a detected timing or rhythm transition.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    /// Sustained discontinuous tempo change.
    TempoJump,
    /// Start or end of a continuous tempo ramp.
    RampBoundary,
    /// Long gap or loss of reliable beat continuity.
    RhythmDiscontinuity,
}

/// A scored change point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChangePoint {
    /// Change timestamp.
    pub time_s: f64,
    /// Change type.
    pub kind: ChangeKind,
    /// Detection score in `[0, 1]`.
    pub score: f64,
    /// Tempo immediately before the change, when defined.
    pub before_bpm: Option<f64>,
    /// Tempo immediately after the change, when defined.
    pub after_bpm: Option<f64>,
}

/// A tempo/rhythm-homogeneous region.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RhythmSection {
    /// Section start.
    pub start_s: f64,
    /// Section end.
    pub end_s: f64,
    /// Representative section tempo.
    pub bpm: Option<f64>,
    /// Tempo stability in `[0, 1]`.
    pub stability: f64,
    /// Number of detected beats inside the section.
    pub beat_count: usize,
}

/// Octave-related metrical interpretation of the global tempo.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TempoHypothesis {
    /// Candidate global tempo.
    pub bpm: f64,
    /// Relative, intentionally uncalibrated score.
    pub relative_score: f64,
    /// Power-of-two relation to the preferred metrical level.
    pub metrical_level: i8,
}

/// Complete output returned by every product surface.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Analysis {
    /// Serialized schema version.
    pub schema_version: u32,
    /// Audio duration.
    pub duration_s: f64,
    /// Source model identity.
    pub source: ModelInfo,
    /// Beat and downbeat events.
    pub beats: Vec<BeatEvent>,
    /// Preferred global tempo summary.
    pub global_bpm: Option<f64>,
    /// Alternative half/double-time interpretations.
    pub tempo_hypotheses: Vec<TempoHypothesis>,
    /// Regularized local BPM samples.
    pub tempo_curve: Vec<TempoPoint>,
    /// Piecewise constant/ramp representation.
    pub tempo_segments: Vec<TempoSegment>,
    /// Timing and rhythm transitions.
    pub change_points: Vec<ChangePoint>,
    /// Tempo/rhythm-homogeneous regions.
    pub rhythm_sections: Vec<RhythmSection>,
    /// Non-fatal quality and interpretation warnings.
    pub warnings: Vec<String>,
}
