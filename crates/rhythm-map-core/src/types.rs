use serde::{Deserialize, Serialize};

/// Current serialized analysis schema version.
pub const ANALYSIS_SCHEMA_VERSION: u32 = 3;

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

/// Uncommitted beat evidence exposed by an observation backend.
///
/// Candidates must correspond to timestamps supported by the backend. They are
/// not accepted beats: they may support confidence-aware tempo regularization
/// or an explicit alternative hypothesis, but cannot silently become selected
/// beat timestamps.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeatCandidate {
    /// Time from the beginning of the decoded audio.
    pub time_s: f64,
    /// Beat confidence in `[0, 1]`.
    pub confidence: f64,
    /// Downbeat confidence at the same timestamp in `[0, 1]`.
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

/// Deterministic spectral-flux onset measurement derived from decoded PCM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioOnsetPoint {
    /// Center of the analysis window in seconds.
    pub time_s: f64,
    /// Positive spectral flux normalized to `[0, 1]` within the track.
    pub strength: f64,
    /// Normalized onset strength contributed by frequencies below 250 Hz.
    #[serde(default)]
    pub low_strength: f64,
    /// Normalized onset strength contributed by 250 Hz through 2 kHz.
    #[serde(default)]
    pub mid_strength: f64,
    /// Normalized onset strength contributed by frequencies above 2 kHz.
    #[serde(default)]
    pub high_strength: f64,
}

/// Deterministic chroma-distance evidence measured around a model-supported event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioHarmonicChangePoint {
    /// Model-supported timestamp around which the comparison was measured.
    pub time_s: f64,
    /// Cosine distance between pitch-class profiles before and after the event.
    pub strength: f64,
}

/// Backend-neutral observations consumed by the timing estimator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RhythmObservations {
    /// Audio duration in seconds.
    pub duration_s: f64,
    /// Sorted beat observations.
    pub beats: Vec<ObservedBeat>,
    /// Sorted model-supported alternatives not committed to the beat sequence.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub beat_candidates: Vec<BeatCandidate>,
    /// Optional deterministic activity envelope derived from decoded PCM.
    #[serde(default)]
    pub activity: Vec<AudioActivityPoint>,
    /// Optional deterministic onset envelope derived from decoded PCM.
    #[serde(default)]
    pub onsets: Vec<AudioOnsetPoint>,
    /// Optional deterministic harmonic-change evidence at supported event times.
    #[serde(default)]
    pub harmonic_changes: Vec<AudioHarmonicChangePoint>,
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

/// Construction used for one backend-supported beat-sequence interpretation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BeatSequenceHypothesisKind {
    /// The sequence selected for the primary tempo-map analysis.
    Selected,
    /// One alternating phase of the selected sequence interpreted at half-time.
    HalfTime,
    /// Real backend candidates inserted near selected interval midpoints.
    DoubleTime,
    /// A candidate-graph path whose local pulse level may change over time.
    LocallyVarying,
}

/// One auditable metrical interpretation made only from backend-supported times.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeatSequenceHypothesis {
    /// How this sequence was constructed from the observation layer.
    pub kind: BeatSequenceHypothesisKind,
    /// Power-of-two relation to the primary selected sequence.
    pub metrical_level: i8,
    /// Alternating phase for a half-time interpretation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<u8>,
    /// Truth-free score relative to the strongest returned hypothesis.
    pub relative_score: f64,
    /// Strictly increasing backend-supported beat timestamps.
    pub beat_times_s: Vec<f64>,
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
    /// Auditable selected and alternative backend-supported beat sequences.
    #[serde(default)]
    pub beat_hypotheses: Vec<BeatSequenceHypothesis>,
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
