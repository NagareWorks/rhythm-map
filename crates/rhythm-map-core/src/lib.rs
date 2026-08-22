//! Backend-independent timing analysis and public result schema.

mod engine;
mod estimator;
mod types;

pub use engine::{BackendError, Engine, EngineError, RhythmObservationBackend};
pub use estimator::{AnalysisError, EstimatorOptions, MetricalSelectionPolicy, TempoMapEstimator};
pub use types::{
    ANALYSIS_SCHEMA_VERSION, Analysis, AudioActivityPoint, BeatEvent, ChangeKind, ChangePoint,
    ModelInfo, ObservedBeat, RhythmObservations, RhythmSection, TempoHypothesis, TempoPoint,
    TempoSegment, TempoSegmentKind,
};
