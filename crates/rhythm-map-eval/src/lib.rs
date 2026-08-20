//! Reproducible evaluation contracts and deterministic rhythm fixtures.

mod manifest;
mod metrics;
mod runner;
mod synthetic;
mod wav;

pub use manifest::{
    AcceptanceThresholds, AssetKind, AssetProvenance, AudioReference, CaseInput, EvaluationCase,
    EvaluationSuite,
};
pub use metrics::{
    BeatMetrics, CaseEvaluation, ChangeMetrics, EvaluationMetrics, TempoMetrics, evaluate_analysis,
};
pub use runner::{SuiteEvaluation, evaluate_core_suite, render_suite, score_prediction_directory};
pub use synthetic::{
    GeneratedTruth, RecipeSegment, SegmentShape, SyntheticRecipe, TruthBeat, TruthChangePoint,
    TruthTempoSegment, generate_truth,
};
