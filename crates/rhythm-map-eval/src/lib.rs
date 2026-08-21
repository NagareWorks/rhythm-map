//! Reproducible evaluation contracts and deterministic rhythm fixtures.

mod dataset;
mod manifest;
mod metrics;
mod public_dataset;
mod runner;
mod synthetic;
mod wav;

pub use dataset::{AudioAssetInspection, ExternalAudioResolver, inspect_audio_asset};
pub use manifest::{
    AcceptanceThresholds, AssetKind, AssetProvenance, AudioReference, CaseInput, EvaluationCase,
    EvaluationSuite,
};
pub use metrics::{
    BeatMetrics, CaseEvaluation, ChangeMetrics, EvaluationMetrics, TempoMetrics, evaluate_analysis,
};
pub use public_dataset::{
    DatasetFetchAsset, DatasetFetchReport, DatasetFetchStatus, PublicDatasetAsset,
    PublicDatasetAssetRole, PublicDatasetLock, fetch_public_dataset,
};
pub use runner::{
    AttributionCase, AttributionDecision, BottleneckEvaluation, CaseMetricDelta, ModelPackIdentity,
    ObservationDiagnostics, SuiteEvaluation, evaluate_backend_suite,
    evaluate_backend_suite_with_audio_directory, evaluate_core_suite, render_suite,
    score_prediction_directory,
};
pub use synthetic::{
    GeneratedTruth, RecipeSegment, SegmentShape, SyntheticAudioProfile, SyntheticRecipe, TruthBeat,
    TruthChangePoint, TruthTempoSegment, generate_truth,
};
