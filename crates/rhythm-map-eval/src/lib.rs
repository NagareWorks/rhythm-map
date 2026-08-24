//! Reproducible evaluation contracts and deterministic rhythm fixtures.

mod artbeat;
mod dataset;
mod manifest;
mod metrics;
mod public_dataset;
mod runner;
mod synthetic;
mod vienna;
mod wav;

pub use artbeat::{ArtbeatTruthImport, import_artbeat_truth};
pub use dataset::{AudioAssetInspection, ExternalAudioResolver, inspect_audio_asset};
pub use manifest::{
    AcceptanceThresholds, AssetKind, AssetProvenance, AudioReference, CaseInput, EvaluationCase,
    EvaluationSuite, SuitePurpose,
};
pub use metrics::{
    BeatMetrics, CaseEvaluation, ChangeMetrics, EvaluationMetrics, TempoMetrics, evaluate_analysis,
};
pub use public_dataset::{
    DatasetFetchAsset, DatasetFetchReport, DatasetFetchStatus, PublicDatasetAsset,
    PublicDatasetAssetRole, PublicDatasetLock, fetch_public_dataset,
};
pub use runner::{
    AttributionCase, AttributionDecision, BottleneckEvaluation, CandidateEvidenceCoverage,
    CaseMetricDelta, DecoderPolicy, DecoderPolicyComparison, DecoderPolicyEvaluation,
    DecoderRecoverabilityCase, DecoderRecoverabilityEvaluation, DecoderSequencePathPolicy,
    DecoderSliceDelta, DecoderSliceMetrics, DecoderSupportedMidpointPolicy, DecoderSweepCandidate,
    DecoderSweepCase, DecoderSweepEvaluation, LogitEvidence, MissingBeatEvidence,
    MissingBeatEvidenceClass, MissingBeatEvidenceCounts, ModelPackIdentity, ObservationDiagnostics,
    PulseEvidenceBreakdown, PulseHypothesisCoverage, PulseHypothesisEvaluation, SuiteEvaluation,
    evaluate_backend_suite, evaluate_backend_suite_with_audio_directory,
    evaluate_backend_suite_with_decoder_policy, evaluate_backend_suite_with_policies,
    evaluate_core_suite, evaluate_decoder_recoverability_with_audio_directory,
    evaluate_decoder_sweep_with_audio_directory,
    evaluate_named_decoder_policy_with_audio_directory, render_suite, score_prediction_directory,
    standard_decoder_policies, standard_decoder_policy,
};
pub use synthetic::{
    GeneratedTruth, RecipeSegment, SegmentShape, SyntheticAudioProfile, SyntheticRecipe, TruthBeat,
    TruthChangePoint, TruthTempoSegment, generate_truth,
};
pub use vienna::{ViennaTruthImport, import_vienna_truth};
