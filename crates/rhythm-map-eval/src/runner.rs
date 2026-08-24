use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use rhythm_map_beat_this::{
    BeatThisBackend, BeatThisDecoderPolicy, PeakPickingOptions, SequencePathOptions,
    SupportedMidpointOptions, decode_audio,
};
use rhythm_map_core::{
    Analysis, Engine, EstimatorOptions, ModelInfo, ObservedBeat, RhythmObservations,
    TempoMapEstimator,
};
use rhythm_map_models::{ModelArtifactRole, VerifiedModelPack, verify_model_pack};
use serde::{Deserialize, Serialize};

use crate::{
    BeatMetrics, CaseEvaluation, CaseInput, EvaluationCase, EvaluationSuite, ExternalAudioResolver,
    GeneratedTruth, SuitePurpose, SyntheticRecipe, evaluate_analysis, generate_truth,
    metrics::{match_event_pairs, score_beats},
    wav::{render_synthetic_audio, synthesize_audio},
};

/// Complete evaluation report for one suite.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SuiteEvaluation {
    /// Evaluation report schema version.
    pub schema_version: u32,
    /// Stable suite identifier.
    pub suite_id: String,
    /// Declared development role of the evaluated suite.
    pub suite_purpose: SuitePurpose,
    /// True only when every case passes.
    pub passed: bool,
    /// Per-case metrics and failed budgets.
    pub cases: Vec<CaseEvaluation>,
}

/// Model-pack identity recorded in an end-to-end evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelPackIdentity {
    /// Stable model-pack identifier.
    pub id: String,
    /// Immutable model-pack version.
    pub version: String,
    /// Observation backend identity.
    pub backend: String,
    /// SHA-256 of the exact manifest bytes.
    pub manifest_sha256: String,
}

/// End-to-end metric difference relative to oracle observations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseMetricDelta {
    /// End-to-end minus oracle beat F1.
    pub beat_f1: f64,
    /// End-to-end minus oracle downbeat F1.
    pub downbeat_f1: f64,
    /// End-to-end minus oracle median tempo error percentage.
    pub tempo_median_error_percent: Option<f64>,
    /// End-to-end minus oracle p95 tempo error percentage.
    pub tempo_p95_error_percent: Option<f64>,
    /// End-to-end minus oracle change-point recall.
    pub change_recall: f64,
}

/// Raw backend events and compact PCM-activity diagnostics for one case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObservationDiagnostics {
    /// Backend and model identity attached to the raw events.
    pub source: ModelInfo,
    /// Exact beat/confidence events before deterministic filtering or metrical selection.
    pub raw_beats: Vec<ObservedBeat>,
    /// Number of beats retained in the product analysis.
    pub analyzed_beat_count: usize,
    /// Number of retained beats classified as downbeats after deterministic repair.
    #[serde(default)]
    pub analyzed_downbeat_count: usize,
    /// Number of deterministic activity-envelope samples.
    pub activity_point_count: usize,
    /// Quietest activity sample relative to peak level.
    pub minimum_relative_db: Option<f64>,
    /// Fraction of activity samples at or below -40 dB.
    pub low_activity_fraction: f64,
    /// Median tempo implied directly by consecutive raw backend events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_median_bpm: Option<f64>,
    /// Confidence-weighted PCM salience of the two alternating raw-event phases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alternating_phase_salience: Option<[f64; 2]>,
    /// Mean backend confidence of the two alternating raw-event phases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alternating_phase_confidence: Option<[f64; 2]>,
    /// Deterministic filtering and metrical-selection decisions.
    pub analysis_warnings: Vec<String>,
}

/// Available oracle evidence and the end-to-end result for one case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttributionCase {
    /// Stable case identifier.
    pub id: String,
    /// Capability slices copied from the suite manifest.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Verified encoded audio identity for an external case.
    #[serde(default)]
    pub audio_sha256: Option<String>,
    /// Estimator result from exact beat observations, absent for tempo-only truth.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oracle: Option<CaseEvaluation>,
    /// Product result from rendered audio and the observation backend.
    pub end_to_end: CaseEvaluation,
    /// Directional end-to-end metric difference, available only with an oracle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<CaseMetricDelta>,
    /// Raw beat/confidence events and activity summary used for diagnosis.
    pub observations: ObservationDiagnostics,
    /// Wall-clock time spent in end-to-end audio analysis.
    pub end_to_end_runtime_ms: f64,
}

/// Coarse bottleneck conclusion supported by suite acceptance gates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttributionDecision {
    /// Exact beats pass while the full audio-to-result path fails.
    ///
    /// This includes observation errors and estimator robustness to plausible
    /// but metrically ambiguous observations; it does not by itself prove the
    /// neural backend must be replaced.
    ObservationPath,
    /// The deterministic estimator fails even with exact beats.
    DeterministicEstimator,
    /// The suite has independently labeled output truth but no exact beat events
    /// from which to construct the estimator-only oracle path.
    EndToEndOnly,
    /// Both paths satisfy the current acceptance gates.
    NoMeasuredBottleneck,
}

/// Oracle attribution where available plus end-to-end evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BottleneckEvaluation {
    /// Evaluation report schema version.
    pub schema_version: u32,
    /// Stable suite identifier.
    pub suite_id: String,
    /// Declared development role of the evaluated suite.
    pub suite_purpose: SuitePurpose,
    /// Verified model-pack identity.
    pub model_pack: ModelPackIdentity,
    /// Explicit non-default decoder exercised by this report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoder_policy: Option<DecoderPolicy>,
    /// Explicit non-default deterministic estimator exercised by this report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimator_policy: Option<String>,
    /// True when the end-to-end path and every available oracle path pass.
    pub passed: bool,
    /// Acceptance-gate bottleneck decision, or `end_to_end_only` without an oracle.
    pub attribution: AttributionDecision,
    /// Per-case paired measurements.
    pub cases: Vec<AttributionCase>,
}

/// Named peak-picking policy included in a decoder sweep.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderPolicy {
    /// Stable report label.
    pub id: String,
    /// Strict lower bound applied to model logits.
    pub logit_threshold: f32,
    /// Local-maximum radius on each side, in 50 Hz frames.
    pub local_max_radius_frames: usize,
    /// Maximum adjacent-frame distance merged into one peak.
    pub deduplicate_width_frames: usize,
    /// Optional conservative weak-midpoint recovery after upstream decoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_midpoints: Option<DecoderSupportedMidpointPolicy>,
    /// Optional Viterbi path over beat-period and phase states.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_path: Option<DecoderSequencePathPolicy>,
}

impl DecoderPolicy {
    fn options(&self) -> PeakPickingOptions {
        PeakPickingOptions {
            logit_threshold: self.logit_threshold,
            local_max_radius_frames: self.local_max_radius_frames,
            deduplicate_width_frames: self.deduplicate_width_frames,
        }
    }

    fn backend_policy(&self) -> BeatThisDecoderPolicy {
        match (&self.supported_midpoints, &self.sequence_path) {
            (Some(midpoints), None) => {
                BeatThisDecoderPolicy::SupportedMidpoints(midpoints.options())
            }
            (None, Some(sequence)) => BeatThisDecoderPolicy::SequencePath(sequence.options()),
            (None, None) => BeatThisDecoderPolicy::PeakPicking(self.options()),
            (Some(_), Some(_)) => unreachable!("registered decoder policy is unambiguous"),
        }
    }
}

/// Serializable supported-midpoint candidate configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderSupportedMidpointPolicy {
    /// Strict lower logit bound for weak candidate peaks.
    pub candidate_logit_threshold: f32,
    /// Maximum midpoint offset as a fraction of the strong-beat interval.
    pub maximum_midpoint_offset_ratio: f64,
    /// Strong-beat gaps inspected on either side.
    pub support_radius_gaps: usize,
    /// Minimum locally supported gaps required before insertion.
    pub minimum_supported_gaps: usize,
}

impl DecoderSupportedMidpointPolicy {
    fn options(&self) -> SupportedMidpointOptions {
        SupportedMidpointOptions {
            candidate_logit_threshold: self.candidate_logit_threshold,
            maximum_midpoint_offset_ratio: self.maximum_midpoint_offset_ratio,
            support_radius_gaps: self.support_radius_gaps,
            minimum_supported_gaps: self.minimum_supported_gaps,
        }
    }
}

/// Serializable Viterbi sequence-path candidate configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderSequencePathPolicy {
    /// Strict lower logit bound for a local maximum to become an event.
    pub candidate_logit_threshold: f32,
    /// Local-maximum radius used for event candidates.
    pub candidate_local_max_radius_frames: usize,
    /// Maximum Viterbi-state to model-peak correction distance.
    pub maximum_peak_correction_frames: usize,
    /// Slowest represented beat period.
    pub minimum_bpm: f64,
    /// Fastest represented beat period.
    pub maximum_bpm: f64,
    /// Squared log-period penalty for tempo changes at beat boundaries.
    pub tempo_change_penalty: f64,
    /// Log-score prior added when the Viterbi path enters a beat state.
    pub beat_state_bias: f64,
    /// Maximum path-beat gap joining events into one weak-event sequence.
    pub support_radius_beats: usize,
    /// Minimum weak candidates required in one connected sequence.
    pub minimum_supported_candidates: usize,
    /// Minimum weak candidates required in the local support radius.
    pub minimum_local_supported_candidates: usize,
    /// Whether recovered weak runs must connect to a model-supported edge.
    pub require_edge_connection: bool,
    /// Maximum path-beat distance between a weak run and an observed edge.
    pub maximum_edge_gap_beats: usize,
}

impl DecoderSequencePathPolicy {
    fn options(&self) -> SequencePathOptions {
        SequencePathOptions {
            candidate_logit_threshold: self.candidate_logit_threshold,
            candidate_local_max_radius_frames: self.candidate_local_max_radius_frames,
            maximum_peak_correction_frames: self.maximum_peak_correction_frames,
            minimum_bpm: self.minimum_bpm,
            maximum_bpm: self.maximum_bpm,
            tempo_change_penalty: self.tempo_change_penalty,
            beat_state_bias: self.beat_state_bias,
            support_radius_beats: self.support_radius_beats,
            minimum_supported_candidates: self.minimum_supported_candidates,
            minimum_local_supported_candidates: self.minimum_local_supported_candidates,
            require_edge_connection: self.require_edge_connection,
            maximum_edge_gap_beats: self.maximum_edge_gap_beats,
        }
    }
}

/// Beat-only result for one case under one peak-picking policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderSweepCase {
    /// Stable case identifier.
    pub id: String,
    /// Capability slices inherited from the suite case.
    pub tags: Vec<String>,
    /// Number of discrete model events produced by this decoder.
    pub predicted_beat_count: usize,
    /// Required beat F1 inherited from suite or case thresholds.
    pub minimum_beat_f1: f64,
    /// Whether this raw decoder result clears its beat gate.
    pub passed: bool,
    /// One-to-one beat timing metrics against independent truth.
    pub beats: BeatMetrics,
}

/// Aggregate raw-beat metrics for one capability tag.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderSliceMetrics {
    /// Capability tag shared by the included cases.
    pub tag: String,
    /// Number of cases carrying the tag.
    pub case_count: usize,
    /// True only when every case in the slice clears its beat gate.
    pub passed: bool,
    /// Arithmetic mean of per-case beat precision.
    pub mean_beat_precision: f64,
    /// Arithmetic mean of per-case beat recall.
    pub mean_beat_recall: f64,
    /// Arithmetic mean of per-case beat F1.
    pub mean_beat_f1: f64,
}

/// Aggregate and per-case measurements for one peak-picking policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderSweepCandidate {
    /// Exact peak-picking policy.
    pub policy: DecoderPolicy,
    /// True only when every case clears its raw beat gate.
    pub passed: bool,
    /// Arithmetic mean of per-case beat F1.
    pub mean_beat_f1: f64,
    /// Arithmetic mean of per-case beat precision.
    pub mean_beat_precision: f64,
    /// Arithmetic mean of per-case beat recall.
    pub mean_beat_recall: f64,
    /// Arithmetic mean of decoded beat counts per case.
    pub mean_predicted_beat_count: f64,
    /// Stable per-tag aggregates for capability-level regression checks.
    pub slices: Vec<DecoderSliceMetrics>,
    /// Per-case beat metrics.
    pub cases: Vec<DecoderSweepCase>,
}

/// Single-inference comparison of several Beat This peak decoders.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderSweepEvaluation {
    /// Report schema version.
    pub schema_version: u32,
    /// Stable suite identifier.
    pub suite_id: String,
    /// Calibration is required because this report compares several policies.
    pub suite_purpose: SuitePurpose,
    /// Verified model pack used for every candidate.
    pub model_pack: ModelPackIdentity,
    /// Candidate policies in caller-provided order.
    pub candidates: Vec<DecoderSweepCandidate>,
    /// Mean F1 after choosing the best tested policy separately for each case.
    ///
    /// This is a diagnostic upper bound, not a deployable decoder result.
    pub per_case_policy_oracle_mean_beat_f1: f64,
}

/// Evaluation of one named decoder policy selected before opening a holdout.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderPolicyEvaluation {
    /// Report schema version.
    pub schema_version: u32,
    /// Stable suite identifier.
    pub suite_id: String,
    /// Declared development role of the evaluated suite.
    pub suite_purpose: SuitePurpose,
    /// Verified model pack used for evaluation.
    pub model_pack: ModelPackIdentity,
    /// Immutable product baseline evaluated in the same inference pass.
    pub baseline: DecoderSweepCandidate,
    /// The only selected candidate, including aggregate, slice, and case metrics.
    pub candidate: DecoderSweepCandidate,
    /// Candidate deltas against the upstream baseline.
    pub comparison: DecoderPolicyComparison,
    /// True when absolute beat gates pass and no case regresses from baseline.
    pub passed: bool,
}

/// Fixed candidate comparison against the upstream product decoder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderPolicyComparison {
    /// Candidate minus baseline mean beat F1.
    pub mean_beat_f1_delta: f64,
    /// Cases with a strictly lower candidate F1.
    pub regressed_case_ids: Vec<String>,
    /// Cases with a strictly higher candidate F1.
    pub improved_case_ids: Vec<String>,
    /// Candidate-minus-baseline aggregates for every capability tag.
    pub slices: Vec<DecoderSliceDelta>,
}

/// Fixed-policy F1 comparison for one capability tag.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderSliceDelta {
    /// Capability tag.
    pub tag: String,
    /// Number of cases carrying the tag.
    pub case_count: usize,
    /// Upstream decoder mean beat F1.
    pub baseline_mean_beat_f1: f64,
    /// Selected candidate mean beat F1.
    pub candidate_mean_beat_f1: f64,
    /// Candidate minus baseline mean beat F1.
    pub mean_beat_f1_delta: f64,
}

/// Evidence available near one truth beat missed by the upstream decoder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MissingBeatEvidence {
    /// Independently annotated beat timestamp.
    pub expected_time_s: f64,
    /// Strongest model frame inside the suite's beat tolerance window.
    pub strongest_frame: LogitEvidence,
    /// Strongest radius-one local maximum inside that window.
    pub radius_1_local_peak: Option<LogitEvidence>,
    /// Strongest upstream-radius local maximum inside that window.
    pub radius_3_local_peak: Option<LogitEvidence>,
    /// Mutually exclusive summary of decoder recoverability.
    pub class: MissingBeatEvidenceClass,
}

/// One model frame or local maximum close to a missed truth beat.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogitEvidence {
    /// Frame timestamp at 50 Hz.
    pub time_s: f64,
    /// Signed difference from the truth beat.
    pub offset_ms: f64,
    /// Raw Beat This beat logit.
    pub logit: f32,
    /// Sigmoid probability corresponding to `logit`.
    pub probability: f64,
}

/// Evidence tier for a truth beat missed by the upstream decoder.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MissingBeatEvidenceClass {
    /// A radius-three peak above zero exists but was consumed by ordered matching.
    DefaultEligiblePeak,
    /// A radius-three peak becomes eligible above logit -1.
    Radius3PeakAboveMinus1,
    /// A radius-three peak becomes eligible above logit -3.
    Radius3PeakAboveMinus3,
    /// A radius-three peak exists only at logit -3 or below.
    WeakerRadius3Peak,
    /// Only a radius-one local maximum exists in the tolerance window.
    Radius1OnlyPeak,
    /// No radius-one local maximum exists in the tolerance window.
    NoLocalPeak,
}

/// Counts for mutually exclusive missed-beat evidence tiers.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MissingBeatEvidenceCounts {
    /// Truth beats missed by the upstream decoder.
    pub missing_beats: usize,
    /// Misses with a radius-three peak above zero.
    pub default_eligible_peaks: usize,
    /// Misses with a radius-three peak in (-1, 0].
    pub radius_3_peaks_above_minus_1: usize,
    /// Misses with a radius-three peak in (-3, -1].
    pub radius_3_peaks_above_minus_3: usize,
    /// Misses with a radius-three peak at or below -3.
    pub weaker_radius_3_peaks: usize,
    /// Misses with a radius-one peak but no radius-three peak.
    pub radius_1_only_peaks: usize,
    /// Misses without a radius-one local peak.
    pub no_local_peaks: usize,
}

/// Per-case decoder recoverability measurements.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderRecoverabilityCase {
    /// Stable case identifier.
    pub id: String,
    /// Annotated beat count.
    pub expected_beat_count: usize,
    /// Upstream-default decoded beat count.
    pub predicted_beat_count: usize,
    /// Truth beats matched by the upstream-default decoder.
    pub matched_beat_count: usize,
    /// Evidence-tier counts for missed truth beats.
    pub counts: MissingBeatEvidenceCounts,
    /// Detailed evidence for each missed truth beat.
    pub missing: Vec<MissingBeatEvidence>,
}

/// Aggregate single-inference diagnosis of missed Beat This events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderRecoverabilityEvaluation {
    /// Report schema version.
    pub schema_version: u32,
    /// Stable suite identifier.
    pub suite_id: String,
    /// Calibration is required because this report inspects truth-assisted evidence.
    pub suite_purpose: SuitePurpose,
    /// Verified model pack used for every case.
    pub model_pack: ModelPackIdentity,
    /// Total annotated beats across the suite.
    pub expected_beat_count: usize,
    /// Total truth beats matched by the upstream decoder.
    pub matched_beat_count: usize,
    /// Aggregate evidence-tier counts for all misses.
    pub counts: MissingBeatEvidenceCounts,
    /// Per-case diagnostics.
    pub cases: Vec<DecoderRecoverabilityCase>,
}

/// Evaluate the timing estimator with ideal observations from every case.
///
/// # Errors
///
/// Returns an error for invalid manifests, recipes, or estimator failures.
pub fn evaluate_core_suite(suite_path: &Path) -> Result<SuiteEvaluation> {
    let (suite, root) = load_suite(suite_path)?;
    let estimator = TempoMapEstimator::default();
    let mut cases = Vec::with_capacity(suite.cases.len());
    for (case_index, case) in suite.cases.iter().enumerate() {
        eprintln!(
            "core evaluation {}/{}: {}",
            case_index + 1,
            suite.cases.len(),
            case.id
        );
        let truth = load_case_truth(case, &root)?;
        let analysis = estimator
            .estimate(&truth.ideal_observations())
            .with_context(|| format!("estimating case {}", case.id))?;
        let thresholds = case.thresholds.as_ref().unwrap_or(&suite.thresholds);
        cases.push(evaluate_analysis(&case.id, &analysis, &truth, thresholds));
    }
    Ok(suite_report(suite.id, suite.purpose, cases))
}

/// Run generated audio through Beat This and compare it with oracle observations.
///
/// # Errors
///
/// Returns an error for invalid suites, model packs, model loading, rendering,
/// inference, or estimator failures.
pub fn evaluate_backend_suite(
    suite_path: &Path,
    model_pack_path: &Path,
    model_root: &Path,
) -> Result<BottleneckEvaluation> {
    evaluate_backend_suite_impl(suite_path, model_pack_path, model_root, None, None, None)
}

/// Run generated and content-addressed external audio through Beat This.
///
/// External audio remains outside the repository and is resolved below the
/// explicit directory by the SHA-256 declared in the suite manifest.
///
/// # Errors
///
/// Returns an error for invalid suites, missing or mismatched external assets,
/// model loading, decoding, inference, or estimator failures.
pub fn evaluate_backend_suite_with_audio_directory(
    suite_path: &Path,
    model_pack_path: &Path,
    model_root: &Path,
    audio_directory: &Path,
) -> Result<BottleneckEvaluation> {
    let resolver = ExternalAudioResolver::new(audio_directory)?;
    evaluate_backend_suite_impl(
        suite_path,
        model_pack_path,
        model_root,
        Some(&resolver),
        None,
        None,
    )
}

/// Run a complete audio-to-analysis calibration with one pre-registered
/// decoder policy. Unlike decoder-eval, tempo-only suites are accepted because
/// this path scores only their declared end-to-end truth and never constructs
/// beat-phase labels.
///
/// # Errors
///
/// Returns an error for an unknown policy or any ordinary backend-evaluation
/// failure.
pub fn evaluate_backend_suite_with_decoder_policy(
    suite_path: &Path,
    model_pack_path: &Path,
    model_root: &Path,
    audio_directory: &Path,
    policy_id: &str,
) -> Result<BottleneckEvaluation> {
    evaluate_backend_suite_with_policies(
        suite_path,
        model_pack_path,
        model_root,
        audio_directory,
        Some(policy_id),
        None,
    )
}

/// Run the complete product path with explicit registered decoder and/or
/// estimator candidates. This is intended for calibration reports; omitted
/// policy IDs preserve the shipping defaults.
///
/// # Errors
///
/// Returns an error for an unknown policy or any ordinary backend-evaluation
/// failure.
pub fn evaluate_backend_suite_with_policies(
    suite_path: &Path,
    model_pack_path: &Path,
    model_root: &Path,
    audio_directory: &Path,
    decoder_policy_id: Option<&str>,
    estimator_policy_id: Option<&str>,
) -> Result<BottleneckEvaluation> {
    let resolver = ExternalAudioResolver::new(audio_directory)?;
    let decoder_policy = decoder_policy_id.map(standard_decoder_policy).transpose()?;
    validate_estimator_policy(estimator_policy_id)?;
    evaluate_backend_suite_impl(
        suite_path,
        model_pack_path,
        model_root,
        Some(&resolver),
        decoder_policy.as_ref(),
        estimator_policy_id,
    )
}

fn evaluate_backend_suite_impl(
    suite_path: &Path,
    model_pack_path: &Path,
    model_root: &Path,
    audio_resolver: Option<&ExternalAudioResolver>,
    decoder_policy: Option<&DecoderPolicy>,
    estimator_policy: Option<&str>,
) -> Result<BottleneckEvaluation> {
    let verified = verify_model_pack(model_pack_path, model_root)
        .with_context(|| format!("verifying model pack {}", model_pack_path.display()))?;
    validate_beat_this_contract(&verified)?;
    let mel_model = required_model_path(&verified, ModelArtifactRole::MelFrontend)?;
    let beat_model = required_model_path(&verified, ModelArtifactRole::BeatModel)?;
    let backend = BeatThisBackend::load(&mel_model, &beat_model)?;
    let backend = match decoder_policy {
        Some(policy) => backend.with_decoder_policy(policy.backend_policy()),
        None => backend,
    };
    let estimator = estimator_for_policy(estimator_policy)?;
    let mut engine = Engine::with_estimator(backend, estimator.clone());
    let (suite, root) = load_suite(suite_path)?;
    let mut cases = Vec::with_capacity(suite.cases.len());

    for (case_index, case) in suite.cases.iter().enumerate() {
        eprintln!(
            "backend evaluation {}/{}: {}",
            case_index + 1,
            suite.cases.len(),
            case.id
        );
        let truth = load_case_truth(case, &root)?;
        let thresholds = case.thresholds.as_ref().unwrap_or(&suite.thresholds);
        let oracle = if truth.beats.is_empty() {
            None
        } else {
            let oracle_analysis = estimator
                .estimate(&truth.ideal_observations())
                .with_context(|| format!("estimating oracle case {}", case.id))?;
            Some(evaluate_analysis(
                &case.id,
                &oracle_analysis,
                &truth,
                thresholds,
            ))
        };

        let (samples, sample_rate, audio_sha256) =
            load_case_audio(case, &suite.id, &root, &truth, audio_resolver)?;
        let started = Instant::now();
        let observations = engine
            .observe_pcm(&samples, sample_rate, 1)
            .with_context(|| format!("observing backend case {}", case.id))?;
        let analysis = engine
            .analyze_observations(&observations)
            .with_context(|| format!("estimating backend case {}", case.id))?;
        let runtime_ms = started.elapsed().as_secs_f64() * 1000.0;
        let end_to_end = evaluate_analysis(&case.id, &analysis, &truth, thresholds);
        let delta = oracle
            .as_ref()
            .map(|oracle| metric_delta(oracle, &end_to_end));
        let observation_diagnostics = observation_diagnostics(&observations, &analysis);
        cases.push(AttributionCase {
            id: case.id.clone(),
            tags: case.tags.clone(),
            audio_sha256,
            oracle,
            end_to_end,
            delta,
            observations: observation_diagnostics,
            end_to_end_runtime_ms: runtime_ms,
        });
    }

    let oracle_passed = cases
        .iter()
        .filter_map(|case| case.oracle.as_ref())
        .all(|oracle| oracle.passed);
    let has_unpaired_cases = cases.iter().any(|case| case.oracle.is_none());
    let end_to_end_passed = cases.iter().all(|case| case.end_to_end.passed);
    let attribution = attribution_decision(oracle_passed, has_unpaired_cases, end_to_end_passed);
    let manifest = verified.manifest();
    Ok(BottleneckEvaluation {
        schema_version: 2,
        suite_id: suite.id,
        suite_purpose: suite.purpose,
        model_pack: ModelPackIdentity {
            id: manifest.id.clone(),
            version: manifest.version.clone(),
            backend: manifest.backend.clone(),
            manifest_sha256: verified.manifest_sha256().to_string(),
        },
        decoder_policy: decoder_policy.cloned(),
        estimator_policy: estimator_policy.map(str::to_string),
        passed: oracle_passed && end_to_end_passed,
        attribution,
        cases,
    })
}

fn validate_estimator_policy(policy_id: Option<&str>) -> Result<()> {
    if policy_id.is_none_or(|id| matches!(id, "metrical-consistency-v1" | "sequence-phase-v1")) {
        return Ok(());
    }
    bail!(
        "unknown estimator policy {}; available policies: metrical-consistency-v1, sequence-phase-v1",
        policy_id.expect("checked as some")
    )
}

fn estimator_for_policy(policy_id: Option<&str>) -> Result<TempoMapEstimator> {
    validate_estimator_policy(policy_id)?;
    match policy_id {
        Some("metrical-consistency-v1") => {
            TempoMapEstimator::new(EstimatorOptions::metrical_consistency_candidate())
                .map_err(Into::into)
        }
        Some("sequence-phase-v1") => {
            TempoMapEstimator::new(EstimatorOptions::sequence_phase_candidate()).map_err(Into::into)
        }
        None => Ok(TempoMapEstimator::default()),
        Some(_) => unreachable!("validated estimator policy"),
    }
}

/// Standard threshold and local-maximum sweep around the upstream decoder.
#[must_use]
pub fn standard_decoder_policies() -> Vec<DecoderPolicy> {
    let mut policies = vec![DecoderPolicy {
        id: "upstream-default".to_string(),
        logit_threshold: 0.0,
        local_max_radius_frames: 3,
        deduplicate_width_frames: 1,
        supported_midpoints: None,
        sequence_path: None,
    }];
    for threshold in [-0.5_f32, -1.0, -2.0, -3.0] {
        policies.push(DecoderPolicy {
            id: format!("logit-minus-{:.1}", -threshold),
            logit_threshold: threshold,
            local_max_radius_frames: 3,
            deduplicate_width_frames: 1,
            supported_midpoints: None,
            sequence_path: None,
        });
    }
    for threshold in [0.0_f32, -1.0, -2.0, -3.0] {
        policies.push(DecoderPolicy {
            id: if threshold == 0.0 {
                "radius-1-default-logit".to_string()
            } else {
                format!("radius-1-logit-minus-{:.1}", -threshold)
            },
            logit_threshold: threshold,
            local_max_radius_frames: 1,
            deduplicate_width_frames: 1,
            supported_midpoints: None,
            sequence_path: None,
        });
    }
    let midpoint_options = SupportedMidpointOptions::default();
    policies.push(DecoderPolicy {
        id: "supported-midpoints-logit-minus-3.0".to_string(),
        logit_threshold: 0.0,
        local_max_radius_frames: 3,
        deduplicate_width_frames: 1,
        supported_midpoints: Some(DecoderSupportedMidpointPolicy {
            candidate_logit_threshold: midpoint_options.candidate_logit_threshold,
            maximum_midpoint_offset_ratio: midpoint_options.maximum_midpoint_offset_ratio,
            support_radius_gaps: midpoint_options.support_radius_gaps,
            minimum_supported_gaps: midpoint_options.minimum_supported_gaps,
        }),
        sequence_path: None,
    });
    let sequence_options = SequencePathOptions::default();
    policies.push(sequence_decoder_policy(
        "viterbi-edge-logit-minus-3.0-bias-2.0",
        sequence_options,
    ));
    policies
}

fn sequence_decoder_policy(id: &str, options: SequencePathOptions) -> DecoderPolicy {
    DecoderPolicy {
        id: id.to_string(),
        logit_threshold: 0.0,
        local_max_radius_frames: 3,
        deduplicate_width_frames: 1,
        supported_midpoints: None,
        sequence_path: Some(DecoderSequencePathPolicy {
            candidate_logit_threshold: options.candidate_logit_threshold,
            candidate_local_max_radius_frames: options.candidate_local_max_radius_frames,
            maximum_peak_correction_frames: options.maximum_peak_correction_frames,
            minimum_bpm: options.minimum_bpm,
            maximum_bpm: options.maximum_bpm,
            tempo_change_penalty: options.tempo_change_penalty,
            beat_state_bias: options.beat_state_bias,
            support_radius_beats: options.support_radius_beats,
            minimum_supported_candidates: options.minimum_supported_candidates,
            minimum_local_supported_candidates: options.minimum_local_supported_candidates,
            require_edge_connection: options.require_edge_connection,
            maximum_edge_gap_beats: options.maximum_edge_gap_beats,
        }),
    }
}

/// Resolve one immutable policy from the standard decoder registry.
///
/// # Errors
///
/// Returns an error when `id` does not name a registered policy.
pub fn standard_decoder_policy(id: &str) -> Result<DecoderPolicy> {
    let policies = standard_decoder_policies();
    policies
        .iter()
        .find(|policy| policy.id == id)
        .cloned()
        .with_context(|| {
            let available = policies
                .iter()
                .map(|policy| policy.id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!("unknown decoder policy {id:?}; available policies: {available}")
        })
}

/// Compare peak-picking policies while running Beat This inference once per case.
///
/// This isolates discrete decoding from the neural frontend and model. It
/// scores raw decoded beat timestamps before deterministic tempo estimation.
///
/// # Errors
///
/// Returns an error for invalid suites, model packs, external assets, policies,
/// decoding, or inference.
pub fn evaluate_decoder_sweep_with_audio_directory(
    suite_path: &Path,
    model_pack_path: &Path,
    model_root: &Path,
    audio_directory: &Path,
    policies: &[DecoderPolicy],
) -> Result<DecoderSweepEvaluation> {
    let (suite, model_pack, candidates, per_case_policy_oracle_mean_beat_f1) =
        evaluate_decoder_candidates_with_audio_directory(
            suite_path,
            model_pack_path,
            model_root,
            audio_directory,
            policies,
            "decoder sweep",
            false,
        )?;
    Ok(DecoderSweepEvaluation {
        schema_version: 1,
        suite_id: suite.id,
        suite_purpose: suite.purpose,
        model_pack,
        candidates,
        per_case_policy_oracle_mean_beat_f1,
    })
}

/// Evaluate one registered candidate against the fixed upstream baseline.
///
/// Unlike a sweep, this entry point exposes only the selected candidate, the
/// immutable product baseline, and their deltas. It does not expose alternate
/// candidates or per-case-oracle results. Select the candidate on calibration
/// data first.
///
/// # Errors
///
/// Returns an error for an unknown policy, invalid suite, model pack, asset,
/// decoding failure, or inference failure.
pub fn evaluate_named_decoder_policy_with_audio_directory(
    suite_path: &Path,
    model_pack_path: &Path,
    model_root: &Path,
    audio_directory: &Path,
    policy_id: &str,
) -> Result<DecoderPolicyEvaluation> {
    let policy = standard_decoder_policy(policy_id)?;
    let baseline_policy = standard_decoder_policy("upstream-default")?;
    let policies = if policy == baseline_policy {
        vec![baseline_policy.clone()]
    } else {
        vec![baseline_policy.clone(), policy]
    };
    let (suite, model_pack, mut candidates, _) = evaluate_decoder_candidates_with_audio_directory(
        suite_path,
        model_pack_path,
        model_root,
        audio_directory,
        &policies,
        "decoder evaluation",
        true,
    )?;
    let candidate = candidates
        .pop()
        .context("fixed decoder evaluation produced no candidate")?;
    let baseline = candidates.pop().unwrap_or_else(|| candidate.clone());
    let comparison = compare_decoder_candidates(&baseline, &candidate)?;
    let passed = candidate.passed && comparison.regressed_case_ids.is_empty();
    Ok(DecoderPolicyEvaluation {
        schema_version: 1,
        suite_id: suite.id,
        suite_purpose: suite.purpose,
        model_pack,
        baseline,
        candidate,
        comparison,
        passed,
    })
}

fn compare_decoder_candidates(
    baseline: &DecoderSweepCandidate,
    candidate: &DecoderSweepCandidate,
) -> Result<DecoderPolicyComparison> {
    if baseline.cases.len() != candidate.cases.len() {
        bail!("decoder comparison requires identical case sets");
    }
    let mut regressed_case_ids = Vec::new();
    let mut improved_case_ids = Vec::new();
    for (baseline_case, candidate_case) in baseline.cases.iter().zip(&candidate.cases) {
        if baseline_case.id != candidate_case.id {
            bail!("decoder comparison case order differs");
        }
        let delta = candidate_case.beats.f1 - baseline_case.beats.f1;
        if delta < -f64::EPSILON {
            regressed_case_ids.push(candidate_case.id.clone());
        } else if delta > f64::EPSILON {
            improved_case_ids.push(candidate_case.id.clone());
        }
    }
    let baseline_slices = baseline
        .slices
        .iter()
        .map(|slice| (slice.tag.as_str(), slice))
        .collect::<BTreeMap<_, _>>();
    let slices = candidate
        .slices
        .iter()
        .map(|candidate_slice| {
            let baseline_slice = baseline_slices
                .get(candidate_slice.tag.as_str())
                .with_context(|| format!("baseline is missing slice {}", candidate_slice.tag))?;
            Ok(DecoderSliceDelta {
                tag: candidate_slice.tag.clone(),
                case_count: candidate_slice.case_count,
                baseline_mean_beat_f1: baseline_slice.mean_beat_f1,
                candidate_mean_beat_f1: candidate_slice.mean_beat_f1,
                mean_beat_f1_delta: candidate_slice.mean_beat_f1 - baseline_slice.mean_beat_f1,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(DecoderPolicyComparison {
        mean_beat_f1_delta: candidate.mean_beat_f1 - baseline.mean_beat_f1,
        regressed_case_ids,
        improved_case_ids,
        slices,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_decoder_candidates_with_audio_directory(
    suite_path: &Path,
    model_pack_path: &Path,
    model_root: &Path,
    audio_directory: &Path,
    policies: &[DecoderPolicy],
    operation: &str,
    allow_holdout: bool,
) -> Result<(
    EvaluationSuite,
    ModelPackIdentity,
    Vec<DecoderSweepCandidate>,
    f64,
)> {
    if policies.is_empty() {
        bail!("{operation} requires at least one policy");
    }
    let (suite, root) = load_suite(suite_path)?;
    if suite.purpose != SuitePurpose::Calibration && !allow_holdout {
        bail!(
            "{operation} requires a calibration suite; {} declares {:?}. Select one candidate on calibration data and use decoder-eval for regression or holdout evidence",
            suite.id,
            suite.purpose
        );
    }
    ensure_timestamped_beat_truth(&suite, &root, operation)?;
    let verified = verify_model_pack(model_pack_path, model_root)
        .with_context(|| format!("verifying model pack {}", model_pack_path.display()))?;
    validate_beat_this_contract(&verified)?;
    let mel_model = required_model_path(&verified, ModelArtifactRole::MelFrontend)?;
    let beat_model = required_model_path(&verified, ModelArtifactRole::BeatModel)?;
    let mut backend = BeatThisBackend::load(&mel_model, &beat_model)?;
    let resolver = ExternalAudioResolver::new(audio_directory)?;
    let mut candidates = policies
        .iter()
        .cloned()
        .map(|policy| DecoderSweepCandidate {
            policy,
            passed: false,
            mean_beat_f1: 0.0,
            mean_beat_precision: 0.0,
            mean_beat_recall: 0.0,
            mean_predicted_beat_count: 0.0,
            slices: Vec::new(),
            cases: Vec::with_capacity(suite.cases.len()),
        })
        .collect::<Vec<_>>();

    for (case_index, case) in suite.cases.iter().enumerate() {
        eprintln!(
            "{operation} {}/{}: {}",
            case_index + 1,
            suite.cases.len(),
            case.id
        );
        score_decoder_case(
            &mut backend,
            &resolver,
            &suite,
            &root,
            case,
            &mut candidates,
            operation,
        )?;
    }
    let per_case_policy_oracle_mean_beat_f1 = finalize_decoder_candidates(&mut candidates);
    let manifest = verified.manifest();
    let model_pack = ModelPackIdentity {
        id: manifest.id.clone(),
        version: manifest.version.clone(),
        backend: manifest.backend.clone(),
        manifest_sha256: verified.manifest_sha256().to_string(),
    };
    Ok((
        suite,
        model_pack,
        candidates,
        per_case_policy_oracle_mean_beat_f1,
    ))
}

#[allow(clippy::too_many_arguments)]
fn score_decoder_case(
    backend: &mut BeatThisBackend,
    resolver: &ExternalAudioResolver,
    suite: &EvaluationSuite,
    root: &Path,
    case: &EvaluationCase,
    candidates: &mut [DecoderSweepCandidate],
    operation: &str,
) -> Result<()> {
    let truth = load_case_truth(case, root)?;
    let (samples, sample_rate, _) = load_case_audio(case, &suite.id, root, &truth, Some(resolver))?;
    let inference = backend
        .infer_mono(&samples, sample_rate)
        .with_context(|| format!("inferring {operation} case {}", case.id))?;
    let expected = truth
        .beats
        .iter()
        .map(|beat| beat.time_s)
        .collect::<Vec<_>>();
    let thresholds = case.thresholds.as_ref().unwrap_or(&suite.thresholds);
    let tolerance_s = thresholds.beat_tolerance_ms / 1000.0;
    for candidate in candidates {
        let decoded =
            backend.decode_inference_with_policy(&inference, candidate.policy.backend_policy());
        let observations = decoded.with_context(|| {
            format!(
                "decoding case {} with policy {}",
                case.id, candidate.policy.id
            )
        })?;
        let predicted = observations
            .beats
            .iter()
            .map(|beat| beat.time_s)
            .collect::<Vec<_>>();
        let beats = score_beats(&predicted, &expected, tolerance_s);
        candidate.cases.push(DecoderSweepCase {
            id: case.id.clone(),
            tags: case.tags.clone(),
            predicted_beat_count: predicted.len(),
            minimum_beat_f1: thresholds.min_beat_f1,
            passed: beats.f1 >= thresholds.min_beat_f1,
            beats,
        });
    }
    Ok(())
}

/// Diagnose the model evidence near every truth beat missed by the upstream decoder.
///
/// Each audio case is inferred once. Ground truth is used only to inspect the
/// logits inside the suite's existing beat-tolerance window; this function does
/// not produce or tune a deployable decoder.
///
/// # Errors
///
/// Returns an error for invalid suites, model packs, external assets, decoding,
/// non-finite or empty logits, or inference.
pub fn evaluate_decoder_recoverability_with_audio_directory(
    suite_path: &Path,
    model_pack_path: &Path,
    model_root: &Path,
    audio_directory: &Path,
) -> Result<DecoderRecoverabilityEvaluation> {
    let (suite, root) = load_suite(suite_path)?;
    if suite.purpose != SuitePurpose::Calibration {
        bail!(
            "decoder recoverability requires a calibration suite; {} declares {:?}",
            suite.id,
            suite.purpose
        );
    }
    ensure_timestamped_beat_truth(&suite, &root, "decoder recoverability")?;
    let verified = verify_model_pack(model_pack_path, model_root)
        .with_context(|| format!("verifying model pack {}", model_pack_path.display()))?;
    validate_beat_this_contract(&verified)?;
    let mel_model = required_model_path(&verified, ModelArtifactRole::MelFrontend)?;
    let beat_model = required_model_path(&verified, ModelArtifactRole::BeatModel)?;
    let mut backend = BeatThisBackend::load(&mel_model, &beat_model)?;
    let resolver = ExternalAudioResolver::new(audio_directory)?;
    let mut cases = Vec::with_capacity(suite.cases.len());
    let mut aggregate_counts = MissingBeatEvidenceCounts::default();
    let mut expected_beat_count = 0;
    let mut matched_beat_count = 0;

    for (case_index, case) in suite.cases.iter().enumerate() {
        eprintln!(
            "decoder recoverability {}/{}: {}",
            case_index + 1,
            suite.cases.len(),
            case.id
        );
        let truth = load_case_truth(case, &root)?;
        let (samples, sample_rate, _) =
            load_case_audio(case, &suite.id, &root, &truth, Some(&resolver))?;
        let inference = backend
            .infer_mono(&samples, sample_rate)
            .with_context(|| format!("inferring recoverability case {}", case.id))?;
        validate_beat_logits(inference.beat_logits(), &case.id)?;
        let observations = backend
            .decode_inference(&inference, PeakPickingOptions::default())
            .with_context(|| format!("decoding recoverability case {}", case.id))?;
        let predicted = observations
            .beats
            .iter()
            .map(|beat| beat.time_s)
            .collect::<Vec<_>>();
        let expected = truth
            .beats
            .iter()
            .map(|beat| beat.time_s)
            .collect::<Vec<_>>();
        let tolerance_s = case
            .thresholds
            .as_ref()
            .unwrap_or(&suite.thresholds)
            .beat_tolerance_ms
            / 1000.0;
        let pairs = match_event_pairs(&predicted, &expected, tolerance_s);
        let mut matched_expected = vec![false; expected.len()];
        for &(_, expected_index) in &pairs {
            matched_expected[expected_index] = true;
        }
        let missing = expected
            .iter()
            .zip(matched_expected)
            .filter(|(_, matched)| !matched)
            .map(|(&expected_time_s, _)| {
                missing_beat_evidence(inference.beat_logits(), expected_time_s, tolerance_s)
            })
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("classifying recoverability case {}", case.id))?;
        let counts = count_missing_evidence(&missing);
        aggregate_counts.add(&counts);
        expected_beat_count += expected.len();
        matched_beat_count += pairs.len();
        cases.push(DecoderRecoverabilityCase {
            id: case.id.clone(),
            expected_beat_count: expected.len(),
            predicted_beat_count: predicted.len(),
            matched_beat_count: pairs.len(),
            counts,
            missing,
        });
    }

    let manifest = verified.manifest();
    Ok(DecoderRecoverabilityEvaluation {
        schema_version: 1,
        suite_id: suite.id,
        suite_purpose: suite.purpose,
        model_pack: ModelPackIdentity {
            id: manifest.id.clone(),
            version: manifest.version.clone(),
            backend: manifest.backend.clone(),
            manifest_sha256: verified.manifest_sha256().to_string(),
        },
        expected_beat_count,
        matched_beat_count,
        counts: aggregate_counts,
        cases,
    })
}

fn validate_beat_logits(logits: &[f32], case_id: &str) -> Result<()> {
    if logits.is_empty() {
        bail!("Beat This returned no beat logits for case {case_id}");
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        bail!("Beat This returned non-finite beat logits for case {case_id}");
    }
    Ok(())
}

fn missing_beat_evidence(
    logits: &[f32],
    expected_time_s: f64,
    tolerance_s: f64,
) -> Result<MissingBeatEvidence> {
    let (start, end) = evidence_frame_bounds(logits.len(), expected_time_s, tolerance_s)
        .context("truth beat has no model frame inside its tolerance window")?;
    let strongest_frame_index = strongest_frame(logits, start, end);
    let radius_1_peak_index = strongest_local_peak(logits, start, end, 1);
    let radius_3_peak_index = strongest_local_peak(logits, start, end, 3);
    let radius_1_local_peak =
        radius_1_peak_index.map(|index| logit_evidence(logits, index, expected_time_s));
    let radius_3_local_peak =
        radius_3_peak_index.map(|index| logit_evidence(logits, index, expected_time_s));
    let class =
        classify_missing_evidence(radius_1_local_peak.as_ref(), radius_3_local_peak.as_ref());
    Ok(MissingBeatEvidence {
        expected_time_s,
        strongest_frame: logit_evidence(logits, strongest_frame_index, expected_time_s),
        radius_1_local_peak,
        radius_3_local_peak,
        class,
    })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn evidence_frame_bounds(
    frame_count: usize,
    expected_time_s: f64,
    tolerance_s: f64,
) -> Option<(usize, usize)> {
    if frame_count == 0 || !expected_time_s.is_finite() || !tolerance_s.is_finite() {
        return None;
    }
    let start = ((expected_time_s - tolerance_s).max(0.0) * 50.0).ceil() as usize;
    let end = ((expected_time_s + tolerance_s).max(0.0) * 50.0).floor() as usize;
    let end = end.min(frame_count - 1);
    (start <= end).then_some((start, end))
}

fn strongest_frame(logits: &[f32], start: usize, end: usize) -> usize {
    (start..=end)
        .max_by(|&left, &right| logits[left].total_cmp(&logits[right]))
        .unwrap_or(start)
}

fn strongest_local_peak(logits: &[f32], start: usize, end: usize, radius: usize) -> Option<usize> {
    (start..=end)
        .filter(|&index| is_local_maximum(logits, index, radius))
        .max_by(|&left, &right| logits[left].total_cmp(&logits[right]))
}

fn is_local_maximum(logits: &[f32], index: usize, radius: usize) -> bool {
    let start = index.saturating_sub(radius);
    let end = index
        .saturating_add(radius)
        .saturating_add(1)
        .min(logits.len());
    logits[start..end]
        .iter()
        .all(|&neighbor| neighbor <= logits[index])
}

#[allow(clippy::cast_precision_loss)]
fn logit_evidence(logits: &[f32], index: usize, expected_time_s: f64) -> LogitEvidence {
    let time_s = index as f64 / 50.0;
    let logit = logits[index];
    LogitEvidence {
        time_s,
        offset_ms: (time_s - expected_time_s) * 1000.0,
        logit,
        probability: 1.0 / (1.0 + (-f64::from(logit)).exp()),
    }
}

fn classify_missing_evidence(
    radius_1_peak: Option<&LogitEvidence>,
    radius_3_peak: Option<&LogitEvidence>,
) -> MissingBeatEvidenceClass {
    if let Some(peak) = radius_3_peak {
        if peak.logit > 0.0 {
            MissingBeatEvidenceClass::DefaultEligiblePeak
        } else if peak.logit > -1.0 {
            MissingBeatEvidenceClass::Radius3PeakAboveMinus1
        } else if peak.logit > -3.0 {
            MissingBeatEvidenceClass::Radius3PeakAboveMinus3
        } else {
            MissingBeatEvidenceClass::WeakerRadius3Peak
        }
    } else if radius_1_peak.is_some() {
        MissingBeatEvidenceClass::Radius1OnlyPeak
    } else {
        MissingBeatEvidenceClass::NoLocalPeak
    }
}

fn count_missing_evidence(missing: &[MissingBeatEvidence]) -> MissingBeatEvidenceCounts {
    let mut counts = MissingBeatEvidenceCounts {
        missing_beats: missing.len(),
        ..MissingBeatEvidenceCounts::default()
    };
    for evidence in missing {
        match evidence.class {
            MissingBeatEvidenceClass::DefaultEligiblePeak => counts.default_eligible_peaks += 1,
            MissingBeatEvidenceClass::Radius3PeakAboveMinus1 => {
                counts.radius_3_peaks_above_minus_1 += 1;
            }
            MissingBeatEvidenceClass::Radius3PeakAboveMinus3 => {
                counts.radius_3_peaks_above_minus_3 += 1;
            }
            MissingBeatEvidenceClass::WeakerRadius3Peak => counts.weaker_radius_3_peaks += 1,
            MissingBeatEvidenceClass::Radius1OnlyPeak => counts.radius_1_only_peaks += 1,
            MissingBeatEvidenceClass::NoLocalPeak => counts.no_local_peaks += 1,
        }
    }
    counts
}

impl MissingBeatEvidenceCounts {
    fn add(&mut self, other: &Self) {
        self.missing_beats += other.missing_beats;
        self.default_eligible_peaks += other.default_eligible_peaks;
        self.radius_3_peaks_above_minus_1 += other.radius_3_peaks_above_minus_1;
        self.radius_3_peaks_above_minus_3 += other.radius_3_peaks_above_minus_3;
        self.weaker_radius_3_peaks += other.weaker_radius_3_peaks;
        self.radius_1_only_peaks += other.radius_1_only_peaks;
        self.no_local_peaks += other.no_local_peaks;
    }
}

fn finalize_decoder_candidates(candidates: &mut [DecoderSweepCandidate]) -> f64 {
    for candidate in &mut *candidates {
        let case_count = usize_to_f64(candidate.cases.len());
        candidate.passed = candidate.cases.iter().all(|case| case.passed);
        candidate.mean_beat_f1 = candidate
            .cases
            .iter()
            .map(|case| case.beats.f1)
            .sum::<f64>()
            / case_count;
        candidate.mean_beat_precision = candidate
            .cases
            .iter()
            .map(|case| case.beats.precision)
            .sum::<f64>()
            / case_count;
        candidate.mean_beat_recall = candidate
            .cases
            .iter()
            .map(|case| case.beats.recall)
            .sum::<f64>()
            / case_count;
        candidate.mean_predicted_beat_count = candidate
            .cases
            .iter()
            .map(|case| usize_to_f64(case.predicted_beat_count))
            .sum::<f64>()
            / case_count;
        let mut slices = BTreeMap::<String, Vec<&DecoderSweepCase>>::new();
        for case in &candidate.cases {
            for tag in &case.tags {
                slices.entry(tag.clone()).or_default().push(case);
            }
        }
        candidate.slices = slices
            .into_iter()
            .map(|(tag, cases)| {
                let case_count = usize_to_f64(cases.len());
                DecoderSliceMetrics {
                    tag,
                    case_count: cases.len(),
                    passed: cases.iter().all(|case| case.passed),
                    mean_beat_precision: cases.iter().map(|case| case.beats.precision).sum::<f64>()
                        / case_count,
                    mean_beat_recall: cases.iter().map(|case| case.beats.recall).sum::<f64>()
                        / case_count,
                    mean_beat_f1: cases.iter().map(|case| case.beats.f1).sum::<f64>() / case_count,
                }
            })
            .collect();
    }
    let case_count = candidates[0].cases.len();
    (0..case_count)
        .map(|case_index| {
            candidates
                .iter()
                .map(|candidate| candidate.cases[case_index].beats.f1)
                .fold(0.0_f64, f64::max)
        })
        .sum::<f64>()
        / usize_to_f64(case_count)
}

fn load_case_audio(
    case: &EvaluationCase,
    suite_id: &str,
    root: &Path,
    truth: &GeneratedTruth,
    audio_resolver: Option<&ExternalAudioResolver>,
) -> Result<(Vec<f32>, u32, Option<String>)> {
    match &case.input {
        CaseInput::Generated { recipe } => {
            let recipe = load_recipe(&root.join(recipe))?;
            let samples = synthesize_audio(&recipe, truth)
                .with_context(|| format!("synthesizing case {}", case.id))?;
            Ok((samples, recipe.sample_rate, None))
        }
        CaseInput::External { audio, .. } => {
            let resolver = audio_resolver.with_context(|| {
                format!(
                    "suite {suite_id} contains external case {}; pass an external audio directory",
                    case.id
                )
            })?;
            let path = resolver
                .resolve(audio)
                .with_context(|| format!("resolving audio for case {}", case.id))?;
            let decoded = decode_audio(&path)
                .with_context(|| format!("decoding audio for case {}", case.id))?;
            let decoded_duration_s =
                usize_to_f64(decoded.samples.len()) / f64::from(decoded.sample_rate);
            if (decoded_duration_s - truth.duration_s).abs() > 0.1 {
                bail!(
                    "case {} truth duration {:.6}s differs from decoded audio duration {:.6}s",
                    case.id,
                    truth.duration_s,
                    decoded_duration_s
                );
            }
            Ok((
                decoded.samples,
                decoded.sample_rate,
                Some(audio.sha256.clone()),
            ))
        }
    }
}

fn observation_diagnostics(
    observations: &RhythmObservations,
    analysis: &Analysis,
) -> ObservationDiagnostics {
    let minimum_relative_db = observations
        .activity
        .iter()
        .map(|point| point.relative_db)
        .min_by(f64::total_cmp);
    let low_activity_count = observations
        .activity
        .iter()
        .filter(|point| point.relative_db <= -40.0)
        .count();
    let low_activity_fraction = if observations.activity.is_empty() {
        0.0
    } else {
        usize_to_f64(low_activity_count) / usize_to_f64(observations.activity.len())
    };
    let raw_median_bpm = median_f64(
        observations
            .beats
            .windows(2)
            .map(|pair| 60.0 / (pair[1].time_s - pair[0].time_s))
            .collect(),
    );
    let alternating_phase_confidence = (observations.beats.len() >= 2).then(|| {
        [0_usize, 1].map(|phase| {
            let values = observations.beats.iter().skip(phase).step_by(2);
            let (sum, count) = values.fold((0.0, 0_usize), |(sum, count), beat| {
                (sum + beat.confidence.clamp(0.0, 1.0), count + 1)
            });
            sum / usize_to_f64(count)
        })
    });
    let alternating_phase_salience =
        (!observations.activity.is_empty() && observations.beats.len() >= 2).then(|| {
            [0_usize, 1].map(|phase| {
                let values = observations
                    .beats
                    .iter()
                    .skip(phase)
                    .step_by(2)
                    .map(|beat| {
                        let relative_db = observations
                            .activity
                            .iter()
                            .min_by(|left, right| {
                                (left.time_s - beat.time_s)
                                    .abs()
                                    .total_cmp(&(right.time_s - beat.time_s).abs())
                            })
                            .map_or(0.0, |point| point.relative_db);
                        beat.confidence.clamp(0.0, 1.0) * 10.0_f64.powf(relative_db / 20.0)
                    });
                let (sum, count) = values.fold((0.0, 0_usize), |(sum, count), value| {
                    (sum + value, count + 1)
                });
                sum / usize_to_f64(count)
            })
        });
    ObservationDiagnostics {
        source: observations.source.clone(),
        raw_beats: observations.beats.clone(),
        analyzed_beat_count: analysis.beats.len(),
        analyzed_downbeat_count: analysis.beats.iter().filter(|beat| beat.downbeat).count(),
        activity_point_count: observations.activity.len(),
        minimum_relative_db,
        low_activity_fraction,
        raw_median_bpm,
        alternating_phase_salience,
        alternating_phase_confidence,
        analysis_warnings: analysis.warnings.clone(),
    }
}

fn median_f64(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        f64::midpoint(values[middle - 1], values[middle])
    } else {
        values[middle]
    })
}

fn validate_beat_this_contract(model_pack: &VerifiedModelPack) -> Result<()> {
    let manifest = model_pack.manifest();
    if manifest.backend != "beat-this-rten" {
        bail!(
            "model pack {} declares backend {}, expected beat-this-rten",
            manifest.id,
            manifest.backend
        );
    }

    let contract = &manifest.feature_contract;
    if contract.sample_rate_hz != 22_050
        || contract.mel_bands != 128
        || (contract.frame_rate_hz - 50.0).abs() > f64::EPSILON
    {
        bail!(
            "model pack {} has incompatible Beat This feature contract: {} Hz, {} mel bands, {} fps",
            manifest.id,
            contract.sample_rate_hz,
            contract.mel_bands,
            contract.frame_rate_hz
        );
    }
    Ok(())
}

/// Render each recipe to a WAV and exact truth JSON in an explicit output path.
///
/// # Errors
///
/// Returns an error for invalid input or filesystem failures.
pub fn render_suite(suite_path: &Path, output: &Path) -> Result<Vec<PathBuf>> {
    let (suite, root) = load_suite(suite_path)?;
    fs::create_dir_all(output)
        .with_context(|| format!("creating output directory {}", output.display()))?;
    let mut rendered = Vec::new();
    for case in &suite.cases {
        let CaseInput::Generated { recipe } = &case.input else {
            bail!(
                "render suite {} contains external case {}",
                suite.id,
                case.id
            );
        };
        let recipe = load_recipe(&root.join(recipe))?;
        let truth = generate_truth(&recipe).map_err(anyhow::Error::msg)?;
        let wav_path = output.join(format!("{}.wav", case.id));
        render_synthetic_audio(&recipe, &truth, &wav_path)
            .with_context(|| format!("rendering {}", wav_path.display()))?;
        let truth_path = output.join(format!("{}.truth.json", case.id));
        fs::write(&truth_path, serde_json::to_vec_pretty(&truth)?)
            .with_context(|| format!("writing {}", truth_path.display()))?;
        rendered.push(wav_path);
        rendered.push(truth_path);
    }
    Ok(rendered)
}

/// Score externally produced analyses named `<case-id>.json`.
///
/// # Errors
///
/// Returns an error for missing predictions or invalid manifests and JSON.
pub fn score_prediction_directory(
    suite_path: &Path,
    prediction_directory: &Path,
) -> Result<SuiteEvaluation> {
    let (suite, root) = load_suite(suite_path)?;
    let mut cases = Vec::with_capacity(suite.cases.len());
    for case in &suite.cases {
        let truth = load_case_truth(case, &root)?;
        let prediction_path = prediction_directory.join(format!("{}.json", case.id));
        let analysis: Analysis = serde_json::from_slice(
            &fs::read(&prediction_path)
                .with_context(|| format!("reading {}", prediction_path.display()))?,
        )
        .with_context(|| format!("parsing {}", prediction_path.display()))?;
        let thresholds = case.thresholds.as_ref().unwrap_or(&suite.thresholds);
        cases.push(evaluate_analysis(&case.id, &analysis, &truth, thresholds));
    }
    Ok(suite_report(suite.id, suite.purpose, cases))
}

fn ensure_timestamped_beat_truth(
    suite: &EvaluationSuite,
    root: &Path,
    operation: &str,
) -> Result<()> {
    for case in &suite.cases {
        if load_case_truth(case, root)?.beats.is_empty() {
            bail!(
                "{operation} requires timestamped beat truth; case {} is tempo-only",
                case.id
            );
        }
    }
    Ok(())
}

fn load_case_truth(case: &EvaluationCase, root: &Path) -> Result<GeneratedTruth> {
    let truth = match &case.input {
        CaseInput::Generated { recipe } => {
            let recipe = load_recipe(&root.join(recipe))?;
            generate_truth(&recipe).map_err(anyhow::Error::msg)
        }
        CaseInput::External { truth, .. } => {
            let path = root.join(truth);
            serde_json::from_slice(
                &fs::read(&path).with_context(|| format!("reading truth {}", path.display()))?,
            )
            .with_context(|| format!("parsing truth {}", path.display()))
        }
    }?;
    truth
        .validate()
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("validating truth for case {}", case.id))?;
    if truth.id != case.id {
        bail!("case {} points to truth with id {}", case.id, truth.id);
    }
    Ok(truth)
}

fn load_suite(path: &Path) -> Result<(EvaluationSuite, PathBuf)> {
    let suite: EvaluationSuite = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("reading suite {}", path.display()))?,
    )
    .with_context(|| format!("parsing suite {}", path.display()))?;
    suite.validate().map_err(anyhow::Error::msg)?;
    let root = path
        .parent()
        .context("suite path must have a parent directory")?
        .to_path_buf();
    Ok((suite, root))
}

fn load_recipe(path: &Path) -> Result<SyntheticRecipe> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("reading recipe {}", path.display()))?,
    )
    .with_context(|| format!("parsing recipe {}", path.display()))
}

fn suite_report(
    suite_id: String,
    suite_purpose: SuitePurpose,
    cases: Vec<CaseEvaluation>,
) -> SuiteEvaluation {
    SuiteEvaluation {
        schema_version: 1,
        suite_id,
        suite_purpose,
        passed: cases.iter().all(|case| case.passed),
        cases,
    }
}

fn required_model_path(pack: &VerifiedModelPack, role: ModelArtifactRole) -> Result<PathBuf> {
    pack.path_for(role)
        .with_context(|| format!("verified model pack is missing role {role:?}"))
}

fn metric_delta(oracle: &CaseEvaluation, end_to_end: &CaseEvaluation) -> CaseMetricDelta {
    CaseMetricDelta {
        beat_f1: end_to_end.metrics.beats.f1 - oracle.metrics.beats.f1,
        downbeat_f1: end_to_end.metrics.downbeats.f1 - oracle.metrics.downbeats.f1,
        tempo_median_error_percent: subtract_options(
            end_to_end.metrics.tempo.median_absolute_error_percent,
            oracle.metrics.tempo.median_absolute_error_percent,
        ),
        tempo_p95_error_percent: subtract_options(
            end_to_end.metrics.tempo.p95_absolute_error_percent,
            oracle.metrics.tempo.p95_absolute_error_percent,
        ),
        change_recall: end_to_end.metrics.changes.recall - oracle.metrics.changes.recall,
    }
}

const fn attribution_decision(
    oracle_passed: bool,
    has_unpaired_cases: bool,
    end_to_end_passed: bool,
) -> AttributionDecision {
    if !oracle_passed {
        AttributionDecision::DeterministicEstimator
    } else if has_unpaired_cases {
        AttributionDecision::EndToEndOnly
    } else if !end_to_end_passed {
        AttributionDecision::ObservationPath
    } else {
        AttributionDecision::NoMeasuredBottleneck
    }
}

fn subtract_options(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    left.zip(right).map(|(left, right)| left - right)
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribution_does_not_invent_an_oracle_for_tempo_only_cases() {
        assert_eq!(
            attribution_decision(true, true, false),
            AttributionDecision::EndToEndOnly
        );
        assert_eq!(
            attribution_decision(true, false, false),
            AttributionDecision::ObservationPath
        );
        assert_eq!(
            attribution_decision(false, false, false),
            AttributionDecision::DeterministicEstimator
        );
    }

    #[test]
    fn standard_decoder_sweep_keeps_upstream_policy_first() {
        let policies = standard_decoder_policies();

        assert_eq!(policies.len(), 11);
        assert_eq!(policies[0].id, "upstream-default");
        assert_eq!(policies[0].options(), PeakPickingOptions::default());
        assert!(policies.iter().any(|policy| {
            (policy.logit_threshold + 3.0).abs() < f32::EPSILON
                && policy.local_max_radius_frames == 1
        }));
        assert!(
            policies
                .iter()
                .any(|policy| policy.supported_midpoints.is_some())
        );
        assert!(policies.last().is_some_and(|policy| {
            policy.id == "viterbi-edge-logit-minus-3.0-bias-2.0" && policy.sequence_path.is_some()
        }));
    }

    #[test]
    fn standard_decoder_policy_requires_a_registered_id() {
        assert_eq!(
            standard_decoder_policy("supported-midpoints-logit-minus-3.0")
                .unwrap()
                .id,
            "supported-midpoints-logit-minus-3.0"
        );
        let sequence = standard_decoder_policy("viterbi-edge-logit-minus-3.0-bias-2.0").unwrap();
        assert_eq!(
            sequence.backend_policy(),
            BeatThisDecoderPolicy::SequencePath(SequencePathOptions::default())
        );
        assert!(standard_decoder_policy("tuned-on-holdout").is_err());
    }

    #[test]
    fn estimator_policy_requires_a_registered_id() {
        assert!(validate_estimator_policy(None).is_ok());
        assert!(validate_estimator_policy(Some("metrical-consistency-v1")).is_ok());
        assert!(validate_estimator_policy(Some("sequence-phase-v1")).is_ok());
        assert!(validate_estimator_policy(Some("tuned-on-holdout")).is_err());
    }

    #[test]
    fn decoder_policy_oracle_uses_best_candidate_per_case() {
        let mut candidates = vec![
            candidate("first", &[(0.4, 4), (0.8, 8)]),
            candidate("second", &[(0.6, 6), (0.5, 5)]),
        ];

        let oracle = finalize_decoder_candidates(&mut candidates);

        assert!((candidates[0].mean_beat_f1 - 0.6).abs() < f64::EPSILON);
        assert!((candidates[0].mean_predicted_beat_count - 6.0).abs() < f64::EPSILON);
        assert!((oracle - 0.7).abs() < f64::EPSILON);
        assert!(!candidates[0].passed);
        assert_eq!(candidates[0].slices.len(), 2);
        assert_eq!(candidates[0].slices[0].tag, "first");
        assert!((candidates[0].slices[0].mean_beat_f1 - 0.4).abs() < f64::EPSILON);
        assert!(!candidates[0].slices[0].passed);
        assert!(candidates[0].slices[1].passed);
    }

    #[test]
    fn fixed_decoder_comparison_reports_improvements_and_regressions() {
        let mut candidates = vec![
            candidate("upstream-default", &[(0.4, 4), (0.8, 8)]),
            candidate("selected", &[(0.6, 6), (0.5, 5)]),
        ];
        finalize_decoder_candidates(&mut candidates);

        let comparison = compare_decoder_candidates(&candidates[0], &candidates[1]).unwrap();

        assert!((comparison.mean_beat_f1_delta + 0.05).abs() < f64::EPSILON);
        assert_eq!(comparison.improved_case_ids, ["case-0"]);
        assert_eq!(comparison.regressed_case_ids, ["case-1"]);
        assert!((comparison.slices[0].mean_beat_f1_delta - 0.2).abs() < f64::EPSILON);
        assert!((comparison.slices[1].mean_beat_f1_delta + 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn missed_beat_classifies_subthreshold_upstream_peak() {
        let mut logits = vec![-10.0; 20];
        logits[5] = -0.5;

        let evidence = missing_beat_evidence(&logits, 0.1, 0.04).unwrap();

        assert_eq!(
            evidence.class,
            MissingBeatEvidenceClass::Radius3PeakAboveMinus1
        );
        assert!((evidence.radius_3_local_peak.unwrap().logit + 0.5).abs() < f32::EPSILON);
        assert!(evidence.strongest_frame.offset_ms.abs() < f64::EPSILON);
    }

    #[test]
    fn missed_beat_distinguishes_narrow_peak_from_upstream_peak() {
        let mut logits = vec![-10.0; 20];
        logits[5] = -0.5;
        logits[8] = 1.0;

        let evidence = missing_beat_evidence(&logits, 0.1, 0.02).unwrap();

        assert_eq!(evidence.class, MissingBeatEvidenceClass::Radius1OnlyPeak);
        assert!(evidence.radius_1_local_peak.is_some());
        assert!(evidence.radius_3_local_peak.is_none());
    }

    #[test]
    fn missed_beat_reports_when_no_local_peak_exists() {
        let mut logits = vec![-10.0; 20];
        logits[5] = -0.5;
        logits[6] = 1.0;

        let evidence = missing_beat_evidence(&logits, 0.1, 0.0).unwrap();

        assert_eq!(evidence.class, MissingBeatEvidenceClass::NoLocalPeak);
        assert!(evidence.radius_1_local_peak.is_none());
        assert!((evidence.strongest_frame.logit + 0.5).abs() < f32::EPSILON);
    }

    fn candidate(id: &str, cases: &[(f64, usize)]) -> DecoderSweepCandidate {
        DecoderSweepCandidate {
            policy: DecoderPolicy {
                id: id.to_string(),
                logit_threshold: 0.0,
                local_max_radius_frames: 3,
                deduplicate_width_frames: 1,
                supported_midpoints: None,
                sequence_path: None,
            },
            passed: false,
            mean_beat_f1: 0.0,
            mean_beat_precision: 0.0,
            mean_beat_recall: 0.0,
            mean_predicted_beat_count: 0.0,
            slices: Vec::new(),
            cases: cases
                .iter()
                .enumerate()
                .map(|(index, &(f1, predicted_beat_count))| DecoderSweepCase {
                    id: format!("case-{index}"),
                    tags: vec![if index == 0 { "first" } else { "second" }.to_string()],
                    predicted_beat_count,
                    minimum_beat_f1: 0.5,
                    passed: f1 >= 0.5,
                    beats: BeatMetrics {
                        matched: 0,
                        precision: f1,
                        recall: f1,
                        f1,
                        median_absolute_error_ms: None,
                        p95_absolute_error_ms: None,
                    },
                })
                .collect(),
        }
    }
}
