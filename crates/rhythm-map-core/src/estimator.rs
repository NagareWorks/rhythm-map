use std::cmp::Ordering;

use thiserror::Error;

use crate::{
    ANALYSIS_SCHEMA_VERSION, Analysis, BeatCandidate, BeatEvent, BeatSequenceHypothesis,
    BeatSequenceHypothesisKind, ChangeKind, ChangePoint, ObservedBeat, RhythmObservations,
    RhythmSection, TempoHypothesis, TempoPoint, TempoSegment, TempoSegmentKind,
};

/// Policy used to resolve metrical half/double-time evidence before tempo estimation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MetricalSelectionPolicy {
    /// Preserve the original confidence-weighted alternating-salience rule.
    #[default]
    SalienceOnly,
    /// Require sequence and bar-phase consistency and repair supported edge runs.
    SequencePhaseV1,
}

/// Tunable policy for deterministic tempo-map inference.
#[derive(Debug, Clone)]
pub struct EstimatorOptions {
    /// Lowest tempo supported when publishing metrical alternatives.
    ///
    /// The primary local curve still preserves slower observed cadence.
    pub min_bpm: f64,
    /// Highest tempo supported when publishing metrical alternatives.
    ///
    /// The primary local curve still preserves faster observed cadence.
    pub max_bpm: f64,
    /// Lower edge of the preferred metrical band.
    pub preferred_min_bpm: f64,
    /// Upper edge of the preferred metrical band.
    pub preferred_max_bpm: f64,
    /// Odd median-filter window in inter-beat intervals.
    pub smoothing_window: usize,
    /// Longest bounded half/double-time outlier run repaired before smoothing.
    pub maximum_metrical_outlier_run: usize,
    /// Sustained relative tempo difference considered a jump.
    pub jump_ratio: f64,
    /// Relative endpoint difference considered constant.
    pub constant_ratio: f64,
    /// Maximum log2 curve error retained during simplification.
    pub curve_tolerance_octaves: f64,
    /// Beat-gap multiplier considered a discontinuity.
    pub discontinuity_factor: f64,
    /// Activity level at or below which audio is treated as silent.
    pub silence_threshold_db: f64,
    /// Shortest low-activity span treated as a rhythm discontinuity.
    pub minimum_silence_s: f64,
    /// Raw median tempo above which alternating onset salience may select half-time.
    pub half_time_min_bpm: f64,
    /// Required retained-to-discarded onset salience ratio for half-time selection.
    pub half_time_salience_ratio: f64,
    /// Evidence policy used for global and edge metrical-level decisions.
    pub metrical_selection_policy: MetricalSelectionPolicy,
    /// Required strong-to-weak accent ratio before half-bar downbeats are rejected.
    pub half_bar_downbeat_salience_ratio: f64,
    /// Longest model-smeared ramp still classified as a tempo jump.
    pub jump_transition_max_s: f64,
    /// Include the fixed calibration candidate whose pulse level may vary locally.
    #[cfg(any(feature = "experimental-policies", test))]
    pub include_local_metrical_path_hypothesis: bool,
}

impl Default for EstimatorOptions {
    fn default() -> Self {
        Self {
            min_bpm: 40.0,
            max_bpm: 320.0,
            preferred_min_bpm: 70.0,
            preferred_max_bpm: 180.0,
            smoothing_window: 7,
            maximum_metrical_outlier_run: 1,
            jump_ratio: 0.12,
            constant_ratio: 0.03,
            curve_tolerance_octaves: 0.04,
            discontinuity_factor: 3.5,
            silence_threshold_db: -40.0,
            minimum_silence_s: 0.8,
            half_time_min_bpm: 150.0,
            half_time_salience_ratio: 1.35,
            metrical_selection_policy: MetricalSelectionPolicy::SalienceOnly,
            half_bar_downbeat_salience_ratio: 1.2,
            jump_transition_max_s: 4.0,
            #[cfg(any(feature = "experimental-policies", test))]
            include_local_metrical_path_hypothesis: false,
        }
    }
}

impl EstimatorOptions {
    /// Experimental calibration candidate that repairs short, bounded runs of
    /// octave-related interval errors while preserving sustained tempo levels.
    #[cfg(any(feature = "experimental-policies", test))]
    #[must_use]
    pub fn metrical_consistency_candidate() -> Self {
        Self {
            maximum_metrical_outlier_run: 3,
            ..Self::default()
        }
    }

    /// Experimental sequence-aware policy for whole-track and one-sided edge
    /// metrical decisions. It includes the bounded-run consistency candidate.
    #[cfg(any(feature = "experimental-policies", test))]
    #[must_use]
    pub fn sequence_phase_candidate() -> Self {
        Self {
            maximum_metrical_outlier_run: 3,
            metrical_selection_policy: MetricalSelectionPolicy::SequencePhaseV1,
            ..Self::default()
        }
    }

    /// Experimental candidate that adds one locally varying, harmonic-aware
    /// real-timestamp path to the ambiguity result without changing primary beats.
    #[cfg(any(feature = "experimental-policies", test))]
    #[must_use]
    pub fn local_metrical_path_candidate() -> Self {
        Self {
            include_local_metrical_path_hypothesis: true,
            ..Self::default()
        }
    }
}

/// Invalid observation or estimator configuration.
#[derive(Debug, Error, PartialEq)]
pub enum AnalysisError {
    /// Duration or timestamp is invalid.
    #[error("invalid observation value: {0}")]
    InvalidValue(String),
    /// Beats are not strictly increasing.
    #[error("beat timestamps must be strictly increasing")]
    UnsortedBeats,
    /// Estimator configuration is inconsistent.
    #[error("invalid estimator options: {0}")]
    InvalidOptions(String),
}

/// Training-free estimator for BPM curves, segments, and timing changes.
#[derive(Debug, Clone, Default)]
pub struct TempoMapEstimator {
    options: EstimatorOptions,
}

impl TempoMapEstimator {
    /// Construct an estimator with explicit policy.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::InvalidOptions`] for an inconsistent tempo
    /// range or smoothing window.
    #[cfg(any(feature = "experimental-policies", test))]
    pub fn new(options: EstimatorOptions) -> Result<Self, AnalysisError> {
        validate_options(&options)?;
        Ok(Self { options })
    }

    pub(crate) const fn requires_harmonic_changes(&self) -> bool {
        local_metrical_path_enabled(&self.options)
    }

    /// Analyze backend-neutral beat observations.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError`] when timestamps, duration, ordering, or
    /// estimator policy is invalid.
    pub fn estimate(&self, input: &RhythmObservations) -> Result<Analysis, AnalysisError> {
        validate_options(&self.options)?;
        validate_observations(input)?;

        let PreparedObservations {
            observations: mut prepared,
            hypothesis_source,
            silence_regions,
            mut warnings,
        } = prepare_observations(input, &self.options);

        if prepared.beats.len() < 3 {
            warnings.push("too_few_beats_for_tempo_curve".to_string());
            return Ok(Analysis {
                schema_version: ANALYSIS_SCHEMA_VERSION,
                duration_s: input.duration_s,
                source: input.source.clone(),
                beats: beat_events(&prepared),
                beat_hypotheses: beat_sequence_hypotheses(
                    &prepared,
                    &hypothesis_source,
                    &self.options,
                ),
                global_bpm: None,
                tempo_hypotheses: Vec::new(),
                tempo_curve: Vec::new(),
                tempo_segments: Vec::new(),
                change_points: Vec::new(),
                rhythm_sections: vec![RhythmSection {
                    start_s: 0.0,
                    end_s: input.duration_s,
                    bpm: None,
                    stability: 0.0,
                    beat_count: prepared.beats.len(),
                }],
                warnings,
            });
        }

        let preliminary = build_tempo_estimate(&prepared, input.duration_s, &self.options);
        if let Some(repaired) =
            recover_short_transition_grids(&prepared, &preliminary.tempo_segments, &self.options)
        {
            prepared = repaired;
            warnings.push("short_transition_beat_grid_recovered".to_string());
        }
        if select_bar_level_downbeats(&mut prepared, &self.options) {
            warnings.push("bar_level_downbeats_selected".to_string());
        }

        let TempoEstimate {
            intervals,
            smoothed,
            tempo_curve,
            tempo_segments,
            repaired_metrical_run,
            repaired_interval_jitter,
        } = build_tempo_estimate(&prepared, input.duration_s, &self.options);
        if repaired_metrical_run {
            warnings.push("short_metrical_outlier_run_repaired".to_string());
        }
        if repaired_interval_jitter {
            warnings.push("quantized_interval_jitter_repaired".to_string());
        }

        let global_bpm = median(smoothed.clone());
        let beat_hypotheses =
            beat_sequence_hypotheses(&prepared, &hypothesis_source, &self.options);
        let tempo_hypotheses = metrical_hypotheses(global_bpm, &self.options);
        if tempo_hypotheses.len() > 1 {
            warnings.push("metrical_level_has_half_or_double_time_alternatives".to_string());
        }

        let mut change_points = detect_jumps(&tempo_curve, self.options.jump_ratio);
        add_segment_transitions(
            &tempo_segments,
            self.options.jump_ratio,
            self.options.jump_transition_max_s,
            &mut change_points,
        );
        add_discontinuities(
            &prepared,
            &intervals,
            self.options.discontinuity_factor,
            &silence_regions,
            &mut change_points,
        );
        change_points.sort_by(|a, b| float_order(a.time_s, b.time_s));
        deduplicate_changes(&mut change_points);

        let rhythm_sections = build_sections(&prepared, &tempo_curve, &change_points);

        Ok(Analysis {
            schema_version: ANALYSIS_SCHEMA_VERSION,
            duration_s: input.duration_s,
            source: input.source.clone(),
            beats: beat_events(&prepared),
            beat_hypotheses,
            global_bpm: Some(global_bpm),
            tempo_hypotheses,
            tempo_curve,
            tempo_segments,
            change_points,
            rhythm_sections,
            warnings,
        })
    }
}

/// Analyze backend-neutral observations with Rhythm Map's single shipping
/// policy.
///
/// Candidate policies are deliberately unavailable through this function. A
/// candidate may replace this implementation only after it passes the product
/// promotion gates; callers never need to select a musical-analysis strategy.
///
/// # Errors
///
/// Returns [`AnalysisError`] when timestamps, duration, or ordering are invalid.
pub fn analyze_observations(input: &RhythmObservations) -> Result<Analysis, AnalysisError> {
    TempoMapEstimator::default().estimate(input)
}

struct TempoEstimate {
    intervals: Vec<f64>,
    smoothed: Vec<f64>,
    tempo_curve: Vec<TempoPoint>,
    tempo_segments: Vec<TempoSegment>,
    repaired_metrical_run: bool,
    repaired_interval_jitter: bool,
}

fn build_tempo_estimate(
    input: &RhythmObservations,
    duration_s: f64,
    options: &EstimatorOptions,
) -> TempoEstimate {
    let intervals = input
        .beats
        .windows(2)
        .map(|pair| pair[1].time_s - pair[0].time_s)
        .collect::<Vec<_>>();
    let raw_bpms = intervals.iter().map(|dt| 60.0 / dt).collect::<Vec<_>>();
    let (raw_bpms, repaired_interval_jitter) = if options.metrical_selection_policy
        == MetricalSelectionPolicy::SequencePhaseV1
        && input.source.frame_rate_hz.is_some()
    {
        repair_quantized_interval_jitter(&intervals, &raw_bpms, options.jump_ratio)
    } else {
        (raw_bpms, false)
    };
    // The observed cadence is evidence, not a nuisance value to force into a
    // global preferred band. A sustained 75 -> 150 BPM change is musically
    // different from a tracker emitting one isolated half- or double-length
    // interval. The median filter below rejects the latter while preserving
    // the former. Evidence-based half-time selection happens earlier, against
    // alternating PCM salience, and alternate global interpretations remain
    // visible in `tempo_hypotheses`.
    let cadence_bpms = raw_bpms;
    let (smoothed, repaired_metrical_run) = smooth_log_tempo(
        input,
        &cadence_bpms,
        options.smoothing_window,
        options.jump_ratio,
        options.maximum_metrical_outlier_run,
    );
    let tempo_curve = smoothed
        .iter()
        .enumerate()
        .map(|(index, &bpm)| {
            let pair = &input.beats[index..=index + 1];
            let observation_confidence = pair[0].confidence.min(pair[1].confidence);
            let deviation = (cadence_bpms[index] / bpm).log2().abs();
            TempoPoint {
                time_s: f64::midpoint(pair[0].time_s, pair[1].time_s),
                bpm,
                confidence: (observation_confidence * (-8.0 * deviation).exp()).clamp(0.0, 1.0),
            }
        })
        .collect::<Vec<_>>();
    let knot_indices = simplify_curve(&tempo_curve, options.curve_tolerance_octaves);
    let tempo_segments = build_segments(
        &tempo_curve,
        &knot_indices,
        duration_s,
        options.constant_ratio,
    );
    TempoEstimate {
        intervals,
        smoothed,
        tempo_curve,
        tempo_segments,
        repaired_metrical_run,
        repaired_interval_jitter,
    }
}

fn repair_quantized_interval_jitter(
    intervals: &[f64],
    raw_bpms: &[f64],
    tolerance: f64,
) -> (Vec<f64>, bool) {
    const CONTEXT_RADIUS: usize = 3;
    let mut repaired = raw_bpms.to_vec();
    let mut changed = false;
    let mut index = 0;
    while index + 1 < intervals.len() {
        let left_start = index.saturating_sub(CONTEXT_RADIUS);
        let right_end = (index + 2 + CONTEXT_RADIUS).min(intervals.len());
        let context = intervals[left_start..index]
            .iter()
            .chain(&intervals[index + 2..right_end])
            .copied()
            .collect::<Vec<_>>();
        if context.len() < 2 {
            index += 1;
            continue;
        }
        let period = median(context.clone());
        let stable_context = context
            .iter()
            .all(|value| ((value / period) - 1.0).abs() <= tolerance);
        let pair_mean = f64::midpoint(intervals[index], intervals[index + 1]);
        let straddles_period = (intervals[index] - period) * (intervals[index + 1] - period) < 0.0;
        if stable_context
            && straddles_period
            && ((pair_mean / period) - 1.0).abs() <= tolerance
            && (intervals[index] / intervals[index + 1] - 1.0).abs() > tolerance
        {
            let bpm = 60.0 / pair_mean;
            repaired[index] = bpm;
            repaired[index + 1] = bpm;
            changed = true;
            index += 2;
        } else {
            index += 1;
        }
    }
    (repaired, changed)
}

fn beat_events(input: &RhythmObservations) -> Vec<BeatEvent> {
    input
        .beats
        .iter()
        .map(|beat| BeatEvent {
            time_s: beat.time_s,
            confidence: beat.confidence.clamp(0.0, 1.0),
            downbeat: beat.downbeat_confidence >= 0.5,
            downbeat_confidence: beat.downbeat_confidence.clamp(0.0, 1.0),
        })
        .collect()
}

fn validate_options(options: &EstimatorOptions) -> Result<(), AnalysisError> {
    if !(options.min_bpm.is_finite()
        && options.max_bpm.is_finite()
        && options.min_bpm > 0.0
        && options.max_bpm > options.min_bpm)
    {
        return Err(AnalysisError::InvalidOptions(
            "tempo range must be finite, positive, and ordered".to_string(),
        ));
    }
    if options.smoothing_window == 0 || options.smoothing_window.is_multiple_of(2) {
        return Err(AnalysisError::InvalidOptions(
            "smoothing window must be a positive odd number".to_string(),
        ));
    }
    if options.maximum_metrical_outlier_run == 0 {
        return Err(AnalysisError::InvalidOptions(
            "maximum metrical outlier run must be greater than zero".to_string(),
        ));
    }
    if !options.silence_threshold_db.is_finite()
        || options.silence_threshold_db >= 0.0
        || !options.minimum_silence_s.is_finite()
        || options.minimum_silence_s <= 0.0
        || !options.half_time_min_bpm.is_finite()
        || options.half_time_min_bpm <= 0.0
        || !options.half_time_salience_ratio.is_finite()
        || options.half_time_salience_ratio <= 1.0
        || !options.half_bar_downbeat_salience_ratio.is_finite()
        || options.half_bar_downbeat_salience_ratio <= 1.0
        || !options.jump_transition_max_s.is_finite()
        || options.jump_transition_max_s <= 0.0
    {
        return Err(AnalysisError::InvalidOptions(
            "activity and metrical-selection options must be finite and ordered".to_string(),
        ));
    }
    Ok(())
}

fn validate_observations(input: &RhythmObservations) -> Result<(), AnalysisError> {
    if !input.duration_s.is_finite() || input.duration_s < 0.0 {
        return Err(AnalysisError::InvalidValue(
            "duration must be finite and non-negative".to_string(),
        ));
    }
    for beat in &input.beats {
        if !beat.time_s.is_finite() || beat.time_s < 0.0 || beat.time_s > input.duration_s {
            return Err(AnalysisError::InvalidValue(
                "beat timestamp is outside the audio duration".to_string(),
            ));
        }
    }
    for candidate in &input.beat_candidates {
        if !candidate.time_s.is_finite()
            || candidate.time_s < 0.0
            || candidate.time_s > input.duration_s
            || !candidate.confidence.is_finite()
            || !candidate.downbeat_confidence.is_finite()
        {
            return Err(AnalysisError::InvalidValue(
                "beat candidate is outside the observation contract".to_string(),
            ));
        }
    }
    validate_audio_evidence(input)?;
    if input
        .beats
        .windows(2)
        .any(|pair| pair[1].time_s <= pair[0].time_s)
    {
        return Err(AnalysisError::UnsortedBeats);
    }
    if input
        .beat_candidates
        .windows(2)
        .any(|pair| pair[1].time_s <= pair[0].time_s)
    {
        return Err(AnalysisError::InvalidValue(
            "beat candidate timestamps must be strictly increasing".to_string(),
        ));
    }
    Ok(())
}

fn validate_audio_evidence(input: &RhythmObservations) -> Result<(), AnalysisError> {
    for point in &input.activity {
        if !point.time_s.is_finite()
            || point.time_s < 0.0
            || point.time_s > input.duration_s
            || !point.rms.is_finite()
            || point.rms < 0.0
            || !point.relative_db.is_finite()
            || point.relative_db > 0.0
        {
            return Err(AnalysisError::InvalidValue(
                "audio activity point is invalid".to_string(),
            ));
        }
    }
    for point in &input.onsets {
        if !point.time_s.is_finite()
            || point.time_s < 0.0
            || point.time_s > input.duration_s
            || !point.strength.is_finite()
            || !(0.0..=1.0).contains(&point.strength)
            || !point.low_strength.is_finite()
            || !(0.0..=1.0).contains(&point.low_strength)
            || !point.mid_strength.is_finite()
            || !(0.0..=1.0).contains(&point.mid_strength)
            || !point.high_strength.is_finite()
            || !(0.0..=1.0).contains(&point.high_strength)
        {
            return Err(AnalysisError::InvalidValue(
                "audio onset point is invalid".to_string(),
            ));
        }
    }
    for point in &input.harmonic_changes {
        if !point.time_s.is_finite()
            || point.time_s < 0.0
            || point.time_s > input.duration_s
            || !point.strength.is_finite()
            || !(0.0..=1.0).contains(&point.strength)
        {
            return Err(AnalysisError::InvalidValue(
                "audio harmonic-change point is invalid".to_string(),
            ));
        }
    }
    if input
        .activity
        .windows(2)
        .any(|pair| pair[1].time_s <= pair[0].time_s)
    {
        return Err(AnalysisError::InvalidValue(
            "audio activity timestamps must be strictly increasing".to_string(),
        ));
    }
    if input
        .onsets
        .windows(2)
        .any(|pair| pair[1].time_s <= pair[0].time_s)
    {
        return Err(AnalysisError::InvalidValue(
            "audio onset timestamps must be strictly increasing".to_string(),
        ));
    }
    if input
        .harmonic_changes
        .windows(2)
        .any(|pair| pair[1].time_s <= pair[0].time_s)
    {
        return Err(AnalysisError::InvalidValue(
            "audio harmonic-change timestamps must be strictly increasing".to_string(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct SilenceRegion {
    start_s: f64,
    end_s: f64,
    depth_db: f64,
}

struct PreparedObservations {
    observations: RhythmObservations,
    hypothesis_source: RhythmObservations,
    silence_regions: Vec<SilenceRegion>,
    warnings: Vec<String>,
}

fn prepare_observations(
    input: &RhythmObservations,
    options: &EstimatorOptions,
) -> PreparedObservations {
    let silence_regions = find_silence_regions(
        input,
        options.silence_threshold_db,
        options.minimum_silence_s,
    );
    let mut observations = input.clone();
    observations
        .beats
        .retain(|beat| !inside_silence(beat.time_s, &silence_regions));
    observations
        .beat_candidates
        .retain(|candidate| !inside_silence(candidate.time_s, &silence_regions));
    let hypothesis_source = observations.clone();
    let rejected_silent_beats = input.beats.len() - observations.beats.len();
    let (observations, repaired_edge_double_time) =
        if options.metrical_selection_policy == MetricalSelectionPolicy::SequencePhaseV1 {
            repair_edge_double_time_runs(&observations)
        } else {
            (observations, false)
        };
    let (observations, selected_half_time, rejected_inconsistent_half_time) =
        select_metrical_level(&observations, options);

    let mut warnings = Vec::new();
    if rejected_silent_beats > 0 {
        warnings.push("low_activity_beats_rejected".to_string());
    }
    if repaired_edge_double_time {
        warnings.push("edge_double_time_events_rejected".to_string());
    }
    if selected_half_time {
        warnings.push("metrical_level_selected_half_time".to_string());
    }
    if rejected_inconsistent_half_time {
        warnings.push("inconsistent_half_time_selection_rejected".to_string());
    }

    PreparedObservations {
        observations,
        hypothesis_source,
        silence_regions,
        warnings,
    }
}

fn find_silence_regions(
    input: &RhythmObservations,
    threshold_db: f64,
    minimum_duration_s: f64,
) -> Vec<SilenceRegion> {
    if input.activity.is_empty() {
        return Vec::new();
    }
    let hop_s = if input.activity.len() >= 2 {
        median(
            input
                .activity
                .windows(2)
                .map(|pair| pair[1].time_s - pair[0].time_s)
                .collect(),
        )
    } else {
        input.duration_s.max(0.001)
    };
    let mut regions = Vec::new();
    let mut start = None;
    for (index, point) in input.activity.iter().enumerate() {
        if point.relative_db <= threshold_db {
            start.get_or_insert(index);
        } else if let Some(start_index) = start.take() {
            push_silence_region(
                input,
                start_index,
                index - 1,
                hop_s,
                minimum_duration_s,
                &mut regions,
            );
        }
    }
    if let Some(start_index) = start {
        push_silence_region(
            input,
            start_index,
            input.activity.len() - 1,
            hop_s,
            minimum_duration_s,
            &mut regions,
        );
    }
    regions
}

fn push_silence_region(
    input: &RhythmObservations,
    start_index: usize,
    end_index: usize,
    hop_s: f64,
    minimum_duration_s: f64,
    regions: &mut Vec<SilenceRegion>,
) {
    let start_s = (input.activity[start_index].time_s - hop_s * 0.5).max(0.0);
    let end_s = (input.activity[end_index].time_s + hop_s * 0.5).min(input.duration_s);
    if end_s - start_s >= minimum_duration_s {
        let depth_db = input.activity[start_index..=end_index]
            .iter()
            .map(|point| point.relative_db)
            .fold(0.0_f64, f64::min);
        regions.push(SilenceRegion {
            start_s,
            end_s,
            depth_db,
        });
    }
}

fn inside_silence(time_s: f64, regions: &[SilenceRegion]) -> bool {
    regions
        .iter()
        .any(|region| time_s >= region.start_s && time_s <= region.end_s)
}

fn repair_edge_double_time_runs(input: &RhythmObservations) -> (RhythmObservations, bool) {
    if input.beats.len() < 12 || input.activity.is_empty() {
        return (input.clone(), false);
    }
    let mut removals = forward_edge_double_time_removals(input);
    let mirrored = mirror_observations(input);
    removals.extend(
        forward_edge_double_time_removals(&mirrored)
            .into_iter()
            .map(|index| input.beats.len() - index - 1),
    );
    removals.sort_unstable();
    removals.dedup();
    if removals.is_empty() {
        return (input.clone(), false);
    }

    let mut repaired = input.clone();
    repaired.beats = input
        .beats
        .iter()
        .enumerate()
        .filter(|(index, _)| removals.binary_search(index).is_err())
        .map(|(_, beat)| beat.clone())
        .collect();
    repaired
        .beat_candidates
        .extend(removals.iter().map(|&index| {
            let beat = &input.beats[index];
            BeatCandidate {
                time_s: beat.time_s,
                confidence: beat.confidence,
                downbeat_confidence: beat.downbeat_confidence,
            }
        }));
    sort_and_deduplicate_candidates(&mut repaired.beat_candidates);
    (repaired, true)
}

fn mirror_observations(input: &RhythmObservations) -> RhythmObservations {
    let mut mirrored = input.clone();
    mirrored.beats = input
        .beats
        .iter()
        .rev()
        .map(|beat| {
            let mut beat = beat.clone();
            beat.time_s = input.duration_s - beat.time_s;
            beat
        })
        .collect();
    mirrored.beat_candidates = input
        .beat_candidates
        .iter()
        .rev()
        .map(|candidate| {
            let mut candidate = candidate.clone();
            candidate.time_s = input.duration_s - candidate.time_s;
            candidate
        })
        .collect();
    mirrored.activity = input
        .activity
        .iter()
        .rev()
        .map(|point| {
            let mut point = point.clone();
            point.time_s = input.duration_s - point.time_s;
            point
        })
        .collect();
    mirrored.onsets = input
        .onsets
        .iter()
        .rev()
        .map(|point| {
            let mut point = point.clone();
            point.time_s = input.duration_s - point.time_s;
            point
        })
        .collect();
    mirrored.harmonic_changes = input
        .harmonic_changes
        .iter()
        .rev()
        .map(|point| {
            let mut point = point.clone();
            point.time_s = input.duration_s - point.time_s;
            point
        })
        .collect();
    mirrored
}

fn forward_edge_double_time_removals(input: &RhythmObservations) -> Vec<usize> {
    const ANCHOR_INTERVALS: usize = 6;
    const ANCHOR_TOLERANCE: f64 = 0.08;
    const GRID_TOLERANCE: f64 = 0.12;
    const MIDPOINT_TOLERANCE: f64 = 0.20;
    const MINIMUM_EXTRAS: usize = 4;
    const MINIMUM_RETAINED: usize = 6;
    const MINIMUM_EVIDENCE_RATIO: f64 = 1.15;

    let beats = &input.beats;
    let mut best = Vec::new();
    for boundary in ANCHOR_INTERVALS..beats.len().saturating_sub(1) {
        let anchor_intervals = beats[boundary - ANCHOR_INTERVALS..=boundary]
            .windows(2)
            .map(|pair| pair[1].time_s - pair[0].time_s)
            .collect::<Vec<_>>();
        let period = median(anchor_intervals.clone());
        if anchor_intervals
            .iter()
            .any(|interval| ((interval / period) - 1.0).abs() > ANCHOR_TOLERANCE)
        {
            continue;
        }
        let first_gap = beats[boundary + 1].time_s - beats[boundary].time_s;
        if ((first_gap / period) - 0.5).abs() > GRID_TOLERANCE {
            continue;
        }

        let mut current = boundary;
        let mut cursor = boundary + 1;
        let mut extras = Vec::new();
        let mut retained = Vec::new();
        let mut valid = true;
        while cursor < beats.len() {
            let expected = beats[current].time_s + period;
            let latest = expected + period * GRID_TOLERANCE;
            let mut candidate = None;
            let mut index = cursor;
            while index < beats.len() && beats[index].time_s <= latest {
                if (beats[index].time_s - expected).abs() <= period * GRID_TOLERANCE {
                    candidate = Some(index);
                }
                index += 1;
            }
            let Some(candidate) = candidate else {
                valid = false;
                break;
            };
            let midpoint = f64::midpoint(beats[current].time_s, beats[candidate].time_s);
            if (cursor..candidate)
                .any(|extra| (beats[extra].time_s - midpoint).abs() > period * MIDPOINT_TOLERANCE)
            {
                valid = false;
                break;
            }
            extras.extend(cursor..candidate);
            retained.push(candidate);
            current = candidate;
            cursor = candidate + 1;
        }
        if !valid
            || extras.len() < MINIMUM_EXTRAS
            || retained.len() < MINIMUM_RETAINED
            || extras.len() <= best.len()
        {
            continue;
        }
        let retained_evidence = mean_event_evidence(input, &retained);
        let extra_evidence = mean_event_evidence(input, &extras).max(1e-9);
        if retained_evidence / extra_evidence >= MINIMUM_EVIDENCE_RATIO {
            best = extras;
        }
    }
    best
}

fn mean_event_evidence(input: &RhythmObservations, indices: &[usize]) -> f64 {
    indices
        .iter()
        .map(|&index| {
            let beat = &input.beats[index];
            let activity = nearest_activity_db(&input.activity, beat.time_s)
                .map_or(1.0, |relative_db| 10.0_f64.powf(relative_db / 20.0));
            beat.confidence.clamp(0.0, 1.0) * activity
                + 0.25 * beat.downbeat_confidence.clamp(0.0, 1.0)
        })
        .sum::<f64>()
        / usize_to_f64(indices.len())
}

fn select_metrical_level(
    input: &RhythmObservations,
    options: &EstimatorOptions,
) -> (RhythmObservations, bool, bool) {
    if input.beats.len() < 8 || input.activity.is_empty() {
        return (input.clone(), false, false);
    }
    let raw_bpms = input
        .beats
        .windows(2)
        .map(|pair| 60.0 / (pair[1].time_s - pair[0].time_s))
        .collect::<Vec<_>>();
    let raw_bpm = median(raw_bpms);
    let half_bpm = raw_bpm * 0.5;
    if raw_bpm < options.half_time_min_bpm
        || half_bpm < options.preferred_min_bpm
        || half_bpm > options.preferred_max_bpm
    {
        return (input.clone(), false, false);
    }

    let phase_scores = [0_usize, 1].map(|phase| mean_phase_salience(input, phase));
    let selected_phase = usize::from(phase_scores[1] > phase_scores[0]);
    let retained = phase_scores[selected_phase];
    let discarded = phase_scores[1 - selected_phase].max(1e-9);
    if retained / discarded < options.half_time_salience_ratio {
        return (input.clone(), false, false);
    }

    let mut selected = input.clone();
    selected.beats = input
        .beats
        .iter()
        .skip(selected_phase)
        .step_by(2)
        .cloned()
        .collect();
    selected.beat_candidates.extend(
        input
            .beats
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 2 != selected_phase)
            .map(|(_, beat)| BeatCandidate {
                time_s: beat.time_s,
                confidence: beat.confidence,
                downbeat_confidence: beat.downbeat_confidence,
            }),
    );
    sort_and_deduplicate_candidates(&mut selected.beat_candidates);
    if options.metrical_selection_policy == MetricalSelectionPolicy::SequencePhaseV1 {
        transfer_discarded_downbeats(input, &mut selected);
        if folded_downbeat_spacing_is_inconsistent(&selected) {
            return (input.clone(), false, true);
        }
    }
    (selected, true, false)
}

fn sort_and_deduplicate_candidates(candidates: &mut Vec<BeatCandidate>) {
    candidates.sort_by(|left, right| float_order(left.time_s, right.time_s));
    candidates.dedup_by(|left, right| (left.time_s - right.time_s).abs() <= f64::EPSILON);
}

fn transfer_discarded_downbeats(input: &RhythmObservations, selected: &mut RhythmObservations) {
    for source in input
        .beats
        .iter()
        .filter(|beat| beat.downbeat_confidence >= 0.5)
    {
        if let Some(target) = selected.beats.iter_mut().min_by(|left, right| {
            (left.time_s - source.time_s)
                .abs()
                .total_cmp(&(right.time_s - source.time_s).abs())
        }) {
            target.downbeat_confidence = target
                .downbeat_confidence
                .max(source.downbeat_confidence * 0.9);
        }
    }
}

fn folded_downbeat_spacing_is_inconsistent(input: &RhythmObservations) -> bool {
    let indices = input
        .beats
        .iter()
        .enumerate()
        .filter_map(|(index, beat)| (beat.downbeat_confidence >= 0.5).then_some(index))
        .collect::<Vec<_>>();
    if indices.len() < 3 {
        return false;
    }
    let spacings = indices
        .windows(2)
        .map(|pair| usize_to_f64(pair[1] - pair[0]))
        .collect::<Vec<_>>();
    median(spacings) < 2.0
}

fn mean_phase_salience(input: &RhythmObservations, phase: usize) -> f64 {
    let values = input
        .beats
        .iter()
        .skip(phase)
        .step_by(2)
        .map(|beat| {
            let activity = nearest_activity_db(&input.activity, beat.time_s)
                .map_or(1.0, |relative_db| 10.0_f64.powf(relative_db / 20.0));
            beat.confidence.clamp(0.0, 1.0) * activity
        })
        .collect::<Vec<_>>();
    values.iter().sum::<f64>() / usize_to_f64(values.len())
}

fn nearest_activity_db(activity: &[crate::AudioActivityPoint], time_s: f64) -> Option<f64> {
    activity
        .iter()
        .min_by(|left, right| {
            float_order((left.time_s - time_s).abs(), (right.time_s - time_s).abs())
        })
        .map(|point| point.relative_db)
}

fn select_bar_level_downbeats(input: &mut RhythmObservations, options: &EstimatorOptions) -> bool {
    const MINIMUM_CANDIDATES_PER_RUN: usize = 6;

    if input.activity.is_empty() {
        return false;
    }
    let candidates = input
        .beats
        .iter()
        .enumerate()
        .filter(|(_, beat)| beat.downbeat_confidence >= 0.5)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if candidates.len() < MINIMUM_CANDIDATES_PER_RUN {
        return false;
    }

    let mut changed = false;
    let mut selected_runs = Vec::new();
    let mut run_start = 0;
    for run_end in 1..=candidates.len() {
        let continues_half_bar_run = run_end < candidates.len()
            && candidates[run_end].saturating_sub(candidates[run_end - 1]) == 2;
        if continues_half_bar_run {
            continue;
        }

        let run = &candidates[run_start..run_end];
        if run.len() >= MINIMUM_CANDIDATES_PER_RUN {
            let phase_scores =
                [0_usize, 1].map(|phase| mean_candidate_phase_salience(input, run, phase));
            let selected_phase = usize::from(phase_scores[1] > phase_scores[0]);
            let retained = phase_scores[selected_phase];
            let discarded = phase_scores[1 - selected_phase].max(1e-9);
            if retained / discarded >= options.half_bar_downbeat_salience_ratio {
                for (ordinal, &beat_index) in run.iter().enumerate() {
                    if ordinal % 2 != selected_phase {
                        input.beats[beat_index].downbeat_confidence = 0.0;
                    }
                }
                selected_runs.push((run_start, run_end, selected_phase));
                changed = true;
            }
        }
        run_start = run_end;
    }
    for (start, end, phase) in selected_runs {
        let run = &candidates[start..end];
        let first_selected = run[phase];
        if first_selected >= 4 && start > 0 {
            changed |= realign_boundary_downbeat(
                input,
                first_selected - 4,
                candidates[start - 1],
                options.half_bar_downbeat_salience_ratio,
            );
        }
        let last_selected = run
            .iter()
            .enumerate()
            .rev()
            .find(|(ordinal, _)| ordinal % 2 == phase)
            .map(|(_, &beat_index)| beat_index)
            .expect("selected downbeat run contains its selected phase");
        if end < candidates.len() && last_selected + 4 < input.beats.len() {
            changed |= realign_boundary_downbeat(
                input,
                last_selected + 4,
                candidates[end],
                options.half_bar_downbeat_salience_ratio,
            );
        }
    }
    changed
}

fn realign_boundary_downbeat(
    input: &mut RhythmObservations,
    expected_index: usize,
    displaced_index: usize,
    salience_ratio: f64,
) -> bool {
    if expected_index.abs_diff(displaced_index) > 1
        || input.beats[expected_index].downbeat_confidence >= 0.5
        || input.beats[displaced_index].downbeat_confidence < 0.5
    {
        return false;
    }
    let expected_salience = beat_activity_salience(input, expected_index);
    let displaced_salience = beat_activity_salience(input, displaced_index).max(1e-9);
    if expected_salience / displaced_salience < salience_ratio {
        return false;
    }
    input.beats[expected_index].downbeat_confidence =
        input.beats[displaced_index].downbeat_confidence;
    input.beats[displaced_index].downbeat_confidence = 0.0;
    true
}

fn beat_activity_salience(input: &RhythmObservations, beat_index: usize) -> f64 {
    nearest_activity_db(&input.activity, input.beats[beat_index].time_s)
        .map_or(0.0, |relative_db| 10.0_f64.powf(relative_db / 20.0))
}

fn mean_candidate_phase_salience(
    input: &RhythmObservations,
    candidates: &[usize],
    phase: usize,
) -> f64 {
    let values = candidates
        .iter()
        .skip(phase)
        .step_by(2)
        .filter_map(|&index| nearest_activity_db(&input.activity, input.beats[index].time_s))
        .map(|relative_db| 10.0_f64.powf(relative_db / 20.0))
        .collect::<Vec<_>>();
    values.iter().sum::<f64>() / usize_to_f64(values.len())
}

fn smooth_log_tempo(
    input: &RhythmObservations,
    values: &[f64],
    window: usize,
    edge_ratio: f64,
    maximum_metrical_outlier_run: usize,
) -> (Vec<f64>, bool) {
    let radius = window / 2;
    let (bounded_runs_repaired, mut repaired_metrical_run) = repair_bounded_metrical_runs(
        input,
        values,
        radius,
        edge_ratio,
        maximum_metrical_outlier_run,
    );
    let mut metrical_outliers_repaired = bounded_runs_repaired.clone();
    for index in 0..bounded_runs_repaired.len() {
        let left_start = index.saturating_sub(radius);
        let right_end = (index + radius + 1).min(bounded_runs_repaired.len());
        let left = &bounded_runs_repaired[left_start..index];
        let right = &bounded_runs_repaired[index + 1..right_end];
        if left.is_empty() || right.is_empty() {
            continue;
        }
        let left_center = median(left.to_vec());
        let right_center = median(right.to_vec());
        if (left_center / right_center).ln().abs() > edge_ratio.ln_1p() {
            continue;
        }
        let context = f64::midpoint(left_center, right_center);
        let octave_offset = (bounded_runs_repaired[index] / context).log2();
        let nearest_octave = octave_offset.round();
        if nearest_octave.abs() >= 1.0
            && (octave_offset - nearest_octave).abs() <= (1.0 + edge_ratio).log2()
            && metrical_repair_has_observation_support(
                input,
                index,
                bounded_runs_repaired[index],
                context,
            )
        {
            metrical_outliers_repaired[index] = context;
            repaired_metrical_run = true;
        }
    }

    let smoothed = (0..metrical_outliers_repaired.len())
        .map(|index| {
            let start = index.saturating_sub(1);
            let end = (index + 2).min(metrical_outliers_repaired.len());
            let neighborhood = &metrical_outliers_repaired[start..end];
            let minimum = neighborhood.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = neighborhood.iter().copied().fold(0.0_f64, f64::max);
            if maximum / minimum > 1.0 + edge_ratio {
                return metrical_outliers_repaired[index];
            }
            let mean_log = metrical_outliers_repaired[start..end]
                .iter()
                .map(|value| value.log2())
                .sum::<f64>()
                / usize_to_f64(end - start);
            2.0_f64.powf(mean_log)
        })
        .collect();
    (smoothed, repaired_metrical_run)
}

fn repair_bounded_metrical_runs(
    input: &RhythmObservations,
    values: &[f64],
    radius: usize,
    edge_ratio: f64,
    maximum_run: usize,
) -> (Vec<f64>, bool) {
    let mut repaired = values.to_vec();
    if maximum_run < 2 || values.len() < 4 {
        return (repaired, false);
    }

    let octave_tolerance = (1.0 + edge_ratio).log2();
    let mut changed = false;
    let mut start = 1;
    while start + 2 < values.len() {
        let largest_run = maximum_run.min(values.len() - start - 1);
        let candidate = (2..=largest_run).rev().find_map(|run_length| {
            let end = start + run_length;
            let left_start = start.saturating_sub(radius);
            let right_end = (end + radius).min(values.len());
            let left = &values[left_start..start];
            let right = &values[end..right_end];
            if left.is_empty() || right.is_empty() {
                return None;
            }
            let left_center = median(left.to_vec());
            let right_center = median(right.to_vec());
            if (left_center / right_center).ln().abs() > edge_ratio.ln_1p() {
                return None;
            }
            let context = f64::midpoint(left_center.log2(), right_center.log2());
            values[start..end]
                .iter()
                .enumerate()
                .all(|(offset, value)| {
                    let octave_offset = value.log2() - context;
                    let nearest_octave = octave_offset.round();
                    nearest_octave.abs() >= 1.0
                        && (octave_offset - nearest_octave).abs() <= octave_tolerance
                        && metrical_repair_has_observation_support(
                            input,
                            start + offset,
                            *value,
                            2.0_f64.powf(context),
                        )
                })
                .then_some((end, left_center, right_center))
        });

        let Some((end, left_center, right_center)) = candidate else {
            start += 1;
            continue;
        };
        let span = usize_to_f64(end - start + 1);
        for (offset, value) in repaired[start..end].iter_mut().enumerate() {
            let position = usize_to_f64(offset + 1) / span;
            *value = 2.0_f64
                .powf(left_center.log2() + position * (right_center.log2() - left_center.log2()));
        }
        changed = true;
        start = end;
    }
    (repaired, changed)
}

fn metrical_repair_has_observation_support(
    input: &RhythmObservations,
    interval_index: usize,
    observed_bpm: f64,
    context_bpm: f64,
) -> bool {
    if observed_bpm >= context_bpm {
        return input.source.frame_rate_hz.is_some();
    }
    let octave_shift = (context_bpm / observed_bpm).log2().round();
    if !(1.0..=4.0).contains(&octave_shift) {
        return false;
    }
    let subdivision_count = if octave_shift < 1.5 {
        2
    } else if octave_shift < 2.5 {
        4
    } else if octave_shift < 3.5 {
        8
    } else {
        16
    };
    let Some(pair) = input.beats.get(interval_index..=interval_index + 1) else {
        return false;
    };
    let start_s = pair[0].time_s;
    let interval_s = pair[1].time_s - start_s;
    let pulse_s = interval_s / usize_to_f64(subdivision_count);
    (1..subdivision_count).all(|subdivision| {
        let expected_s = start_s + usize_to_f64(subdivision) * pulse_s;
        has_candidate_near(
            &input.beat_candidates,
            expected_s,
            0.2 * pulse_s,
            start_s,
            pair[1].time_s,
        )
    })
}

fn has_candidate_near(
    candidates: &[BeatCandidate],
    expected_s: f64,
    tolerance_s: f64,
    interval_start_s: f64,
    interval_end_s: f64,
) -> bool {
    let insertion = candidates.partition_point(|candidate| candidate.time_s < expected_s);
    [insertion.checked_sub(1), Some(insertion)]
        .into_iter()
        .flatten()
        .filter_map(|index| candidates.get(index))
        .any(|candidate| {
            candidate.time_s > interval_start_s
                && candidate.time_s < interval_end_s
                && (candidate.time_s - expected_s).abs() <= tolerance_s
        })
}

fn metrical_hypotheses(global_bpm: f64, options: &EstimatorOptions) -> Vec<TempoHypothesis> {
    [(-1_i8, 0.5_f64), (0, 1.0), (1, 0.5)]
        .into_iter()
        .filter_map(|(level, score)| {
            let bpm = global_bpm * 2.0_f64.powi(i32::from(level));
            (bpm >= options.min_bpm && bpm <= options.max_bpm).then_some(TempoHypothesis {
                bpm,
                relative_score: score,
                metrical_level: level,
            })
        })
        .collect()
}

#[derive(Debug, Clone)]
struct HypothesisBeat {
    time_s: f64,
    confidence: f64,
    downbeat_confidence: f64,
    selected: bool,
}

type BeatHypothesisCandidate = (
    BeatSequenceHypothesisKind,
    i8,
    Option<u8>,
    Vec<HypothesisBeat>,
);

fn beat_sequence_hypotheses(
    input: &RhythmObservations,
    hypothesis_source: &RhythmObservations,
    options: &EstimatorOptions,
) -> Vec<BeatSequenceHypothesis> {
    let selected = input
        .beats
        .iter()
        .map(|beat| HypothesisBeat {
            time_s: beat.time_s,
            confidence: beat.confidence,
            downbeat_confidence: beat.downbeat_confidence,
            selected: true,
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Vec::new();
    }
    if selected.len() < 3 {
        return vec![BeatSequenceHypothesis {
            kind: BeatSequenceHypothesisKind::Selected,
            metrical_level: 0,
            phase: None,
            relative_score: 1.0,
            beat_times_s: selected.iter().map(|event| event.time_s).collect(),
        }];
    }

    let mut scored = vec![(
        BeatSequenceHypothesisKind::Selected,
        0_i8,
        None,
        selected.clone(),
    )];
    if selected.len() >= 8 && sequence_bpm_is_in_range(&selected, -1, options) {
        for phase in [0_usize, 1] {
            scored.push((
                BeatSequenceHypothesisKind::HalfTime,
                -1,
                Some(u8::try_from(phase).expect("alternating phase fits u8")),
                selected.iter().skip(phase).step_by(2).cloned().collect(),
            ));
        }
    }

    let mut doubled = selected.clone();
    let mut additions = Vec::new();
    for pair in selected.windows(2) {
        let gap = pair[1].time_s - pair[0].time_s;
        let midpoint = f64::midpoint(pair[0].time_s, pair[1].time_s);
        let maximum_offset = gap * 0.2;
        if let Some(candidate) = input
            .beat_candidates
            .iter()
            .filter(|candidate| candidate.time_s > pair[0].time_s)
            .filter(|candidate| candidate.time_s < pair[1].time_s)
            .filter(|candidate| (candidate.time_s - midpoint).abs() <= maximum_offset)
            .max_by(|left, right| {
                left.confidence.total_cmp(&right.confidence).then_with(|| {
                    (right.time_s - midpoint)
                        .abs()
                        .total_cmp(&(left.time_s - midpoint).abs())
                })
            })
        {
            additions.push(HypothesisBeat {
                time_s: candidate.time_s,
                confidence: candidate.confidence,
                downbeat_confidence: candidate.downbeat_confidence,
                selected: false,
            });
        }
    }
    if additions.len() >= 3 && sequence_bpm_is_in_range(&selected, 1, options) {
        doubled.extend(additions);
        doubled.sort_by(|left, right| float_order(left.time_s, right.time_s));
        doubled.dedup_by(|left, right| (left.time_s - right.time_s).abs() <= f64::EPSILON);
        scored.push((BeatSequenceHypothesisKind::DoubleTime, 1, None, doubled));
    }
    if local_metrical_path_enabled(options)
        && let Some(path) = locally_varying_metrical_path(hypothesis_source, options)
    {
        scored.push((BeatSequenceHypothesisKind::LocallyVarying, 0, None, path));
    }

    score_beat_sequence_hypotheses(input, scored)
}

fn score_beat_sequence_hypotheses(
    input: &RhythmObservations,
    scored: Vec<BeatHypothesisCandidate>,
) -> Vec<BeatSequenceHypothesis> {
    let mut hypotheses = scored
        .into_iter()
        .map(|(kind, metrical_level, phase, events)| {
            let score = beat_sequence_score(input, &events);
            (
                BeatSequenceHypothesis {
                    kind,
                    metrical_level,
                    phase,
                    relative_score: score,
                    beat_times_s: events.iter().map(|event| event.time_s).collect(),
                },
                score,
            )
        })
        .collect::<Vec<_>>();
    let maximum_score = hypotheses
        .iter()
        .map(|(_, score)| *score)
        .fold(0.0_f64, f64::max)
        .max(1e-9);
    for (hypothesis, score) in &mut hypotheses {
        hypothesis.relative_score = (*score / maximum_score).clamp(0.0, 1.0);
    }
    hypotheses
        .into_iter()
        .map(|(hypothesis, _)| hypothesis)
        .collect()
}

#[cfg(any(feature = "experimental-policies", test))]
const fn local_metrical_path_enabled(options: &EstimatorOptions) -> bool {
    options.include_local_metrical_path_hypothesis
}

#[cfg(not(any(feature = "experimental-policies", test)))]
const fn local_metrical_path_enabled(_options: &EstimatorOptions) -> bool {
    false
}

fn locally_varying_metrical_path(
    input: &RhythmObservations,
    options: &EstimatorOptions,
) -> Option<Vec<HypothesisBeat>> {
    const MINIMUM_EVENTS: usize = 8;
    const EVENT_THRESHOLD: f64 = 0.95;
    const TEMPO_WEIGHT: f64 = 2.0;
    const METRICAL_SWITCH_COST: f64 = 0.5;
    const HARMONIC_WEIGHT: f64 = 5.0;
    let candidates = &input.beat_candidates;
    if candidates.len() < MINIMUM_EVENTS
        || input.harmonic_changes.len() < candidates.len().saturating_div(2)
    {
        return None;
    }
    let minimum_interval = 60.0 / options.max_bpm;
    let maximum_interval = 60.0 / options.min_bpm;
    let count = candidates.len();
    let mut scores = vec![f64::NEG_INFINITY; count * count];
    let mut back = vec![usize::MAX; count * count];
    let event_score = |index: usize| {
        candidates[index].confidence
            + 0.1 * candidates[index].downbeat_confidence
            + HARMONIC_WEIGHT
                * nearest_harmonic_change_strength(input, candidates[index].time_s).unwrap_or(0.0)
            - EVENT_THRESHOLD
    };
    for first in 0..count {
        if candidates[first].time_s > maximum_interval {
            break;
        }
        for second in first + 1..count {
            let interval = candidates[second].time_s - candidates[first].time_s;
            if interval > maximum_interval {
                break;
            }
            if interval >= minimum_interval {
                scores[first * count + second] = event_score(first) + event_score(second);
            }
        }
    }
    for previous in 0..count {
        for current in previous + 1..count {
            let state = previous * count + current;
            if !scores[state].is_finite() {
                continue;
            }
            let previous_interval = candidates[current].time_s - candidates[previous].time_s;
            for next in current + 1..count {
                let next_interval = candidates[next].time_s - candidates[current].time_s;
                if next_interval > maximum_interval {
                    break;
                }
                if next_interval < minimum_interval {
                    continue;
                }
                let log_ratio = (next_interval / previous_interval).ln();
                let ordinary = TEMPO_WEIGHT * log_ratio.powi(2);
                let octave = METRICAL_SWITCH_COST
                    + TEMPO_WEIGHT * (log_ratio.abs() - std::f64::consts::LN_2).powi(2);
                let next_score = scores[state] + event_score(next) - ordinary.min(octave);
                let next_state = current * count + next;
                if next_score > scores[next_state] {
                    scores[next_state] = next_score;
                    back[next_state] = previous;
                }
            }
        }
    }
    let path = best_local_metrical_path_indices(
        candidates,
        &scores,
        &back,
        input.duration_s,
        maximum_interval,
    )?;
    if path.len() < MINIMUM_EVENTS {
        return None;
    }
    let events = path
        .into_iter()
        .map(|index| {
            let candidate = &candidates[index];
            HypothesisBeat {
                time_s: candidate.time_s,
                confidence: candidate.confidence,
                downbeat_confidence: candidate.downbeat_confidence,
                selected: input
                    .beats
                    .iter()
                    .any(|beat| (beat.time_s - candidate.time_s).abs() <= f64::EPSILON),
            }
        })
        .collect::<Vec<_>>();
    let differs_from_selected = events.len() != input.beats.len()
        || events
            .iter()
            .zip(&input.beats)
            .any(|(event, beat)| (event.time_s - beat.time_s).abs() > f64::EPSILON);
    differs_from_selected.then_some(events)
}

fn best_local_metrical_path_indices(
    candidates: &[BeatCandidate],
    scores: &[f64],
    back: &[usize],
    duration_s: f64,
    maximum_interval: f64,
) -> Option<Vec<usize>> {
    let count = candidates.len();
    let mut terminal = None;
    for previous in 0..count {
        for (current, candidate) in candidates.iter().enumerate().skip(previous + 1) {
            if candidate.time_s < duration_s - maximum_interval {
                continue;
            }
            let state = previous * count + current;
            if scores[state].is_finite() && terminal.is_none_or(|(_, _, best)| scores[state] > best)
            {
                terminal = Some((previous, current, scores[state]));
            }
        }
    }
    let (mut previous, mut current, _) = terminal?;
    let mut path = vec![current, previous];
    loop {
        let predecessor = back[previous * count + current];
        if predecessor == usize::MAX {
            break;
        }
        current = previous;
        previous = predecessor;
        path.push(previous);
    }
    path.reverse();
    Some(path)
}

fn nearest_harmonic_change_strength(input: &RhythmObservations, time_s: f64) -> Option<f64> {
    input
        .harmonic_changes
        .iter()
        .min_by(|left, right| {
            (left.time_s - time_s)
                .abs()
                .total_cmp(&(right.time_s - time_s).abs())
        })
        .filter(|point| (point.time_s - time_s).abs() <= 0.02)
        .map(|point| point.strength)
}

fn sequence_bpm_is_in_range(
    selected: &[HypothesisBeat],
    metrical_level: i8,
    options: &EstimatorOptions,
) -> bool {
    let intervals = selected
        .windows(2)
        .map(|pair| pair[1].time_s - pair[0].time_s)
        .collect::<Vec<_>>();
    if intervals.is_empty() {
        return false;
    }
    let bpm = 60.0 / median(intervals) * 2.0_f64.powi(i32::from(metrical_level));
    bpm >= options.min_bpm && bpm <= options.max_bpm
}

fn beat_sequence_score(input: &RhythmObservations, events: &[HypothesisBeat]) -> f64 {
    if events.len() < 3 {
        return 0.0;
    }
    let evidence = events
        .iter()
        .map(|event| hypothesis_beat_evidence(input, event))
        .collect::<Vec<_>>();
    let mean_event_evidence = evidence.iter().sum::<f64>() / usize_to_f64(evidence.len());
    let intervals = events
        .windows(2)
        .map(|pair| pair[1].time_s - pair[0].time_s)
        .collect::<Vec<_>>();
    let median_interval = median(intervals.clone());
    let mean_log_error = intervals
        .iter()
        .map(|interval| (interval / median_interval).ln().abs())
        .sum::<f64>()
        / usize_to_f64(intervals.len());
    let interval_continuity = (-4.0 * mean_log_error).exp();
    let total_selected_evidence = input
        .beats
        .iter()
        .map(|beat| HypothesisBeat {
            time_s: beat.time_s,
            confidence: beat.confidence,
            downbeat_confidence: beat.downbeat_confidence,
            selected: true,
        })
        .map(|event| hypothesis_beat_evidence(input, &event))
        .sum::<f64>();
    let retained_selected_evidence = events
        .iter()
        .zip(&evidence)
        .filter_map(|(event, evidence)| event.selected.then_some(*evidence))
        .sum::<f64>();
    let selected_evidence_retention = if total_selected_evidence > 0.0 {
        (retained_selected_evidence / total_selected_evidence).clamp(0.0, 1.0)
    } else {
        usize_to_f64(events.iter().filter(|event| event.selected).count())
            / usize_to_f64(input.beats.len().max(1))
    };
    0.45 * mean_event_evidence + 0.30 * interval_continuity + 0.25 * selected_evidence_retention
}

fn hypothesis_beat_evidence(input: &RhythmObservations, event: &HypothesisBeat) -> f64 {
    let confidence = event.confidence.clamp(0.0, 1.0);
    let activity = nearest_activity_db(&input.activity, event.time_s)
        .map_or(confidence, |relative_db| {
            10.0_f64.powf(relative_db / 20.0).clamp(0.0, 1.0)
        });
    0.75 * confidence + 0.20 * activity + 0.05 * event.downbeat_confidence.clamp(0.0, 1.0)
}

fn simplify_curve(points: &[TempoPoint], tolerance: f64) -> Vec<usize> {
    if points.len() <= 2 {
        return (0..points.len()).collect();
    }
    let mut keep = vec![0, points.len() - 1];
    simplify_range(points, 0, points.len() - 1, tolerance, &mut keep);
    keep.sort_unstable();
    keep.dedup();
    keep
}

fn simplify_range(
    points: &[TempoPoint],
    start: usize,
    end: usize,
    tolerance: f64,
    keep: &mut Vec<usize>,
) {
    if end <= start + 1 {
        return;
    }
    let time_span = points[end].time_s - points[start].time_s;
    if time_span <= f64::EPSILON {
        return;
    }
    let start_log = points[start].bpm.log2();
    let end_log = points[end].bpm.log2();
    let candidate = (start + 1..end)
        .map(|index| {
            let ratio = (points[index].time_s - points[start].time_s) / time_span;
            let expected = start_log + ratio * (end_log - start_log);
            (index, (points[index].bpm.log2() - expected).abs())
        })
        .max_by(|left, right| float_order(left.1, right.1));

    if let Some((index, error)) = candidate
        && error > tolerance
    {
        keep.push(index);
        simplify_range(points, start, index, tolerance, keep);
        simplify_range(points, index, end, tolerance, keep);
    }
}

fn build_segments(
    curve: &[TempoPoint],
    knots: &[usize],
    duration_s: f64,
    constant_ratio: f64,
) -> Vec<TempoSegment> {
    if curve.is_empty() {
        return Vec::new();
    }
    if knots.len() == 1 {
        return vec![TempoSegment {
            start_s: 0.0,
            end_s: duration_s,
            kind: TempoSegmentKind::Constant,
            start_bpm: curve[0].bpm,
            end_bpm: curve[0].bpm,
            confidence: curve[0].confidence,
        }];
    }

    knots
        .windows(2)
        .enumerate()
        .map(|(segment_index, pair)| {
            let start_index = pair[0];
            let end_index = pair[1];
            let start_bpm = curve[start_index].bpm;
            let end_bpm = curve[end_index].bpm;
            let ratio = (end_bpm / start_bpm).ln().abs();
            let confidence = curve[start_index..=end_index]
                .iter()
                .map(|point| point.confidence)
                .sum::<f64>()
                / usize_to_f64(end_index - start_index + 1);
            TempoSegment {
                start_s: if segment_index == 0 {
                    0.0
                } else {
                    curve[start_index].time_s
                },
                end_s: if end_index == curve.len() - 1 {
                    duration_s
                } else {
                    curve[end_index].time_s
                },
                kind: if ratio <= constant_ratio.ln_1p() {
                    TempoSegmentKind::Constant
                } else {
                    TempoSegmentKind::Ramp
                },
                start_bpm,
                end_bpm,
                confidence,
            }
        })
        .collect()
}

fn detect_jumps(curve: &[TempoPoint], jump_ratio: f64) -> Vec<ChangePoint> {
    if curve.len() < 5 {
        return Vec::new();
    }
    let mut changes = Vec::new();
    for index in 2..curve.len() - 2 {
        let before = median(curve[index - 2..index].iter().map(|p| p.bpm).collect());
        let after = median(curve[index..index + 2].iter().map(|p| p.bpm).collect());
        let ratio = (after / before).ln().abs();
        if ratio >= jump_ratio.ln_1p() {
            let score = ((ratio - jump_ratio.ln_1p()) / 0.25).clamp(0.0, 1.0);
            changes.push(ChangePoint {
                time_s: curve[index].time_s,
                kind: ChangeKind::TempoJump,
                score,
                before_bpm: Some(before),
                after_bpm: Some(after),
            });
        }
    }
    changes
}

#[derive(Debug, Clone, Copy)]
struct SegmentTransition {
    ramp_start: usize,
    ramp_end: usize,
    before_bpm: f64,
    after_bpm: f64,
    confidence: f64,
}

impl SegmentTransition {
    fn start_s(self, segments: &[TempoSegment]) -> f64 {
        segments[self.ramp_start].start_s
    }

    fn end_s(self, segments: &[TempoSegment]) -> f64 {
        segments[self.ramp_end - 1].end_s
    }

    fn duration_s(self, segments: &[TempoSegment]) -> f64 {
        self.end_s(segments) - self.start_s(segments)
    }

    fn is_bracketed(self, segments: &[TempoSegment]) -> bool {
        self.ramp_start > 0 && self.ramp_end < segments.len()
    }
}

fn segment_transitions(segments: &[TempoSegment], jump_ratio: f64) -> Vec<SegmentTransition> {
    let mut transitions = Vec::new();
    let mut index = 0;
    while index < segments.len() {
        if segments[index].kind != TempoSegmentKind::Ramp {
            index += 1;
            continue;
        }
        let ramp_start = index;
        while index < segments.len() {
            if segments[index].kind == TempoSegmentKind::Ramp {
                index += 1;
                continue;
            }
            let segment = &segments[index];
            let mean_bpm = f64::midpoint(segment.start_bpm, segment.end_bpm);
            let implied_beats = (segment.end_s - segment.start_s) * mean_bpm / 60.0;
            let bridges_another_ramp = implied_beats < 6.0
                && index + 1 < segments.len()
                && segments[index + 1].kind == TempoSegmentKind::Ramp;
            if bridges_another_ramp {
                index += 1;
                continue;
            }
            break;
        }
        let ramp_end = index;
        let first = &segments[ramp_start];
        let last = &segments[ramp_end - 1];
        let before_bpm = if ramp_start > 0 {
            segments[ramp_start - 1].end_bpm
        } else {
            first.start_bpm
        };
        let after_bpm = if ramp_end < segments.len() {
            segments[ramp_end].start_bpm
        } else {
            last.end_bpm
        };
        let ratio = (after_bpm / before_bpm).ln().abs();
        if ratio < jump_ratio.ln_1p() {
            continue;
        }
        let confidence = segments[ramp_start..ramp_end]
            .iter()
            .map(|segment| segment.confidence)
            .sum::<f64>()
            / usize_to_f64(ramp_end - ramp_start);
        transitions.push(SegmentTransition {
            ramp_start,
            ramp_end,
            before_bpm,
            after_bpm,
            confidence,
        });
    }
    transitions
}

#[derive(Debug, Clone, Copy)]
struct RegularGrid {
    origin: f64,
    period: f64,
    maximum_error: f64,
}

#[derive(Debug, Clone, Copy)]
struct TransitionGridContext {
    start_s: f64,
    end_s: f64,
    left: RegularGrid,
    right: RegularGrid,
}

fn fit_regular_grid(beats: &[&ObservedBeat]) -> Option<RegularGrid> {
    if beats.len() < 6 {
        return None;
    }
    let mean_index = (usize_to_f64(beats.len()) - 1.0) * 0.5;
    let mean_time = beats.iter().map(|beat| beat.time_s).sum::<f64>() / usize_to_f64(beats.len());
    let (numerator, denominator) =
        beats
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(numerator, denominator), (index, beat)| {
                let centered_index = usize_to_f64(index) - mean_index;
                (
                    numerator + centered_index * (beat.time_s - mean_time),
                    denominator + centered_index.powi(2),
                )
            });
    let period = numerator / denominator;
    if !period.is_finite() || period <= 0.0 {
        return None;
    }
    let origin = mean_time - period * mean_index;
    let maximum_error = beats
        .iter()
        .enumerate()
        .map(|(index, beat)| (beat.time_s - origin - period * usize_to_f64(index)).abs())
        .fold(0.0_f64, f64::max);
    Some(RegularGrid {
        origin,
        period,
        maximum_error,
    })
}

fn fit_regular_grid_before_boundary(beats: &[&ObservedBeat]) -> Option<(RegularGrid, usize)> {
    (6..=beats.len()).rev().find_map(|end| {
        let grid = fit_regular_grid(&beats[..end])?;
        (grid.maximum_error <= grid.period * 0.08).then_some((grid, end))
    })
}

fn fit_regular_grid_after_boundary(beats: &[&ObservedBeat]) -> Option<(RegularGrid, usize)> {
    (0..=beats.len().saturating_sub(6)).find_map(|start| {
        let grid = fit_regular_grid(&beats[start..])?;
        (grid.maximum_error <= grid.period * 0.08).then_some((grid, start))
    })
}

fn fit_transition_grid_context(
    beats: &[ObservedBeat],
    segments: &[TempoSegment],
    transition: SegmentTransition,
) -> Option<TransitionGridContext> {
    let mut start_s = transition.start_s(segments);
    let mut end_s = transition.end_s(segments);
    let left_segment = &segments[transition.ramp_start - 1];
    let right_segment = &segments[transition.ramp_end];
    let left_beats = beats
        .iter()
        .filter(|beat| beat.time_s >= left_segment.start_s && beat.time_s < start_s)
        .collect::<Vec<_>>();
    let right_beats = beats
        .iter()
        .filter(|beat| beat.time_s > end_s && beat.time_s <= right_segment.end_s)
        .collect::<Vec<_>>();
    let (left, left_end) = fit_regular_grid_before_boundary(&left_beats)?;
    let (right, right_start) = fit_regular_grid_after_boundary(&right_beats)?;
    if left_end < left_beats.len() {
        start_s = start_s.min(f64::midpoint(
            left_beats[left_end - 1].time_s,
            left_beats[left_end].time_s,
        ));
    }
    if right_start > 0 {
        end_s = end_s.max(right_beats[right_start].time_s - right.period * 0.5);
    }
    Some(TransitionGridContext {
        start_s,
        end_s,
        left,
        right,
    })
}

fn recover_short_transition_grids(
    input: &RhythmObservations,
    segments: &[TempoSegment],
    options: &EstimatorOptions,
) -> Option<RhythmObservations> {
    let mut repaired = input.clone();
    let mut changed = false;
    for transition in segment_transitions(segments, options.jump_ratio) {
        if !transition.is_bracketed(segments)
            || transition.duration_s(segments) > options.jump_transition_max_s
        {
            continue;
        }
        if let Some(beats) = recover_transition_grid(&repaired.beats, segments, transition, options)
        {
            repaired.beats = beats;
            changed = true;
        }
    }
    changed.then_some(repaired)
}

fn recover_transition_grid(
    beats: &[ObservedBeat],
    segments: &[TempoSegment],
    transition: SegmentTransition,
    options: &EstimatorOptions,
) -> Option<Vec<ObservedBeat>> {
    let context = fit_transition_grid_context(beats, segments, transition)?;
    let start_s = context.start_s;
    let end_s = context.end_s;
    let left_grid = context.left;
    let right_grid = context.right;
    if (right_grid.period / left_grid.period).ln().abs() < options.jump_ratio.ln_1p() {
        return None;
    }

    let shorter_period = left_grid.period.min(right_grid.period);
    let longer_period = left_grid.period.max(right_grid.period);
    let transition_evidence = beats
        .iter()
        .filter(|beat| {
            beat.time_s >= start_s - longer_period && beat.time_s <= end_s + longer_period
        })
        .collect::<Vec<_>>();
    let has_duplicate_or_missed_event = transition_evidence.windows(2).any(|pair| {
        let interval = pair[1].time_s - pair[0].time_s;
        interval < shorter_period * 0.65
            || indicates_missed_event(interval, left_grid.period, right_grid.period)
    });
    if !has_duplicate_or_missed_event {
        return None;
    }

    let mut grid_time = right_grid.origin;
    while grid_time - right_grid.period >= start_s {
        grid_time -= right_grid.period;
    }
    while grid_time < start_s {
        grid_time += right_grid.period;
    }
    let first_right_grid_time = grid_time;
    while grid_time <= end_s && distance_to_grid(grid_time, left_grid) > shorter_period * 0.12 {
        grid_time += right_grid.period;
    }
    if grid_time > end_s {
        grid_time = first_right_grid_time;
    }
    let recovery_start_s = beats
        .iter()
        .filter(|beat| beat.time_s <= grid_time)
        .min_by(|left, right| {
            float_order(
                (left.time_s - grid_time).abs(),
                (right.time_s - grid_time).abs(),
            )
        })
        .filter(|beat| (beat.time_s - grid_time).abs() <= shorter_period * 0.12)
        .map_or(grid_time, |beat| beat.time_s);
    let original = beats
        .iter()
        .filter(|beat| beat.time_s >= recovery_start_s && beat.time_s <= end_s)
        .collect::<Vec<_>>();
    let mut replacement = Vec::new();
    while grid_time <= end_s {
        let matched = original
            .iter()
            .copied()
            .min_by(|left, right| {
                float_order(
                    (left.time_s - grid_time).abs(),
                    (right.time_s - grid_time).abs(),
                )
            })
            .filter(|beat| (beat.time_s - grid_time).abs() < right_grid.period * 0.45);
        replacement.push(ObservedBeat {
            time_s: grid_time,
            confidence: matched.map_or(transition.confidence * 0.5, |beat| beat.confidence),
            downbeat_confidence: matched.map_or(0.0, |beat| beat.downbeat_confidence),
        });
        grid_time += right_grid.period;
    }
    if replacement.len() == original.len()
        && replacement
            .iter()
            .zip(&original)
            .all(|(repaired, raw)| (repaired.time_s - raw.time_s).abs() <= right_grid.period * 0.08)
    {
        return None;
    }

    let mut result = beats
        .iter()
        .filter(|beat| beat.time_s < recovery_start_s || beat.time_s > end_s)
        .cloned()
        .chain(replacement)
        .collect::<Vec<_>>();
    result.sort_by(|left, right| float_order(left.time_s, right.time_s));
    Some(result)
}

fn distance_to_grid(time_s: f64, grid: RegularGrid) -> f64 {
    let position = (time_s - grid.origin) / grid.period;
    (position - position.round()).abs() * grid.period
}

fn indicates_missed_event(interval: f64, left_period: f64, right_period: f64) -> bool {
    let matches_single = |period: f64| (interval / period - 1.0).abs() <= 0.2;
    if matches_single(left_period) || matches_single(right_period) {
        return false;
    }
    [left_period, right_period].iter().any(|period| {
        let multiple = interval / period;
        (1.6..=3.25).contains(&multiple) && (multiple - multiple.round()).abs() <= 0.15
    })
}

fn add_segment_transitions(
    segments: &[TempoSegment],
    jump_ratio: f64,
    jump_transition_max_s: f64,
    changes: &mut Vec<ChangePoint>,
) {
    for transition in segment_transitions(segments, jump_ratio) {
        let first = &segments[transition.ramp_start];
        let last = &segments[transition.ramp_end - 1];
        if transition.is_bracketed(segments)
            && transition.duration_s(segments) <= jump_transition_max_s
        {
            changes.push(ChangePoint {
                time_s: first.start_s,
                kind: ChangeKind::TempoJump,
                score: (transition.confidence
                    * (transition.after_bpm / transition.before_bpm).ln().abs()
                    / 0.25)
                    .clamp(0.0, 1.0),
                before_bpm: Some(transition.before_bpm),
                after_bpm: Some(transition.after_bpm),
            });
            continue;
        }
        if transition.ramp_start > 0 {
            changes.push(ChangePoint {
                time_s: first.start_s,
                kind: ChangeKind::RampBoundary,
                score: transition.confidence,
                before_bpm: Some(transition.before_bpm),
                after_bpm: Some(first.start_bpm),
            });
        }
        if transition.ramp_end < segments.len() {
            changes.push(ChangePoint {
                time_s: last.end_s,
                kind: ChangeKind::RampBoundary,
                score: transition.confidence,
                before_bpm: Some(last.end_bpm),
                after_bpm: Some(transition.after_bpm),
            });
        }
    }
}

fn add_discontinuities(
    input: &RhythmObservations,
    intervals: &[f64],
    factor: f64,
    silence_regions: &[SilenceRegion],
    changes: &mut Vec<ChangePoint>,
) {
    let typical = median(intervals.to_vec());
    for (index, &interval) in intervals.iter().enumerate() {
        if interval > typical * factor && interval > 1.0 {
            changes.push(ChangePoint {
                time_s: f64::midpoint(input.beats[index].time_s, input.beats[index + 1].time_s),
                kind: ChangeKind::RhythmDiscontinuity,
                score: (1.0 - typical / interval).clamp(0.0, 1.0),
                before_bpm: None,
                after_bpm: None,
            });
        }
    }
    for region in silence_regions {
        changes.push(ChangePoint {
            time_s: f64::midpoint(region.start_s, region.end_s),
            kind: ChangeKind::RhythmDiscontinuity,
            score: ((-region.depth_db - 20.0) / 60.0).clamp(0.0, 1.0),
            before_bpm: None,
            after_bpm: None,
        });
    }
}

fn deduplicate_changes(changes: &mut Vec<ChangePoint>) {
    let mut result: Vec<ChangePoint> = Vec::new();
    for change in changes.drain(..) {
        if let Some(previous) = result.last_mut()
            && (previous.time_s - change.time_s).abs() < 0.75
            && previous.kind == change.kind
        {
            if change.score > previous.score {
                *previous = change;
            }
            continue;
        }
        result.push(change);
    }
    *changes = result;
}

fn build_sections(
    input: &RhythmObservations,
    curve: &[TempoPoint],
    changes: &[ChangePoint],
) -> Vec<RhythmSection> {
    let mut boundaries = vec![0.0];
    boundaries.extend(changes.iter().map(|change| change.time_s));
    boundaries.push(input.duration_s);
    boundaries.sort_by(|a, b| float_order(*a, *b));
    boundaries.dedup_by(|left, right| (*left - *right).abs() < 0.5);

    boundaries
        .windows(2)
        .filter(|pair| pair[1] > pair[0])
        .map(|pair| {
            let values = curve
                .iter()
                .filter(|point| point.time_s >= pair[0] && point.time_s < pair[1])
                .map(|point| point.bpm)
                .collect::<Vec<_>>();
            let bpm = (!values.is_empty()).then(|| median(values.clone()));
            let stability = bpm.map_or(0.0, |center| {
                let mean_deviation = values
                    .iter()
                    .map(|value| (value / center).log2().abs())
                    .sum::<f64>()
                    / usize_to_f64(values.len());
                (1.0 - mean_deviation * 8.0).clamp(0.0, 1.0)
            });
            let beat_count = input
                .beats
                .iter()
                .filter(|beat| beat.time_s >= pair[0] && beat.time_s < pair[1])
                .count();
            RhythmSection {
                start_s: pair[0],
                end_s: pair[1],
                bpm,
                stability,
                beat_count,
            }
        })
        .collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| float_order(*a, *b));
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f64::midpoint(values[middle - 1], values[middle])
    } else {
        values[middle]
    }
}

fn float_order(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioActivityPoint, BeatCandidate, ModelInfo, ObservedBeat};

    fn observations_from_bpms(bpms: &[f64]) -> RhythmObservations {
        let mut time = 0.0;
        let mut beats = vec![ObservedBeat {
            time_s: time,
            confidence: 0.95,
            downbeat_confidence: 0.9,
        }];
        for (index, bpm) in bpms.iter().enumerate() {
            time += 60.0 / bpm;
            beats.push(ObservedBeat {
                time_s: time,
                confidence: 0.95,
                downbeat_confidence: if (index + 1).is_multiple_of(4) {
                    0.9
                } else {
                    0.1
                },
            });
        }
        RhythmObservations {
            duration_s: time + 0.25,
            beats,
            beat_candidates: Vec::new(),
            activity: Vec::new(),
            onsets: Vec::new(),
            harmonic_changes: Vec::new(),
            source: ModelInfo {
                backend: "test".to_string(),
                model: "synthetic".to_string(),
                version: Some("1".to_string()),
                frame_rate_hz: None,
            },
        }
    }

    #[test]
    fn constant_tempo_stays_constant() {
        let input = observations_from_bpms(&[120.0; 32]);
        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        assert!((analysis.global_bpm.unwrap() - 120.0).abs() < 0.01);
        assert!(
            analysis
                .tempo_segments
                .iter()
                .all(|segment| segment.kind == TempoSegmentKind::Constant)
        );
        assert!(
            analysis
                .change_points
                .iter()
                .all(|change| change.kind != ChangeKind::TempoJump)
        );
    }

    #[test]
    fn observed_cadence_outside_hypothesis_range_is_not_clamped() {
        for expected_bpm in [28.0, 360.0] {
            let input = observations_from_bpms(&[expected_bpm; 16]);
            let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
            assert!((analysis.global_bpm.unwrap() - expected_bpm).abs() < 0.01);
            assert!(
                analysis
                    .tempo_curve
                    .iter()
                    .all(|point| (point.bpm - expected_bpm).abs() < 0.01)
            );
            assert!(
                analysis
                    .tempo_hypotheses
                    .iter()
                    .all(|hypothesis| { (40.0..=320.0).contains(&hypothesis.bpm) })
            );
        }
    }

    #[test]
    fn irrelevant_candidates_do_not_change_shipping_analysis() {
        let baseline = observations_from_bpms(&[120.0; 16]);
        let mut with_candidates = baseline.clone();
        with_candidates.beat_candidates = vec![BeatCandidate {
            time_s: 0.25,
            confidence: 0.99,
            downbeat_confidence: 0.99,
        }];

        assert_eq!(
            TempoMapEstimator::default().estimate(&baseline).unwrap(),
            TempoMapEstimator::default()
                .estimate(&with_candidates)
                .unwrap()
        );
    }

    #[test]
    fn candidate_timestamps_must_be_strictly_increasing() {
        let mut input = observations_from_bpms(&[120.0; 8]);
        input.beat_candidates = vec![
            BeatCandidate {
                time_s: 1.0,
                confidence: 0.5,
                downbeat_confidence: 0.1,
            },
            BeatCandidate {
                time_s: 0.5,
                confidence: 0.6,
                downbeat_confidence: 0.2,
            },
        ];

        assert!(matches!(
            TempoMapEstimator::default().estimate(&input),
            Err(AnalysisError::InvalidValue(message))
                if message.contains("candidate timestamps")
        ));
    }

    #[test]
    fn sustained_step_creates_jump() {
        let mut bpms = vec![120.0; 20];
        bpms.extend(vec![160.0; 20]);
        let input = observations_from_bpms(&bpms);
        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        assert!(
            analysis
                .change_points
                .iter()
                .any(|change| change.kind == ChangeKind::TempoJump),
            "segments: {:?}; changes: {:?}",
            analysis.tempo_segments,
            analysis.change_points
        );
        assert!(analysis.rhythm_sections.len() >= 2);
        assert!(
            !analysis
                .warnings
                .contains(&"short_transition_beat_grid_recovered".to_string())
        );
        assert_eq!(analysis.beats, beat_events(&input));
    }

    #[test]
    fn sustained_octave_related_step_preserves_both_tempos() {
        let mut bpms = vec![75.0; 20];
        bpms.extend(vec![150.0; 20]);
        let input = observations_from_bpms(&bpms);
        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        let before = analysis
            .tempo_curve
            .iter()
            .filter(|point| point.time_s < 12.0)
            .map(|point| point.bpm)
            .collect::<Vec<_>>();
        let after = analysis
            .tempo_curve
            .iter()
            .filter(|point| point.time_s > 17.0)
            .map(|point| point.bpm)
            .collect::<Vec<_>>();

        assert!((median(before) - 75.0).abs() < 0.01);
        assert!((median(after) - 150.0).abs() < 0.01);
        assert!(
            analysis
                .change_points
                .iter()
                .any(|change| change.kind == ChangeKind::TempoJump)
        );
    }

    #[test]
    fn sustained_fast_tempo_is_not_folded_without_salience_evidence() {
        let input = observations_from_bpms(&[240.0; 32]);
        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();

        assert!((analysis.global_bpm.unwrap() - 240.0).abs() < 0.01);
    }

    #[test]
    fn isolated_missed_beat_does_not_create_a_half_tempo_point() {
        let mut input = observations_from_bpms(&[120.0; 32]);
        let missing = input.beats.remove(16);
        input.beat_candidates.push(BeatCandidate {
            time_s: missing.time_s,
            confidence: 0.2,
            downbeat_confidence: missing.downbeat_confidence,
        });
        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();

        assert!(
            analysis
                .tempo_curve
                .iter()
                .all(|point| (point.bpm - 120.0).abs() < 0.01)
        );
        assert!(
            analysis
                .warnings
                .contains(&"short_metrical_outlier_run_repaired".to_string())
        );
    }

    #[test]
    fn unsupported_octave_rubato_gesture_is_not_flattened() {
        let input = observations_from_bpms(&[120.0, 120.0, 60.0, 120.0, 120.0]);
        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();

        assert!((analysis.tempo_curve[2].bpm - 60.0).abs() < 0.01);
        assert!(
            !analysis
                .warnings
                .contains(&"short_metrical_outlier_run_repaired".to_string())
        );
    }

    #[test]
    fn fixed_frame_backend_can_regularize_isolated_dense_interval() {
        let mut input = observations_from_bpms(&[120.0, 120.0, 240.0, 120.0, 120.0]);
        input.source.frame_rate_hz = Some(50.0);
        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();

        assert!((analysis.tempo_curve[2].bpm - 120.0).abs() < 0.01);
        assert!(analysis.tempo_curve[2].confidence < 0.01);
        assert!(
            analysis
                .warnings
                .contains(&"short_metrical_outlier_run_repaired".to_string())
        );
    }

    #[test]
    fn bounded_metrical_run_candidate_repairs_three_missing_events() {
        let mut bpms = vec![150.0; 8];
        bpms.extend([75.0; 3]);
        bpms.extend([150.0; 8]);
        let mut input = observations_from_bpms(&bpms);
        input.beat_candidates.extend((8..11).map(|index| {
            let pair = &input.beats[index..=index + 1];
            BeatCandidate {
                time_s: f64::midpoint(pair[0].time_s, pair[1].time_s),
                confidence: 0.2,
                downbeat_confidence: 0.0,
            }
        }));

        let baseline = TempoMapEstimator::default().estimate(&input).unwrap();
        assert!(
            baseline
                .tempo_curve
                .iter()
                .any(|point| (point.bpm - 150.0).abs() > 1.0)
        );

        let candidate = TempoMapEstimator::new(EstimatorOptions::metrical_consistency_candidate())
            .unwrap()
            .estimate(&input)
            .unwrap();
        assert!(
            candidate
                .tempo_curve
                .iter()
                .all(|point| (point.bpm - 150.0).abs() < 0.01)
        );
        assert!(
            candidate
                .warnings
                .contains(&"short_metrical_outlier_run_repaired".to_string())
        );
    }

    #[test]
    fn bounded_metrical_run_candidate_preserves_sustained_octave_step() {
        let mut bpms = vec![75.0; 20];
        bpms.extend([150.0; 20]);
        let input = observations_from_bpms(&bpms);
        let analysis = TempoMapEstimator::new(EstimatorOptions::metrical_consistency_candidate())
            .unwrap()
            .estimate(&input)
            .unwrap();

        assert!((analysis.tempo_curve.first().unwrap().bpm - 75.0).abs() < 0.01);
        assert!((analysis.tempo_curve.last().unwrap().bpm - 150.0).abs() < 0.01);
    }

    #[test]
    fn non_metrical_rubato_gesture_is_not_flattened() {
        let input = observations_from_bpms(&[100.0, 100.0, 80.0, 100.0, 100.0]);
        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();

        assert!((analysis.tempo_curve[2].bpm - 80.0).abs() < 0.01);
        let candidate = TempoMapEstimator::new(EstimatorOptions::sequence_phase_candidate())
            .unwrap()
            .estimate(&input)
            .unwrap();
        assert!((candidate.tempo_curve[2].bpm - 80.0).abs() < 0.01);
    }

    #[test]
    fn sequence_policy_repairs_opposing_quantized_edge_jitter() {
        let mut bpms = vec![200.0; 12];
        bpms.extend([187.5, 230.769_230_769, 187.5]);
        let mut input = observations_from_bpms(&bpms);
        input.source.frame_rate_hz = Some(50.0);
        let analysis = TempoMapEstimator::new(EstimatorOptions::sequence_phase_candidate())
            .unwrap()
            .estimate(&input)
            .unwrap();

        assert!(
            analysis
                .tempo_curve
                .iter()
                .all(|point| (point.bpm / 200.0 - 1.0).abs() < 0.1),
            "curve: {:?}",
            analysis.tempo_curve
        );
        assert!(
            analysis
                .warnings
                .contains(&"quantized_interval_jitter_repaired".to_string())
        );
    }

    #[test]
    fn sequence_policy_does_not_quantize_exact_timestamp_observations() {
        let mut bpms = vec![200.0; 12];
        bpms.extend([187.5, 230.769_230_769, 187.5]);
        let input = observations_from_bpms(&bpms);
        let analysis = TempoMapEstimator::new(EstimatorOptions::sequence_phase_candidate())
            .unwrap()
            .estimate(&input)
            .unwrap();

        assert!(
            !analysis
                .warnings
                .contains(&"quantized_interval_jitter_repaired".to_string())
        );
    }

    #[test]
    fn gradual_change_produces_ramp_segment() {
        let bpms = (0..48)
            .map(|index| 100.0 + f64::from(index) * 50.0 / 47.0)
            .collect::<Vec<_>>();
        let input = observations_from_bpms(&bpms);
        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        assert!(
            analysis
                .tempo_segments
                .iter()
                .any(|segment| segment.kind == TempoSegmentKind::Ramp)
        );
    }

    #[test]
    fn rejects_unsorted_beats() {
        let mut input = observations_from_bpms(&[120.0, 120.0, 120.0]);
        input.beats[2].time_s = input.beats[1].time_s;
        assert_eq!(
            TempoMapEstimator::default().estimate(&input),
            Err(AnalysisError::UnsortedBeats)
        );
    }

    #[test]
    fn alternating_weak_onsets_select_half_time() {
        let mut input = observations_from_bpms(&[180.0; 40]);
        input.activity = input
            .beats
            .iter()
            .enumerate()
            .map(|(index, beat)| AudioActivityPoint {
                time_s: beat.time_s,
                rms: if index.is_multiple_of(2) { 1.0 } else { 0.1 },
                relative_db: if index.is_multiple_of(2) { 0.0 } else { -20.0 },
            })
            .collect();

        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        assert!((analysis.global_bpm.unwrap() - 90.0).abs() < 0.01);
        assert_eq!(analysis.beats.len(), 21);
        assert!(
            analysis
                .warnings
                .contains(&"metrical_level_selected_half_time".to_string())
        );
        let doubled = analysis
            .beat_hypotheses
            .iter()
            .find(|hypothesis| hypothesis.kind == BeatSequenceHypothesisKind::DoubleTime)
            .expect("discarded real beats remain available as a double-time alternative");
        assert_eq!(doubled.beat_times_s.len(), input.beats.len());
        assert!(doubled.beat_times_s.iter().all(|time_s| {
            input
                .beats
                .iter()
                .any(|beat| (beat.time_s - *time_s).abs() <= f64::EPSILON)
        }));
    }

    #[test]
    fn beat_sequence_hypotheses_never_invent_timestamps() {
        let mut input = observations_from_bpms(&[120.0; 12]);
        input.beat_candidates = input
            .beats
            .windows(2)
            .map(|pair| BeatCandidate {
                time_s: f64::midpoint(pair[0].time_s, pair[1].time_s),
                confidence: 0.6,
                downbeat_confidence: 0.0,
            })
            .collect();
        let supported = input
            .beats
            .iter()
            .map(|beat| beat.time_s)
            .chain(
                input
                    .beat_candidates
                    .iter()
                    .map(|candidate| candidate.time_s),
            )
            .collect::<Vec<_>>();

        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();

        assert_eq!(analysis.schema_version, 3);
        assert_eq!(analysis.beat_hypotheses.len(), 4);
        assert!(analysis.beat_hypotheses.iter().any(|hypothesis| {
            hypothesis.kind == BeatSequenceHypothesisKind::HalfTime && hypothesis.phase == Some(0)
        }));
        assert!(analysis.beat_hypotheses.iter().any(|hypothesis| {
            hypothesis.kind == BeatSequenceHypothesisKind::HalfTime && hypothesis.phase == Some(1)
        }));
        assert!(analysis.beat_hypotheses.iter().any(|hypothesis| {
            hypothesis.kind == BeatSequenceHypothesisKind::DoubleTime
                && hypothesis.beat_times_s.len() == input.beats.len() * 2 - 1
        }));
        assert!(analysis.beat_hypotheses.iter().all(|hypothesis| {
            (0.0..=1.0).contains(&hypothesis.relative_score)
                && hypothesis
                    .beat_times_s
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
                && hypothesis.beat_times_s.iter().all(|time_s| {
                    supported
                        .iter()
                        .any(|supported| (*supported - *time_s).abs() <= f64::EPSILON)
                })
        }));
    }

    #[test]
    fn only_local_metrical_candidate_requests_harmonic_changes() {
        assert!(!TempoMapEstimator::default().requires_harmonic_changes());
        let candidate =
            TempoMapEstimator::new(EstimatorOptions::local_metrical_path_candidate()).unwrap();
        assert!(candidate.requires_harmonic_changes());
    }

    #[test]
    fn local_metrical_candidate_uses_harmonic_evidence_and_real_timestamps() {
        let candidates = (0..=32)
            .map(|index| {
                let time_s = usize_to_f64(index) * 0.25;
                let retained = time_s >= 4.0 || index.is_multiple_of(2);
                BeatCandidate {
                    time_s,
                    confidence: if retained { 0.9 } else { 0.2 },
                    downbeat_confidence: if retained { 0.5 } else { 0.0 },
                }
            })
            .collect::<Vec<_>>();
        let mut input = observations_from_bpms(&[240.0; 31]);
        input.duration_s = 8.0;
        input.beats = candidates
            .iter()
            .map(|candidate| ObservedBeat {
                time_s: candidate.time_s,
                confidence: candidate.confidence,
                downbeat_confidence: candidate.downbeat_confidence,
            })
            .collect();
        input.beat_candidates.clone_from(&candidates);
        input.harmonic_changes = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| crate::AudioHarmonicChangePoint {
                time_s: candidate.time_s,
                strength: if candidate.time_s >= 4.0 || index.is_multiple_of(2) {
                    0.1
                } else {
                    0.0
                },
            })
            .collect();

        let analysis = TempoMapEstimator::new(EstimatorOptions::local_metrical_path_candidate())
            .unwrap()
            .estimate(&input)
            .unwrap();
        let path = analysis
            .beat_hypotheses
            .iter()
            .find(|hypothesis| hypothesis.kind == BeatSequenceHypothesisKind::LocallyVarying)
            .expect("candidate should expose one locally varying path");
        let intervals = path
            .beat_times_s
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect::<Vec<_>>();

        assert!(
            intervals
                .iter()
                .any(|interval| (*interval - 0.5).abs() < 1e-9)
        );
        assert!(
            intervals
                .iter()
                .any(|interval| (*interval - 0.25).abs() < 1e-9)
        );
        assert!(path.beat_times_s.iter().all(|time_s| {
            candidates
                .iter()
                .any(|candidate| (candidate.time_s - *time_s).abs() <= f64::EPSILON)
        }));
    }

    #[test]
    fn beat_sequence_hypotheses_respect_the_supported_tempo_range() {
        let mut input = observations_from_bpms(&[240.0; 12]);
        input.beat_candidates = input
            .beats
            .windows(2)
            .map(|pair| BeatCandidate {
                time_s: f64::midpoint(pair[0].time_s, pair[1].time_s),
                confidence: 0.9,
                downbeat_confidence: 0.0,
            })
            .collect();

        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();

        assert!(
            analysis
                .beat_hypotheses
                .iter()
                .any(|hypothesis| { hypothesis.kind == BeatSequenceHypothesisKind::HalfTime })
        );
        assert!(
            analysis
                .beat_hypotheses
                .iter()
                .all(|hypothesis| { hypothesis.kind != BeatSequenceHypothesisKind::DoubleTime })
        );
    }

    #[test]
    fn equally_salient_fast_beats_stay_at_the_observed_level() {
        let mut input = observations_from_bpms(&[180.0; 40]);
        input.activity = input
            .beats
            .iter()
            .map(|beat| AudioActivityPoint {
                time_s: beat.time_s,
                rms: 1.0,
                relative_db: 0.0,
            })
            .collect();

        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        assert!((analysis.global_bpm.unwrap() - 180.0).abs() < 0.01);
        assert_eq!(analysis.beats.len(), 41);
    }

    #[test]
    fn sequence_policy_rejects_half_time_with_one_downbeat_per_retained_beat() {
        let mut input = observations_from_bpms(&[200.0; 40]);
        for (index, beat) in input.beats.iter_mut().enumerate() {
            beat.downbeat_confidence = if index.is_multiple_of(2) { 0.9 } else { 0.0 };
        }
        input.activity = input
            .beats
            .iter()
            .enumerate()
            .map(|(index, beat)| AudioActivityPoint {
                time_s: beat.time_s,
                rms: if index.is_multiple_of(2) { 1.0 } else { 0.5 },
                relative_db: if index.is_multiple_of(2) { 0.0 } else { -6.0 },
            })
            .collect();

        let analysis = TempoMapEstimator::new(EstimatorOptions::sequence_phase_candidate())
            .unwrap()
            .estimate(&input)
            .unwrap();
        assert!((analysis.global_bpm.unwrap() - 200.0).abs() < 0.01);
        assert_eq!(analysis.beats.len(), input.beats.len());
        assert!(
            analysis
                .warnings
                .contains(&"inconsistent_half_time_selection_rejected".to_string())
        );
    }

    #[test]
    fn sequence_policy_transfers_downbeats_when_selecting_real_subdivisions() {
        let mut input = observations_from_bpms(&[180.0; 40]);
        input.activity = input
            .beats
            .iter()
            .enumerate()
            .map(|(index, beat)| AudioActivityPoint {
                time_s: beat.time_s,
                rms: if index.is_multiple_of(2) { 1.0 } else { 0.1 },
                relative_db: if index.is_multiple_of(2) { 0.0 } else { -20.0 },
            })
            .collect();

        let analysis = TempoMapEstimator::new(EstimatorOptions::sequence_phase_candidate())
            .unwrap()
            .estimate(&input)
            .unwrap();
        assert!((analysis.global_bpm.unwrap() - 90.0).abs() < 0.01);
        assert!(analysis.beats.iter().any(|beat| beat.downbeat));
    }

    #[test]
    fn sequence_policy_rejects_supported_double_time_events_at_track_end() {
        let mut input = observations_from_bpms(&[60.0; 30]);
        for integer in 13..=29 {
            input.beats.push(ObservedBeat {
                time_s: f64::from(integer) + 0.5,
                confidence: 0.7,
                downbeat_confidence: 0.0,
            });
        }
        input
            .beats
            .sort_by(|left, right| left.time_s.total_cmp(&right.time_s));
        input.activity = input
            .beats
            .iter()
            .map(|beat| {
                let retained = beat.time_s.fract().abs() < 1e-9;
                AudioActivityPoint {
                    time_s: beat.time_s,
                    rms: if retained { 1.0 } else { 0.2 },
                    relative_db: if retained { 0.0 } else { -14.0 },
                }
            })
            .collect();

        let analysis = TempoMapEstimator::new(EstimatorOptions::sequence_phase_candidate())
            .unwrap()
            .estimate(&input)
            .unwrap();
        assert!((analysis.global_bpm.unwrap() - 60.0).abs() < 0.01);
        assert_eq!(analysis.beats.len(), 31);
        assert!(
            analysis
                .warnings
                .contains(&"edge_double_time_events_rejected".to_string())
        );
    }

    #[test]
    fn sequence_policy_preserves_equally_supported_real_tempo_doubling() {
        let mut bpms = vec![60.0; 12];
        bpms.extend([120.0; 24]);
        let mut input = observations_from_bpms(&bpms);
        for beat in &mut input.beats {
            beat.downbeat_confidence = 0.0;
        }
        input.activity = input
            .beats
            .iter()
            .map(|beat| AudioActivityPoint {
                time_s: beat.time_s,
                rms: 1.0,
                relative_db: 0.0,
            })
            .collect();
        let expected_count = input.beats.len();

        let analysis = TempoMapEstimator::new(EstimatorOptions::sequence_phase_candidate())
            .unwrap()
            .estimate(&input)
            .unwrap();
        assert_eq!(analysis.beats.len(), expected_count);
        assert!((analysis.tempo_curve.last().unwrap().bpm - 120.0).abs() < 0.01);
        assert!(
            !analysis
                .warnings
                .contains(&"edge_double_time_events_rejected".to_string())
        );
    }

    #[test]
    fn stronger_alternating_accents_select_bar_level_downbeats() {
        let mut input = observations_from_bpms(&[120.0; 40]);
        for (index, beat) in input.beats.iter_mut().enumerate() {
            beat.downbeat_confidence = if index.is_multiple_of(2) { 0.9 } else { 0.1 };
        }
        input.activity = input
            .beats
            .iter()
            .enumerate()
            .map(|(index, beat)| AudioActivityPoint {
                time_s: beat.time_s,
                rms: if index.is_multiple_of(4) { 1.0 } else { 0.5 },
                relative_db: if index.is_multiple_of(4) { 0.0 } else { -6.0 },
            })
            .collect();

        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        let downbeat_indices = analysis
            .beats
            .iter()
            .enumerate()
            .filter(|(_, beat)| beat.downbeat)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(downbeat_indices, (0..=40).step_by(4).collect::<Vec<_>>());
        assert!(
            analysis
                .warnings
                .contains(&"bar_level_downbeats_selected".to_string())
        );
    }

    #[test]
    fn equally_salient_half_bar_accents_preserve_model_downbeats() {
        let mut input = observations_from_bpms(&[120.0; 40]);
        for (index, beat) in input.beats.iter_mut().enumerate() {
            beat.downbeat_confidence = if index.is_multiple_of(2) { 0.9 } else { 0.1 };
        }
        input.activity = input
            .beats
            .iter()
            .map(|beat| AudioActivityPoint {
                time_s: beat.time_s,
                rms: 1.0,
                relative_db: 0.0,
            })
            .collect();

        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        assert_eq!(
            analysis.beats.iter().filter(|beat| beat.downbeat).count(),
            21
        );
        assert!(
            !analysis
                .warnings
                .contains(&"bar_level_downbeats_selected".to_string())
        );
    }

    #[test]
    fn quarter_bar_downbeat_candidates_are_not_reinterpreted() {
        let mut input = observations_from_bpms(&[120.0; 40]);
        input.activity = input
            .beats
            .iter()
            .enumerate()
            .map(|(index, beat)| AudioActivityPoint {
                time_s: beat.time_s,
                rms: if index.is_multiple_of(8) { 1.0 } else { 0.5 },
                relative_db: if index.is_multiple_of(8) { 0.0 } else { -6.0 },
            })
            .collect();

        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        assert_eq!(
            analysis.beats.iter().filter(|beat| beat.downbeat).count(),
            11
        );
        assert!(
            !analysis
                .warnings
                .contains(&"bar_level_downbeats_selected".to_string())
        );
    }

    #[test]
    fn displaced_boundary_downbeat_is_realigned_to_selected_bar_grid() {
        let mut input = observations_from_bpms(&[120.0; 44]);
        for beat in &mut input.beats {
            beat.downbeat_confidence = 0.1;
        }
        for index in (0..=20).step_by(2) {
            input.beats[index].downbeat_confidence = 0.9;
        }
        input.beats[23].downbeat_confidence = 0.9;
        for index in (26..=44).step_by(2) {
            input.beats[index].downbeat_confidence = 0.9;
        }
        input.activity = input
            .beats
            .iter()
            .enumerate()
            .map(|(index, beat)| AudioActivityPoint {
                time_s: beat.time_s,
                rms: if index.is_multiple_of(4) { 1.0 } else { 0.25 },
                relative_db: if index.is_multiple_of(4) { 0.0 } else { -12.0 },
            })
            .collect();

        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        let downbeat_indices = analysis
            .beats
            .iter()
            .enumerate()
            .filter(|(_, beat)| beat.downbeat)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(downbeat_indices, (0..=44).step_by(4).collect::<Vec<_>>());
    }

    #[test]
    fn low_activity_span_rejects_hallucinated_beats_and_marks_discontinuity() {
        let mut input = observations_from_bpms(&[120.0; 16]);
        input.activity = (0..=82)
            .map(|index| {
                let time_s = f64::from(index) * 0.1;
                let silent = (2.0..=4.0).contains(&time_s);
                AudioActivityPoint {
                    time_s,
                    rms: if silent { 0.0001 } else { 1.0 },
                    relative_db: if silent { -80.0 } else { 0.0 },
                }
            })
            .collect();

        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        assert!(
            analysis
                .beats
                .iter()
                .all(|beat| beat.time_s < 2.0 || beat.time_s > 4.0)
        );
        assert!(analysis.change_points.iter().any(|change| {
            change.kind == ChangeKind::RhythmDiscontinuity && (change.time_s - 3.0).abs() < 0.2
        }));
    }

    #[test]
    fn short_smeared_transition_is_recovered_as_jump() {
        let mut bpms = vec![120.0; 20];
        bpms.extend([125.0, 130.4, 750.0, 150.0, 130.4, 157.9, 375.0, 272.7]);
        bpms.extend(vec![160.0; 20]);
        let input = observations_from_bpms(&bpms);
        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        assert!(
            analysis
                .change_points
                .iter()
                .any(|change| change.kind == ChangeKind::TempoJump),
            "segments: {:?}; changes: {:?}",
            analysis.tempo_segments,
            analysis.change_points
        );
        assert!(
            analysis
                .warnings
                .contains(&"short_transition_beat_grid_recovered".to_string()),
            "segments: {:?}; changes: {:?}; warnings: {:?}",
            analysis.tempo_segments,
            analysis.change_points,
            analysis.warnings
        );
        assert!(
            analysis
                .beats
                .windows(2)
                .all(|pair| pair[1].time_s - pair[0].time_s >= 0.3),
            "segments: {:?}; beats: {:?}",
            analysis.tempo_segments,
            analysis.beats
        );
    }

    #[test]
    fn missed_event_around_step_is_reconstructed() {
        let mut bpms = vec![120.0; 20];
        bpms.extend(vec![160.0; 20]);
        let mut input = observations_from_bpms(&bpms);
        input.beats.remove(22);

        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        assert!(
            analysis
                .warnings
                .contains(&"short_transition_beat_grid_recovered".to_string()),
            "segments: {:?}; changes: {:?}; warnings: {:?}",
            analysis.tempo_segments,
            analysis.change_points,
            analysis.warnings
        );
        assert_eq!(analysis.beats.len(), 41);
        assert!(
            analysis
                .beats
                .windows(2)
                .all(|pair| pair[1].time_s - pair[0].time_s >= 0.3)
        );
    }

    #[test]
    fn isolated_duplicate_without_tempo_change_is_not_suppressed() {
        let mut input = observations_from_bpms(&[120.0; 40]);
        input.beats.push(ObservedBeat {
            time_s: 10.1,
            confidence: 0.8,
            downbeat_confidence: 0.0,
        });
        input
            .beats
            .sort_by(|left, right| float_order(left.time_s, right.time_s));

        let analysis = TempoMapEstimator::default().estimate(&input).unwrap();
        assert!(
            !analysis
                .warnings
                .contains(&"short_transition_beat_grid_recovered".to_string())
        );
        assert_eq!(analysis.beats.len(), input.beats.len());
    }
}
