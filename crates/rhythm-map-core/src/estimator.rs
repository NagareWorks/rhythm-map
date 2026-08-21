use std::cmp::Ordering;

use thiserror::Error;

use crate::{
    ANALYSIS_SCHEMA_VERSION, Analysis, BeatEvent, ChangeKind, ChangePoint, ObservedBeat,
    RhythmObservations, RhythmSection, TempoHypothesis, TempoPoint, TempoSegment, TempoSegmentKind,
};

/// Tunable policy for deterministic tempo-map inference.
#[derive(Debug, Clone)]
pub struct EstimatorOptions {
    /// Lowest accepted tempo candidate.
    pub min_bpm: f64,
    /// Highest accepted tempo candidate.
    pub max_bpm: f64,
    /// Lower edge of the preferred metrical band.
    pub preferred_min_bpm: f64,
    /// Upper edge of the preferred metrical band.
    pub preferred_max_bpm: f64,
    /// Odd median-filter window in inter-beat intervals.
    pub smoothing_window: usize,
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
    /// Required strong-to-weak accent ratio before half-bar downbeats are rejected.
    pub half_bar_downbeat_salience_ratio: f64,
    /// Longest model-smeared ramp still classified as a tempo jump.
    pub jump_transition_max_s: f64,
}

impl Default for EstimatorOptions {
    fn default() -> Self {
        Self {
            min_bpm: 40.0,
            max_bpm: 320.0,
            preferred_min_bpm: 70.0,
            preferred_max_bpm: 180.0,
            smoothing_window: 7,
            jump_ratio: 0.12,
            constant_ratio: 0.03,
            curve_tolerance_octaves: 0.04,
            discontinuity_factor: 3.5,
            silence_threshold_db: -40.0,
            minimum_silence_s: 0.8,
            half_time_min_bpm: 150.0,
            half_time_salience_ratio: 1.35,
            half_bar_downbeat_salience_ratio: 1.2,
            jump_transition_max_s: 4.0,
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
    pub fn new(options: EstimatorOptions) -> Result<Self, AnalysisError> {
        validate_options(&options)?;
        Ok(Self { options })
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
        } = build_tempo_estimate(&prepared, input.duration_s, &self.options);

        let global_bpm = median(smoothed.clone());
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

struct TempoEstimate {
    intervals: Vec<f64>,
    smoothed: Vec<f64>,
    tempo_curve: Vec<TempoPoint>,
    tempo_segments: Vec<TempoSegment>,
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
    let reference = preferred_reference(&raw_bpms, options);
    let normalized = raw_bpms
        .iter()
        .map(|&bpm| normalize_metrical_level(bpm, reference, options))
        .collect::<Vec<_>>();
    let smoothed = smooth_log_tempo(&normalized, options.smoothing_window);
    let tempo_curve = smoothed
        .iter()
        .enumerate()
        .map(|(index, &bpm)| {
            let pair = &input.beats[index..=index + 1];
            let observation_confidence = pair[0].confidence.min(pair[1].confidence);
            let deviation = (normalized[index] / bpm).log2().abs();
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
    }
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
    if input
        .beats
        .windows(2)
        .any(|pair| pair[1].time_s <= pair[0].time_s)
    {
        return Err(AnalysisError::UnsortedBeats);
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
    let rejected_silent_beats = input.beats.len() - observations.beats.len();
    let (observations, selected_half_time) = select_metrical_level(&observations, options);

    let mut warnings = Vec::new();
    if rejected_silent_beats > 0 {
        warnings.push("low_activity_beats_rejected".to_string());
    }
    if selected_half_time {
        warnings.push("metrical_level_selected_half_time".to_string());
    }

    PreparedObservations {
        observations,
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

fn select_metrical_level(
    input: &RhythmObservations,
    options: &EstimatorOptions,
) -> (RhythmObservations, bool) {
    if input.beats.len() < 8 || input.activity.is_empty() {
        return (input.clone(), false);
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
        return (input.clone(), false);
    }

    let phase_scores = [0_usize, 1].map(|phase| mean_phase_salience(input, phase));
    let selected_phase = usize::from(phase_scores[1] > phase_scores[0]);
    let retained = phase_scores[selected_phase];
    let discarded = phase_scores[1 - selected_phase].max(1e-9);
    if retained / discarded < options.half_time_salience_ratio {
        return (input.clone(), false);
    }

    let mut selected = input.clone();
    selected.beats = input
        .beats
        .iter()
        .skip(selected_phase)
        .step_by(2)
        .cloned()
        .collect();
    (selected, true)
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

fn preferred_reference(raw_bpms: &[f64], options: &EstimatorOptions) -> f64 {
    let folded = raw_bpms
        .iter()
        .map(|&value| {
            let mut bpm = value;
            while bpm < options.preferred_min_bpm {
                bpm *= 2.0;
            }
            while bpm > options.preferred_max_bpm {
                bpm *= 0.5;
            }
            bpm
        })
        .collect::<Vec<_>>();
    median(folded)
}

fn normalize_metrical_level(raw: f64, reference: f64, options: &EstimatorOptions) -> f64 {
    (-3..=3)
        .map(|level| (raw * 2.0_f64.powi(level), level))
        .filter(|(bpm, _)| *bpm >= options.min_bpm && *bpm <= options.max_bpm)
        .min_by(|(left, left_level), (right, right_level)| {
            let left_score = (left / reference).log2().abs() + 0.02 * f64::from(left_level.abs());
            let right_score =
                (right / reference).log2().abs() + 0.02 * f64::from(right_level.abs());
            float_order(left_score, right_score)
        })
        .map_or(raw.clamp(options.min_bpm, options.max_bpm), |(bpm, _)| bpm)
}

fn smooth_log_tempo(values: &[f64], window: usize) -> Vec<f64> {
    let radius = window / 2;
    let median_filtered = (0..values.len())
        .map(|index| {
            let start = index.saturating_sub(radius);
            let end = (index + radius + 1).min(values.len());
            median(values[start..end].to_vec())
        })
        .collect::<Vec<_>>();

    (0..median_filtered.len())
        .map(|index| {
            let start = index.saturating_sub(1);
            let end = (index + 2).min(median_filtered.len());
            let mean_log = median_filtered[start..end]
                .iter()
                .map(|value| value.log2())
                .sum::<f64>()
                / usize_to_f64(end - start);
            2.0_f64.powf(mean_log)
        })
        .collect()
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
        while index < segments.len() && segments[index].kind == TempoSegmentKind::Ramp {
            index += 1;
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
    let start_s = transition.start_s(segments);
    let end_s = transition.end_s(segments);
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
    let left_grid = fit_regular_grid(&left_beats)?;
    let right_grid = fit_regular_grid(&right_beats)?;
    if left_grid.maximum_error > left_grid.period * 0.08
        || right_grid.maximum_error > right_grid.period * 0.08
        || (right_grid.period / left_grid.period).ln().abs() < options.jump_ratio.ln_1p()
    {
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
    use crate::{AudioActivityPoint, ModelInfo, ObservedBeat};

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
            activity: Vec::new(),
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
