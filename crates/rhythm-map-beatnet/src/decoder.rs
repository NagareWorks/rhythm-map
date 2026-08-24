//! Evidence-constrained tempo and meter sequence decoding.
//!
//! The neural model emits dense beat/downbeat/non-beat probabilities. This
//! decoder searches only the graph of real pulse maxima: a state carries the
//! most recent inter-beat interval and bar phase, and an edge pays for tempo or
//! metrical-level movement. A weak maximum can therefore be skipped instead
//! of being accepted merely because a virtual frame grid passes near it.

const FRAME_RATE_HZ: f64 = 50.0;
const MINIMUM_BPM: f64 = 40.0;
const MAXIMUM_BPM: f64 = 320.0;
const BEATS_PER_BAR: [usize; 3] = [2, 3, 4];
const BEAT_STATE_BIAS: f64 = 2.0;
const TEMPO_CHANGE_PENALTY: f64 = 50.0;
const METRICAL_LEVEL_CHANGE_PENALTY: f64 = 8.0;
const METER_EVIDENCE_WEIGHT: f64 = 0.25;
const ACTIVE_PULSE_THRESHOLD: f32 = 0.05;
const GRID_PRIOR_BONUS: f64 = 0.5;
const GRID_TEMPO_CHANGE_PENALTY: f64 = 100.0;
const MAXIMUM_GRID_CORRECTION_FRAMES: usize = 3;
const MINIMUM_CONTINUITY_RETENTION: f64 = 0.7;
const EDGE_INTERVAL_TOLERANCE: f64 = 0.3;

#[derive(Debug)]
struct DecodedPath {
    candidate_indices: Vec<usize>,
    score: f64,
}

pub(super) fn decode_candidate_path(
    beat_probabilities: &[f32],
    downbeat_probabilities: &[f32],
    nonbeat_probabilities: &[f32],
    candidate_frames: &[usize],
) -> Vec<usize> {
    if beat_probabilities.is_empty()
        || beat_probabilities.len() != downbeat_probabilities.len()
        || beat_probabilities.len() != nonbeat_probabilities.len()
        || candidate_frames.is_empty()
    {
        return Vec::new();
    }

    let Some((first_active, last_active)) =
        active_candidate_range(beat_probabilities, downbeat_probabilities, candidate_frames)
    else {
        return Vec::new();
    };
    let pulse_probabilities = beat_probabilities
        .iter()
        .zip(downbeat_probabilities)
        .map(|(beat, downbeat)| beat + downbeat)
        .collect::<Vec<_>>();
    let grid_prior = decode_grid_prior(&pulse_probabilities, candidate_frames);
    let best = BEATS_PER_BAR
        .into_iter()
        .filter_map(|meter| {
            viterbi_candidate_path(
                beat_probabilities,
                downbeat_probabilities,
                nonbeat_probabilities,
                candidate_frames,
                first_active,
                last_active,
                meter,
                &grid_prior,
            )
        })
        .max_by(|left, right| left.score.total_cmp(&right.score));
    let mut selected = best.map_or_else(Vec::new, |path| {
        path.candidate_indices
            .into_iter()
            .map(|index| candidate_frames[index])
            .collect()
    });
    if graph_violates_grid_safety(&selected, &grid_prior) {
        selected.clone_from(&grid_prior);
    }
    complete_grid_supported_edges(&mut selected, &grid_prior);
    selected
}

#[allow(clippy::too_many_arguments)]
fn viterbi_candidate_path(
    beat_probabilities: &[f32],
    downbeat_probabilities: &[f32],
    nonbeat_probabilities: &[f32],
    candidate_frames: &[usize],
    first_active: usize,
    last_active: usize,
    beats_per_bar: usize,
    grid_prior: &[usize],
) -> Option<DecodedPath> {
    let minimum_period = minimum_period_frames();
    let maximum_period = maximum_period_frames();
    let period_count = maximum_period - minimum_period + 1;
    let state_count = candidate_frames.len() * period_count * beats_per_bar;
    let mut scores = vec![f64::NEG_INFINITY; state_count];
    let mut back_candidates = vec![u32::MAX; state_count];
    let mut back_periods = vec![u16::MAX; state_count];

    initialize_candidate_paths(
        beat_probabilities,
        downbeat_probabilities,
        nonbeat_probabilities,
        candidate_frames,
        first_active,
        minimum_period,
        maximum_period,
        beats_per_bar,
        period_count,
        grid_prior,
        &mut scores,
        &mut back_candidates,
        &mut back_periods,
    );

    for current in first_active..candidate_frames.len() {
        for period_index in 0..period_count {
            let previous_period = period_index + minimum_period;
            for beat_in_bar in 0..beats_per_bar {
                let state = state_index(
                    current,
                    period_index,
                    beat_in_bar,
                    period_count,
                    beats_per_bar,
                );
                let score = scores[state];
                if !score.is_finite() {
                    continue;
                }
                let next_beat_in_bar = (beat_in_bar + 1) % beats_per_bar;
                for next in current + 1..candidate_frames.len() {
                    let next_period = candidate_frames[next] - candidate_frames[current];
                    if next_period > maximum_period {
                        break;
                    }
                    if next_period < minimum_period {
                        continue;
                    }
                    let next_period_index = next_period - minimum_period;
                    let next_score = score
                        + tempo_transition_score(previous_period, next_period)
                        + candidate_score(
                            beat_probabilities,
                            downbeat_probabilities,
                            nonbeat_probabilities,
                            candidate_frames[next],
                            next_beat_in_bar,
                            grid_prior,
                        );
                    let next_state = state_index(
                        next,
                        next_period_index,
                        next_beat_in_bar,
                        period_count,
                        beats_per_bar,
                    );
                    if next_score > scores[next_state] {
                        scores[next_state] = next_score;
                        back_candidates[next_state] =
                            u32::try_from(current).expect("candidate index fits u32");
                        back_periods[next_state] =
                            u16::try_from(period_index).expect("BeatNet period index fits u16");
                    }
                }
            }
        }
    }

    backtrack_candidate_path(
        candidate_frames,
        first_active,
        last_active,
        maximum_period,
        period_count,
        beats_per_bar,
        &scores,
        &back_candidates,
        &back_periods,
    )
}

#[allow(clippy::too_many_arguments)]
fn initialize_candidate_paths(
    beat_probabilities: &[f32],
    downbeat_probabilities: &[f32],
    nonbeat_probabilities: &[f32],
    candidate_frames: &[usize],
    first_active: usize,
    minimum_period: usize,
    maximum_period: usize,
    beats_per_bar: usize,
    period_count: usize,
    grid_prior: &[usize],
    scores: &mut [f64],
    back_candidates: &mut [u32],
    back_periods: &mut [u16],
) {
    let last_start_frame = candidate_frames[first_active].saturating_add(maximum_period);
    for first in first_active..candidate_frames.len() {
        if candidate_frames[first] > last_start_frame {
            break;
        }
        for second in first + 1..candidate_frames.len() {
            let period = candidate_frames[second] - candidate_frames[first];
            if period > maximum_period {
                break;
            }
            if period < minimum_period {
                continue;
            }
            let period_index = period - minimum_period;
            for second_beat_in_bar in 0..beats_per_bar {
                let first_beat_in_bar = (second_beat_in_bar + beats_per_bar - 1) % beats_per_bar;
                let score = candidate_score(
                    beat_probabilities,
                    downbeat_probabilities,
                    nonbeat_probabilities,
                    candidate_frames[first],
                    first_beat_in_bar,
                    grid_prior,
                ) + candidate_score(
                    beat_probabilities,
                    downbeat_probabilities,
                    nonbeat_probabilities,
                    candidate_frames[second],
                    second_beat_in_bar,
                    grid_prior,
                );
                let state = state_index(
                    second,
                    period_index,
                    second_beat_in_bar,
                    period_count,
                    beats_per_bar,
                );
                if score > scores[state] {
                    scores[state] = score;
                    back_candidates[state] =
                        u32::try_from(first).expect("candidate index fits u32");
                    back_periods[state] = u16::MAX;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn backtrack_candidate_path(
    candidate_frames: &[usize],
    first_active: usize,
    last_active: usize,
    maximum_period: usize,
    period_count: usize,
    beats_per_bar: usize,
    scores: &[f64],
    back_candidates: &[u32],
    back_periods: &[u16],
) -> Option<DecodedPath> {
    let first_terminal_frame = candidate_frames[last_active].saturating_sub(maximum_period);
    let mut terminal = None;
    for (candidate, &frame) in candidate_frames.iter().enumerate().skip(first_active) {
        if frame < first_terminal_frame {
            continue;
        }
        for period_index in 0..period_count {
            for beat_in_bar in 0..beats_per_bar {
                let state = state_index(
                    candidate,
                    period_index,
                    beat_in_bar,
                    period_count,
                    beats_per_bar,
                );
                let score = scores[state];
                if terminal.is_none_or(|(_, best_score)| score > best_score) {
                    terminal = Some((state, score));
                }
            }
        }
    }
    let (mut state, score) = terminal.filter(|(_, score)| score.is_finite())?;
    let (mut candidate, _, mut beat_in_bar) =
        decode_state_index(state, period_count, beats_per_bar);
    let mut path = vec![candidate];
    loop {
        let previous_candidate = usize::try_from(back_candidates[state]).ok()?;
        path.push(previous_candidate);
        let previous_period = back_periods[state];
        if previous_period == u16::MAX {
            break;
        }
        candidate = previous_candidate;
        let period_index = usize::from(previous_period);
        beat_in_bar = (beat_in_bar + beats_per_bar - 1) % beats_per_bar;
        state = state_index(
            candidate,
            period_index,
            beat_in_bar,
            period_count,
            beats_per_bar,
        );
    }
    path.reverse();
    Some(DecodedPath {
        candidate_indices: path,
        score,
    })
}

fn decode_grid_prior(probabilities: &[f32], candidates: &[usize]) -> Vec<usize> {
    let mut snapped = Vec::new();
    let mut last = None;
    for frame in viterbi_grid_path(probabilities) {
        let candidate = candidates
            .iter()
            .copied()
            .filter(|candidate| last.is_none_or(|last| *candidate > last))
            .filter(|candidate| candidate.abs_diff(frame) <= MAXIMUM_GRID_CORRECTION_FRAMES)
            .min_by_key(|candidate| candidate.abs_diff(frame));
        if let Some(candidate) = candidate {
            snapped.push(candidate);
            last = Some(candidate);
        }
    }
    snapped
}

fn viterbi_grid_path(probabilities: &[f32]) -> Vec<usize> {
    let minimum_period = minimum_period_frames();
    let maximum_period = maximum_period_frames();
    let periods = (minimum_period..=maximum_period).collect::<Vec<_>>();
    let mut offsets = Vec::with_capacity(periods.len());
    let mut total_states = 0;
    for &period in &periods {
        offsets.push(total_states);
        total_states += period;
    }

    let (beat_emission, nonbeat_emission) = grid_emissions(probabilities[0]);
    let mut scores = vec![f64::NEG_INFINITY; total_states];
    for (&period, &offset) in periods.iter().zip(&offsets) {
        scores[offset] = beat_emission;
        scores[offset + 1..offset + period].fill(nonbeat_emission);
    }
    let mut back_periods = vec![u16::MAX; probabilities.len() * periods.len()];

    for (frame, &probability) in probabilities.iter().enumerate().skip(1) {
        let (beat_emission, nonbeat_emission) = grid_emissions(probability);
        let mut next = vec![f64::NEG_INFINITY; total_states];
        for (period_index, (&period, &offset)) in periods.iter().zip(&offsets).enumerate() {
            for phase in 1..period {
                next[offset + phase] = scores[offset + phase - 1] + nonbeat_emission;
            }
            let mut best_transition = f64::NEG_INFINITY;
            let mut best_period_index = 0;
            for (source_index, (&source_period, &source_offset)) in
                periods.iter().zip(&offsets).enumerate()
            {
                let log_ratio = (usize_to_f64(period) / usize_to_f64(source_period)).ln();
                let penalty = GRID_TEMPO_CHANGE_PENALTY * log_ratio * log_ratio;
                let score = scores[source_offset + source_period - 1] - penalty;
                if score > best_transition {
                    best_transition = score;
                    best_period_index = source_index;
                }
            }
            next[offset] = best_transition + beat_emission;
            back_periods[frame * periods.len() + period_index] =
                u16::try_from(best_period_index).expect("BeatNet period index fits u16");
        }
        scores = next;
    }

    let mut period_index = 0;
    let mut phase = 0;
    let mut terminal_score = f64::NEG_INFINITY;
    for (candidate_period, (&period, &offset)) in periods.iter().zip(&offsets).enumerate() {
        for candidate_phase in 0..period {
            let score = scores[offset + candidate_phase];
            if score > terminal_score {
                period_index = candidate_period;
                phase = candidate_phase;
                terminal_score = score;
            }
        }
    }
    let mut beats = Vec::new();
    for frame in (0..probabilities.len()).rev() {
        if phase == 0 {
            beats.push(frame);
        }
        if frame == 0 {
            break;
        }
        if phase == 0 {
            period_index = usize::from(back_periods[frame * periods.len() + period_index]);
            phase = periods[period_index] - 1;
        } else {
            phase -= 1;
        }
    }
    beats.reverse();
    beats
}

fn grid_emissions(probability: f32) -> (f64, f64) {
    let probability = f64::from(probability).clamp(1e-7, 1.0 - 1e-7);
    (probability.ln() + BEAT_STATE_BIAS, (-probability).ln_1p())
}

fn sequence_continuity(frames: &[usize]) -> f64 {
    if frames.len() < 3 {
        return 0.0;
    }
    let intervals = frames
        .windows(2)
        .map(|pair| usize_to_f64(pair[1] - pair[0]))
        .collect::<Vec<_>>();
    let median = median(intervals.clone());
    let mean_log_error = intervals
        .iter()
        .map(|interval| (interval / median).ln().abs())
        .sum::<f64>()
        / usize_to_f64(intervals.len());
    (-4.0 * mean_log_error).exp()
}

fn graph_violates_grid_safety(selected: &[usize], grid_prior: &[usize]) -> bool {
    let selected_continuity = sequence_continuity(selected);
    let grid_continuity = sequence_continuity(grid_prior);
    selected_continuity < MINIMUM_CONTINUITY_RETENTION * grid_continuity
        || (selected.len() >= grid_prior.len().saturating_add(2)
            && selected_continuity < grid_continuity)
}

fn complete_grid_supported_edges(selected: &mut Vec<usize>, grid_prior: &[usize]) {
    if selected.len() < 3 || grid_prior.is_empty() {
        return;
    }
    let intervals = selected
        .windows(2)
        .map(|pair| usize_to_f64(pair[1] - pair[0]))
        .collect::<Vec<_>>();
    let median_interval = median(intervals);
    let minimum = median_interval * (1.0 - EDGE_INTERVAL_TOLERANCE);
    let maximum = median_interval * (1.0 + EDGE_INTERVAL_TOLERANCE);
    if let Some(&previous) = grid_prior.iter().rev().find(|&&frame| frame < selected[0]) {
        let interval = usize_to_f64(selected[0] - previous);
        if (minimum..=maximum).contains(&interval) {
            selected.insert(0, previous);
        }
    }
    if let Some(&next) = grid_prior
        .iter()
        .find(|&&frame| frame > selected[selected.len() - 1])
    {
        let interval = usize_to_f64(next - selected[selected.len() - 1]);
        if (minimum..=maximum).contains(&interval) {
            selected.push(next);
        }
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f64::midpoint(values[middle - 1], values[middle])
    } else {
        values[middle]
    }
}

fn active_candidate_range(
    beat_probabilities: &[f32],
    downbeat_probabilities: &[f32],
    candidate_frames: &[usize],
) -> Option<(usize, usize)> {
    let active = candidate_frames
        .iter()
        .enumerate()
        .filter(|&(_, &frame)| {
            beat_probabilities[frame] + downbeat_probabilities[frame] >= ACTIVE_PULSE_THRESHOLD
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    Some((*active.first()?, *active.last()?))
}

fn candidate_score(
    beat_probabilities: &[f32],
    downbeat_probabilities: &[f32],
    nonbeat_probabilities: &[f32],
    frame: usize,
    beat_in_bar: usize,
    grid_prior: &[usize],
) -> f64 {
    let pulse =
        f64::from(beat_probabilities[frame] + downbeat_probabilities[frame]).clamp(1e-12, 1.0);
    let nonbeat = f64::from(nonbeat_probabilities[frame]).clamp(1e-12, 1.0);
    let metrical_class = if beat_in_bar == 0 {
        downbeat_probabilities[frame]
    } else {
        beat_probabilities[frame]
    };
    let metrical_fraction = (f64::from(metrical_class) / pulse).clamp(1e-12, 1.0);
    let grid_bonus = if grid_prior.binary_search(&frame).is_ok() {
        GRID_PRIOR_BONUS
    } else {
        0.0
    };
    (pulse / nonbeat).ln()
        + BEAT_STATE_BIAS
        + METER_EVIDENCE_WEIGHT * metrical_fraction.ln()
        + grid_bonus
}

fn tempo_transition_score(previous_period: usize, next_period: usize) -> f64 {
    let log_ratio = (usize_to_f64(next_period) / usize_to_f64(previous_period)).ln();
    let ordinary = TEMPO_CHANGE_PENALTY * log_ratio * log_ratio;
    let octave_distance = log_ratio.abs() - std::f64::consts::LN_2;
    let metrical =
        METRICAL_LEVEL_CHANGE_PENALTY + TEMPO_CHANGE_PENALTY * octave_distance * octave_distance;
    -ordinary.min(metrical)
}

fn state_index(
    candidate: usize,
    period_index: usize,
    beat_in_bar: usize,
    period_count: usize,
    beats_per_bar: usize,
) -> usize {
    (candidate * period_count + period_index) * beats_per_bar + beat_in_bar
}

fn decode_state_index(
    state: usize,
    period_count: usize,
    beats_per_bar: usize,
) -> (usize, usize, usize) {
    let beat_in_bar = state % beats_per_bar;
    let interval_state = state / beats_per_bar;
    (
        interval_state / period_count,
        interval_state % period_count,
        beat_in_bar,
    )
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn minimum_period_frames() -> usize {
    (60.0 * FRAME_RATE_HZ / MAXIMUM_BPM).ceil() as usize
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn maximum_period_frames() -> usize {
    (60.0 * FRAME_RATE_HZ / MINIMUM_BPM).floor() as usize
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn activations_with_peaks(
        frame_count: usize,
        peaks: &[(usize, f32, bool)],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<usize>) {
        let mut beat = vec![0.001; frame_count];
        let mut downbeat = vec![0.001; frame_count];
        let mut nonbeat = vec![0.998; frame_count];
        let mut candidates = Vec::new();
        for &(frame, pulse, is_downbeat) in peaks {
            if is_downbeat {
                downbeat[frame] = pulse;
            } else {
                beat[frame] = pulse;
            }
            nonbeat[frame] = 1.0 - pulse - 0.001;
            candidates.push(frame);
        }
        (beat, downbeat, nonbeat, candidates)
    }

    #[test]
    fn candidate_path_does_not_force_a_beat_at_frame_zero() {
        let peaks = [
            (10, 0.94, true),
            (30, 0.94, false),
            (50, 0.94, false),
            (70, 0.94, false),
            (90, 0.94, true),
        ];
        let (beat, downbeat, nonbeat, candidates) = activations_with_peaks(110, &peaks);
        assert_eq!(
            decode_candidate_path(&beat, &downbeat, &nonbeat, &candidates),
            [10, 30, 50, 70, 90]
        );
    }

    #[test]
    fn weak_subdivision_is_not_kept_to_complete_a_grid() {
        let peaks = [
            (10, 0.94, true),
            (20, 0.04, false),
            (30, 0.94, false),
            (40, 0.04, false),
            (50, 0.94, false),
            (60, 0.04, false),
            (70, 0.94, false),
            (80, 0.04, false),
            (90, 0.94, true),
        ];
        let (beat, downbeat, nonbeat, candidates) = activations_with_peaks(110, &peaks);
        assert_eq!(
            decode_candidate_path(&beat, &downbeat, &nonbeat, &candidates),
            [10, 30, 50, 70, 90]
        );
    }

    #[test]
    fn sustained_strong_subdivision_can_change_metrical_level() {
        let peaks = [
            (10, 0.94, true),
            (30, 0.94, false),
            (50, 0.94, false),
            (70, 0.94, false),
            (80, 0.94, true),
            (90, 0.94, false),
            (100, 0.94, false),
            (110, 0.94, false),
            (120, 0.94, true),
            (130, 0.94, false),
            (140, 0.94, false),
        ];
        let (beat, downbeat, nonbeat, candidates) = activations_with_peaks(150, &peaks);
        assert_eq!(
            decode_candidate_path(&beat, &downbeat, &nonbeat, &candidates),
            candidates
        );
    }

    #[test]
    fn continuity_gate_detects_fragmented_metrical_switches() {
        let coherent = [10, 30, 50, 70, 90, 110];
        let fragmented = [10, 20, 50, 60, 90, 100];
        assert!(
            sequence_continuity(&fragmented)
                < MINIMUM_CONTINUITY_RETENTION * sequence_continuity(&coherent)
        );
    }

    #[test]
    fn multiple_extra_events_must_not_reduce_continuity() {
        let grid = [10, 30, 50, 70, 90, 110];
        let graph = [10, 20, 30, 50, 70, 90, 100, 110];
        assert!(graph_violates_grid_safety(&graph, &grid));
    }

    #[test]
    fn real_grid_candidate_can_complete_one_track_edge() {
        let mut selected = vec![10, 30, 50, 70];
        complete_grid_supported_edges(&mut selected, &[10, 30, 50, 70, 92]);
        assert_eq!(selected, [10, 30, 50, 70, 92]);
    }
}
