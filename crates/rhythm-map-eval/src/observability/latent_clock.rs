//! Frozen evaluation-only missing-step state model; see TRAINING-DECISION.md.

use anyhow::{Result, ensure};
use rhythm_map_core::RhythmObservations;

const STATES: usize = 8;
const MISSING_COST: f64 = 0.05;

#[derive(Debug, PartialEq)]
pub(super) struct Interval {
    pub start_s: f64,
    pub end_s: f64,
    pub advancement: usize,
    pub bpm: f64,
}

#[derive(Debug, PartialEq)]
pub(super) struct Clock {
    pub intervals: Vec<Interval>,
    pub objective: f64,
}

impl Clock {
    pub fn tempo_at(&self, time_s: f64) -> Option<f64> {
        let index = self.intervals.partition_point(|i| i.end_s <= time_s);
        self.intervals
            .get(index)
            .filter(|i| i.start_s <= time_s && time_s < i.end_s)
            .map(|i| i.bpm)
    }
}

#[allow(clippy::cast_precision_loss)]
fn period(duration: f64, state: usize) -> f64 {
    duration / (state + 1) as f64
}

#[allow(clippy::cast_precision_loss)]
fn missing_cost(state: usize) -> f64 {
    MISSING_COST * state as f64
}

/// Truth and masks are deliberately unavailable across this boundary.
pub(super) fn decode(observations: &RhythmObservations) -> Result<Clock> {
    ensure!(
        observations
            .beats
            .iter()
            .all(|b| b.time_s.is_finite() && b.time_s >= 0.0),
        "clock requires finite nonnegative observation times"
    );
    let durations = observations
        .beats
        .windows(2)
        .map(|b| b[1].time_s - b[0].time_s)
        .collect::<Vec<_>>();
    ensure!(
        durations.iter().all(|d| d.is_finite() && *d > 0.0),
        "clock requires strictly increasing finite observation times"
    );
    let mut predecessors = Vec::with_capacity(durations.len());
    let mut costs = [0.0; STATES];
    for (index, &duration) in durations.iter().enumerate() {
        let mut next = [f64::INFINITY; STATES];
        let mut previous_states = [0; STATES];
        for state in 0..STATES {
            if index == 0 {
                next[state] = missing_cost(state);
                continue;
            }
            for (previous, &previous_cost) in costs.iter().enumerate() {
                let ratio = period(duration, state) / period(durations[index - 1], previous);
                let cost = previous_cost + ratio.log2().powi(2) + missing_cost(state);
                if cost < next[state] {
                    next[state] = cost;
                    previous_states[state] = previous;
                }
            }
        }
        costs = next;
        predecessors.push(previous_states);
    }
    let mut state = (0..STATES)
        .min_by(|&a, &b| costs[a].total_cmp(&costs[b]).then(a.cmp(&b)))
        .unwrap_or(0);
    let objective = costs[state];
    let mut intervals = Vec::with_capacity(durations.len());
    for index in (0..durations.len()).rev() {
        intervals.push(Interval {
            start_s: observations.beats[index].time_s,
            end_s: observations.beats[index + 1].time_s,
            advancement: state + 1,
            bpm: 60.0 / period(durations[index], state),
        });
        state = predecessors[index][state];
    }
    intervals.reverse();
    Ok(Clock {
        intervals,
        objective,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::{Mask, fixture, masked};

    #[test]
    fn state_can_advance_across_missing_beats_without_emitting_them() {
        let truth = fixture(&[120.0; 3]);
        for mask in [Mask::Intact, Mask::EveryEighth, Mask::MiddleFour] {
            let observations = masked(&truth, mask, false);
            let clock = decode(&observations).unwrap();
            assert_eq!(clock.intervals.len(), observations.beats.len() - 1);
            assert!(clock.intervals.iter().all(|i| (i.bpm - 120.0).abs() < 1e-9));
            assert!(
                clock
                    .intervals
                    .iter()
                    .zip(observations.beats.windows(2))
                    .all(|(i, b)| i.start_s.to_bits() == b[0].time_s.to_bits()
                        && i.end_s.to_bits() == b[1].time_s.to_bits())
            );
        }
    }

    #[test]
    fn identical_inputs_remain_ambiguous_despite_a_smoothing_prior() {
        let sparse = masked(&fixture(&[120.0; 3]), Mask::MiddleAlternating, false);
        let slow = masked(&fixture(&[120.0, 60.0, 120.0]), Mask::Intact, false);
        assert_eq!(decode(&sparse).unwrap(), decode(&slow).unwrap());
    }

    #[test]
    fn no_extrapolation_beyond_observed_phase_anchors() {
        let observations = masked(&fixture(&[120.0]), Mask::Intact, true);
        let clock = decode(&observations).unwrap();
        assert_eq!(clock.tempo_at(-0.01), None);
        assert_eq!(clock.tempo_at(7.5), None);
        assert_eq!(clock.tempo_at(8.0), None);
        assert!(clock.tempo_at(0.25).is_some());
    }

    #[test]
    fn empty_or_single_event_has_no_tempo_and_invalid_times_fail_closed() {
        let mut observations = masked(&fixture(&[120.0]), Mask::Intact, true);
        observations.beats.truncate(1);
        assert!(decode(&observations).unwrap().intervals.is_empty());
        observations.beats[0].time_s = f64::NAN;
        assert!(decode(&observations).is_err());
        observations.beats.clear();
        assert!(decode(&observations).unwrap().intervals.is_empty());
        let mut duplicate = masked(&fixture(&[120.0]), Mask::Intact, true);
        duplicate.beats[1].time_s = duplicate.beats[0].time_s;
        assert!(decode(&duplicate).is_err());
    }

    #[test]
    fn dynamic_program_matches_exhaustive_short_path() {
        let mut observations = masked(&fixture(&[120.0]), Mask::Intact, true);
        observations.beats.truncate(4);
        observations.beats[2].time_s = 1.5;
        observations.beats[3].time_s = 2.0;
        let clock = decode(&observations).unwrap();
        let durations = [0.5, 1.0, 0.5];
        let mut best = f64::INFINITY;
        for a in 0..STATES {
            for b in 0..STATES {
                for c in 0..STATES {
                    let cost = missing_cost(a)
                        + missing_cost(b)
                        + missing_cost(c)
                        + (period(durations[1], b) / period(durations[0], a))
                            .log2()
                            .powi(2)
                        + (period(durations[2], c) / period(durations[1], b))
                            .log2()
                            .powi(2);
                    best = best.min(cost);
                }
            }
        }
        assert!((clock.objective - best).abs() < 1e-12);
    }
}
