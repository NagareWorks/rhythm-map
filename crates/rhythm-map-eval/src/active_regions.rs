//! Evaluation-owned, unranked active-interval proposals. Not an estimator policy.
//!
//! The v1 recurrence and weights are frozen to the historical full-file path.
//! Only its connectivity domain and storage layout differ. Do not import this
//! module into a product surface or silently stitch these proposals into Analysis.

use anyhow::{Result, ensure};
use rhythm_map_core::{BeatCandidate, RhythmObservations};
use serde::{Deserialize, Serialize};
use std::mem::size_of;

const MIN_INTERVAL: f64 = 60.0 / 320.0;
const MAX_INTERVAL: f64 = 60.0 / 40.0;
const MIN_EVENTS: usize = 8;

/// Unranked, calibration-only output; never a replacement for primary analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActiveRegionCandidates {
    /// Immutable generator identity, including fixed weights and boundaries.
    pub generator: String,
    /// Inclusive silence boundaries from the frozen default activity rule.
    pub silence_regions: Vec<[f64; 2]>,
    /// Unsupported spans; boundary events may be supported. Preserve baseline
    /// beats here. Order follows discovery (tail before internal gaps).
    pub unknown_gaps: Vec<[f64; 2]>,
    /// Independent components, including explicit fallbacks.
    pub proposals: Vec<ActiveRegionCandidate>,
    /// Work and storage accounting for the sparse path search, excluding inputs.
    pub work: ActiveRegionWork,
}

/// One connected candidate domain; bounds need not be shared beat anchors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActiveRegionCandidate {
    /// Inclusive candidate-domain start.
    pub start_s: f64,
    /// Inclusive candidate-domain end.
    pub end_s: f64,
    /// Real backend events after silence filtering in this domain.
    pub candidate_count: usize,
    /// Availability, not a recommendation to adopt.
    pub status: ActiveRegionStatus,
    /// Primary timestamps in the domain, retained even on fallback.
    pub original_times_s: Vec<f64>,
    /// A supported, strictly increasing path, or absent on fallback.
    pub proposal_times_s: Option<Vec<f64>>,
    /// Shared-anchor geometry following `MetricalAmbiguityRegion` semantics, but
    /// without a fabricated relative score for an unranked local proposal.
    pub disagreements: Vec<ActiveRegionDisagreement>,
}

/// The absence of a valid proposal never authorizes deleting the baseline.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActiveRegionStatus {
    /// At least eight events connect the domain under the fixed interval bounds.
    Proposal,
    /// Too few events, insufficient harmonic coverage, or no valid full path.
    FallbackNoValidPath,
}

/// Geometry/count subset of the existing metrical ambiguity contract.
/// Anchors indicate agreement, not annotation correctness or confidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActiveRegionDisagreement {
    /// Domain edge or common beat timestamp.
    pub start_s: f64,
    /// Domain edge or common beat timestamp.
    pub end_s: f64,
    /// Both sequences contain the left boundary beat.
    pub left_anchored: bool,
    /// Both sequences contain the right boundary beat.
    pub right_anchored: bool,
    /// Primary-only beats, excluding shared boundary anchors.
    pub primary_only_beat_count: usize,
    /// Proposal-only beats, excluding shared boundary anchors.
    pub alternative_only_beat_count: usize,
}

/// Deterministic algorithmic counts, not process RSS or a runtime guarantee.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActiveRegionWork {
    /// Total allocated ordered-pair states across all components.
    pub pair_states: usize,
    /// Scored transitions from reachable pair states.
    pub transitions: usize,
    /// Maximum simultaneous requested DP vector capacity in bytes, including
    /// row headers and scores/backpointers, excluding allocator bookkeeping,
    /// rewards, timestamps, input observations and output paths.
    pub peak_dp_capacity_bytes: usize,
}

/// Generate frozen-v1 candidates without reading annotations or changing output.
///
/// `selected_times_s` must come from the caller's baseline analysis of the same
/// observations. No primary analysis, neural inference, ranking, or adoption is
/// performed here. Unknown spans and unsuccessful components retain that baseline.
///
/// # Errors
/// Rejects invalid duration, unsorted/out-of-range times, invalid confidence,
/// harmonic or activity evidence. Unused dense activations/onsets are not read.
pub fn generate_active_region_candidates(
    input: &RhythmObservations,
    selected_times_s: &[f64],
) -> Result<ActiveRegionCandidates> {
    validate(input, selected_times_s)?;
    let silence = silence_regions(input);
    let candidates = input
        .beat_candidates
        .iter()
        .filter(|e| !silence.iter().any(|r| r[0] <= e.time_s && e.time_s <= r[1]))
        .collect::<Vec<_>>();
    // Deliberately a whole-filtered-case gate, matching the frozen experiment.
    let harmonic_gate = input.harmonic_changes.len() >= candidates.len() / 2;
    let mut active = Vec::new();
    let mut cursor = 0.0;
    for &[a, b] in &silence {
        if a > cursor {
            active.push([cursor, a]);
        }
        cursor = b;
    }
    if cursor < input.duration_s {
        active.push([cursor, input.duration_s]);
    }
    let mut result = ActiveRegionCandidates {
        generator: "active-interval-path-v1".into(),
        silence_regions: silence,
        unknown_gaps: Vec::new(),
        proposals: Vec::new(),
        work: ActiveRegionWork::default(),
    };
    for [mut a, b] in active {
        let first = candidates.partition_point(|e| e.time_s < a);
        let last = candidates.partition_point(|e| e.time_s <= b);
        let events = &candidates[first..last];
        if events.is_empty() {
            result.unknown_gaps.push([a, b]);
            continue;
        }
        if events[0].time_s - a > MAX_INTERVAL {
            result.unknown_gaps.push([a, events[0].time_s]);
            a = events[0].time_s;
        }
        let last_time = events[events.len() - 1].time_s;
        let tail = if b - last_time > MAX_INTERVAL {
            result.unknown_gaps.push([last_time, b]);
            last_time
        } else {
            b
        };
        let mut first = 0;
        for i in 1..events.len() {
            if events[i].time_s - events[i - 1].time_s > MAX_INTERVAL {
                result.proposals.push(component(
                    input,
                    selected_times_s,
                    &events[first..i],
                    [a, events[i - 1].time_s],
                    harmonic_gate,
                    &mut result.work,
                ));
                result
                    .unknown_gaps
                    .push([events[i - 1].time_s, events[i].time_s]);
                a = events[i].time_s;
                first = i;
            }
        }
        result.proposals.push(component(
            input,
            selected_times_s,
            &events[first..],
            [a, tail],
            harmonic_gate,
            &mut result.work,
        ));
    }
    Ok(result)
}

fn validate(input: &RhythmObservations, selected: &[f64]) -> Result<()> {
    ensure!(
        input.duration_s.is_finite() && input.duration_s >= 0.0,
        "invalid duration"
    );
    let ordered = |times: Vec<f64>| {
        times
            .iter()
            .all(|t| t.is_finite() && *t >= 0.0 && *t <= input.duration_s)
            && times.windows(2).all(|p| p[0] < p[1])
    };
    ensure!(ordered(selected.to_vec()), "invalid selected timestamps");
    ensure!(
        ordered(input.beat_candidates.iter().map(|e| e.time_s).collect()),
        "invalid candidate timestamps"
    );
    ensure!(
        ordered(input.activity.iter().map(|e| e.time_s).collect()),
        "invalid activity timestamps"
    );
    ensure!(
        ordered(input.harmonic_changes.iter().map(|e| e.time_s).collect()),
        "invalid harmonic timestamps"
    );
    ensure!(
        input
            .beat_candidates
            .iter()
            .all(|e| (0.0..=1.0).contains(&e.confidence)
                && (0.0..=1.0).contains(&e.downbeat_confidence)),
        "invalid candidate confidence"
    );
    ensure!(
        input
            .harmonic_changes
            .iter()
            .all(|e| (0.0..=1.0).contains(&e.strength)),
        "invalid harmonic strength"
    );
    ensure!(
        input.activity.iter().all(|e| e.relative_db.is_finite()
            && e.relative_db <= 0.0
            && e.rms.is_finite()
            && e.rms >= 0.0),
        "invalid activity value"
    );
    Ok(())
}

#[allow(clippy::manual_midpoint)] // Preserve the frozen default median arithmetic.
fn silence_regions(input: &RhythmObservations) -> Vec<[f64; 2]> {
    let mut hops = input
        .activity
        .windows(2)
        .map(|p| p[1].time_s - p[0].time_s)
        .collect::<Vec<_>>();
    hops.sort_by(f64::total_cmp);
    let hop = if hops.is_empty() {
        input.duration_s.max(0.001)
    } else if hops.len() % 2 == 0 {
        (hops[hops.len() / 2 - 1] + hops[hops.len() / 2]) * 0.5
    } else {
        hops[hops.len() / 2]
    };
    let mut result = Vec::new();
    let mut start = 0;
    while start < input.activity.len() {
        if input.activity[start].relative_db > -40.0 {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < input.activity.len() && input.activity[end].relative_db <= -40.0 {
            end += 1;
        }
        let a = (input.activity[start].time_s - hop * 0.5).max(0.0);
        let b = (input.activity[end - 1].time_s + hop * 0.5).min(input.duration_s);
        if b - a >= 0.8 {
            result.push([a, b]);
        }
        start = end;
    }
    result
}

fn reward(input: &RhythmObservations, event: &BeatCandidate) -> f64 {
    let points = &input.harmonic_changes;
    let right = points.partition_point(|h| h.time_s < event.time_s);
    // Sorted evidence permits adjacent lookup; equal-distance ties choose left.
    let nearest = match (right.checked_sub(1).map(|i| &points[i]), points.get(right)) {
        (Some(left), Some(right)) => Some(
            if event.time_s - left.time_s <= right.time_s - event.time_s {
                left
            } else {
                right
            },
        ),
        (left, right) => left.or(right),
    };
    let strength = nearest
        .filter(|h| (h.time_s - event.time_s).abs() <= 0.02)
        .map_or(0.0, |h| h.strength);
    event.confidence + 0.1 * event.downbeat_confidence + 5.0 * strength - 0.95
}

fn component(
    input: &RhythmObservations,
    selected: &[f64],
    events: &[&BeatCandidate],
    [a, b]: [f64; 2],
    harmonic_gate: bool,
    work: &mut ActiveRegionWork,
) -> ActiveRegionCandidate {
    let original = selected
        .iter()
        .copied()
        .filter(|t| a <= *t && *t <= b)
        .collect::<Vec<_>>();
    let proposal = if harmonic_gate && events.len() >= MIN_EVENTS {
        let times = events.iter().map(|e| e.time_s).collect::<Vec<_>>();
        let rewards = events.iter().map(|e| reward(input, e)).collect::<Vec<_>>();
        path(&times, &rewards, a, b, work)
    } else {
        None
    };
    let disagreements = proposal
        .as_ref()
        .map_or_else(Vec::new, |p| disagreement_regions(&original, p, a, b));
    ActiveRegionCandidate {
        start_s: a,
        end_s: b,
        candidate_count: events.len(),
        status: if proposal.is_some() {
            ActiveRegionStatus::Proposal
        } else {
            ActiveRegionStatus::FallbackNoValidPath
        },
        original_times_s: original,
        proposal_times_s: proposal,
        disagreements,
    }
}

#[derive(Clone, Copy)]
struct State {
    score: f64,
    predecessor: usize,
}
struct Row {
    first: usize,
    states: Vec<State>,
}

fn path(
    times: &[f64],
    rewards: &[f64],
    start: f64,
    end: f64,
    work: &mut ActiveRegionWork,
) -> Option<Vec<f64>> {
    if times.len() < MIN_EVENTS {
        return None;
    }
    let mut rows = Vec::with_capacity(times.len());
    for (i, &t) in times.iter().enumerate() {
        // Contiguous, strictly ordered successor ranges avoid N*N allocation.
        let mut first = i + 1;
        while first < times.len() && times[first] - t < MIN_INTERVAL {
            first += 1;
        }
        let mut last = first;
        while last < times.len() && times[last] - t <= MAX_INTERVAL {
            last += 1;
        }
        let states = (first..last)
            .map(|j| State {
                score: if t <= start + MAX_INTERVAL {
                    rewards[i] + rewards[j]
                } else {
                    f64::NEG_INFINITY
                },
                predecessor: usize::MAX,
            })
            .collect::<Vec<_>>();
        work.pair_states += states.len();
        rows.push(Row { first, states });
    }
    let bytes = rows.capacity() * size_of::<Row>()
        + rows
            .iter()
            .map(|r| r.states.capacity() * size_of::<State>())
            .sum::<usize>();
    work.peak_dp_capacity_bytes = work.peak_dp_capacity_bytes.max(bytes);
    for previous in 0..times.len() {
        for offset in 0..rows[previous].states.len() {
            let current = rows[previous].first + offset;
            let score = rows[previous].states[offset].score;
            if !score.is_finite() {
                continue;
            }
            let interval = times[current] - times[previous];
            let first = rows[current].first;
            for (offset, state) in rows[current].states.iter_mut().enumerate() {
                let next = first + offset;
                let r = ((times[next] - times[current]) / interval).ln();
                let penalty =
                    (2.0 * r.powi(2)).min(0.5 + 2.0 * (r.abs() - std::f64::consts::LN_2).powi(2));
                let value = score + rewards[next] - penalty;
                work.transitions += 1;
                if value > state.score {
                    *state = State {
                        score: value,
                        predecessor: previous,
                    };
                }
            }
        }
    }
    let mut best = f64::NEG_INFINITY;
    let mut terminal = None;
    for (previous, row) in rows.iter().enumerate() {
        for (offset, state) in row.states.iter().enumerate() {
            let current = row.first + offset;
            if times[current] >= end - MAX_INTERVAL && state.score > best {
                best = state.score;
                terminal = Some((previous, current));
            }
        }
    }
    let (mut previous, mut current) = terminal?;
    let mut result = vec![times[current], times[previous]];
    loop {
        let predecessor = rows[previous].states[current - rows[previous].first].predecessor;
        if predecessor == usize::MAX {
            break;
        }
        current = previous;
        previous = predecessor;
        result.push(times[previous]);
    }
    result.reverse();
    (result.len() >= MIN_EVENTS).then_some(result)
}

fn disagreement_regions(
    primary: &[f64],
    alternative: &[f64],
    domain_start: f64,
    domain_end: f64,
) -> Vec<ActiveRegionDisagreement> {
    let mut anchors = Vec::new();
    let (mut i, mut j) = (0, 0);
    let (mut primary_only, mut alternative_only) = (Vec::new(), Vec::new());
    while i < primary.len() && j < alternative.len() {
        let difference = primary[i] - alternative[j];
        if difference.abs() <= f64::EPSILON {
            anchors.push(primary[i]);
            i += 1;
            j += 1;
        } else if difference < 0.0 {
            primary_only.push(primary[i]);
            i += 1;
        } else {
            alternative_only.push(alternative[j]);
            j += 1;
        }
    }
    primary_only.extend_from_slice(&primary[i..]);
    alternative_only.extend_from_slice(&alternative[j..]);
    let count = |times: &[f64], start, end, left, right| {
        let from = times.partition_point(|t| if left { *t <= start } else { *t < start });
        let to = times.partition_point(|t| if right { *t < end } else { *t <= end });
        to.saturating_sub(from)
    };
    let mut result = Vec::new();
    let (mut start, mut left) = (domain_start, false);
    for (end, right) in anchors
        .into_iter()
        .map(|t| (t, true))
        .chain([(domain_end, false)])
    {
        let p = count(&primary_only, start, end, left, right);
        let q = count(&alternative_only, start, end, left, right);
        if p != 0 || q != 0 {
            result.push(ActiveRegionDisagreement {
                start_s: start,
                end_s: end,
                left_anchored: left,
                right_anchored: right,
                primary_only_beat_count: p,
                alternative_only_beat_count: q,
            });
        }
        start = end;
        left = right;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhythm_map_core::{
        AudioActivityPoint, AudioHarmonicChangePoint, ModelInfo, ObservedBeat, analyze_observations,
    };

    #[allow(clippy::needless_pass_by_value)] // Convenient ownership of authored fixture vectors.
    fn fixture(times: Vec<f64>, duration_s: f64) -> RhythmObservations {
        RhythmObservations {
            duration_s,
            beats: times
                .iter()
                .map(|&time_s| ObservedBeat {
                    time_s,
                    confidence: 0.8,
                    downbeat_confidence: 0.0,
                })
                .collect(),
            beat_candidates: times
                .iter()
                .map(|&time_s| BeatCandidate {
                    time_s,
                    confidence: 0.8,
                    downbeat_confidence: 0.0,
                })
                .collect(),
            harmonic_changes: times
                .iter()
                .map(|&time_s| AudioHarmonicChangePoint {
                    time_s,
                    strength: 0.2,
                })
                .collect(),
            activations: None,
            activity: vec![],
            onsets: vec![],
            source: ModelInfo {
                backend: "fixture".into(),
                model: "authored".into(),
                version: None,
                frame_rate_hz: None,
            },
        }
    }

    // Intentionally independent dense traversal. Tests ordering, floating-point
    // boundary inclusion, ties and traceback, not just score equality.
    #[allow(clippy::many_single_char_names, clippy::needless_range_loop)] // Independent dense mathematical reference.
    fn dense(times: &[f64], rewards: &[f64], a: f64, b: f64) -> Option<Vec<f64>> {
        let n = times.len();
        let mut scores = vec![vec![f64::NEG_INFINITY; n]; n];
        let mut back = vec![vec![None; n]; n];
        for i in 0..n {
            for j in i + 1..n {
                if times[i] <= a + MAX_INTERVAL
                    && (MIN_INTERVAL..=MAX_INTERVAL).contains(&(times[j] - times[i]))
                {
                    scores[i][j] = rewards[i] + rewards[j];
                }
            }
        }
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    if scores[i][j].is_finite()
                        && (MIN_INTERVAL..=MAX_INTERVAL).contains(&(times[k] - times[j]))
                    {
                        let r = ((times[k] - times[j]) / (times[j] - times[i])).ln();
                        let value = scores[i][j] + rewards[k]
                            - (2.0 * r * r).min(0.5 + 2.0 * (r.abs() - 2.0_f64.ln()).powi(2));
                        if value > scores[j][k] {
                            scores[j][k] = value;
                            back[j][k] = Some(i);
                        }
                    }
                }
            }
        }
        let mut best = f64::NEG_INFINITY;
        let mut terminal = None;
        for i in 0..n {
            for j in i + 1..n {
                if times[j] >= b - MAX_INTERVAL && scores[i][j] > best {
                    terminal = Some((i, j));
                    best = scores[i][j];
                }
            }
        }
        let (mut i, mut j) = terminal?;
        let mut result = vec![times[j], times[i]];
        while let Some(previous) = back[i][j] {
            j = i;
            i = previous;
            result.push(times[i]);
        }
        result.reverse();
        (result.len() >= 8).then_some(result)
    }

    #[test]
    fn sparse_matches_dense_including_negative_rewards_ties_and_offsets() {
        for spacing in [0.125, 0.1875, 0.25, 0.5, 1.0, 1.5, 2.0] {
            for offset in [0.0, 10.0] {
                for reward in [-1.0, 0.0, 1.0] {
                    for varied in [false, true] {
                        let ts = (0..16)
                            .map(|i| offset + f64::from(i) * spacing)
                            .collect::<Vec<_>>();
                        let rs = (0..16)
                            .map(|i| reward + if varied { f64::from(i % 3) * 0.01 } else { 0.0 })
                            .collect::<Vec<_>>();
                        assert_eq!(
                            path(&ts, &rs, offset, ts[15], &mut ActiveRegionWork::default()),
                            dense(&ts, &rs, offset, ts[15])
                        );
                    }
                }
            }
        }
        // Deterministic nonuniform graphs, independent of calibration material.
        for seed in 0..32 {
            let mut ts = vec![0.0];
            let mut rs = vec![0.8];
            for i in 1..24 {
                ts.push(
                    ts[i - 1]
                        + f64::from(((seed + 3) * i32::try_from(i).unwrap()) % 13 + 1) * 0.0625,
                );
                rs.push(f64::from((seed + i32::try_from(i).unwrap()) % 9) * 0.2 - 0.5);
            }
            assert_eq!(
                path(&ts, &rs, 0.0, ts[23], &mut ActiveRegionWork::default()),
                dense(&ts, &rs, 0.0, ts[23])
            );
        }
    }

    #[test]
    fn unsupported_padding_and_rest_split_without_bridging_or_adoption() {
        let times = (0..8)
            .map(|i| 3.0 + f64::from(i) * 0.5)
            .chain((0..8).map(|i| 13.0 + f64::from(i) * 0.5))
            .collect::<Vec<_>>();
        let input = fixture(times.clone(), 19.0);
        let before = analyze_observations(&input).unwrap();
        let selected = before.beats.iter().map(|b| b.time_s).collect::<Vec<_>>();
        let result = generate_active_region_candidates(&input, &selected).unwrap();
        assert_eq!(
            result.unknown_gaps,
            vec![[0.0, 3.0], [16.5, 19.0], [6.5, 13.0]]
        );
        assert_eq!(result.proposals.len(), 2);
        let proposed = result
            .proposals
            .iter()
            .flat_map(|p| p.proposal_times_s.clone().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(proposed, times);
        assert!(result.proposals.iter().all(|p| p.disagreements.is_empty()));
        assert_eq!(before, analyze_observations(&input).unwrap());
    }

    #[test]
    fn inclusive_silence_removes_candidates_but_does_not_mutate_primary() {
        let mut input = fixture((0..33).map(|i| f64::from(i) * 0.5).collect(), 16.0);
        input.activity = (0..32)
            .map(|i| {
                let time_s = f64::from(i) * 0.5 + 0.25;
                AudioActivityPoint {
                    time_s,
                    rms: 1.0,
                    relative_db: if (6.0..10.0).contains(&time_s) {
                        -40.0
                    } else {
                        0.0
                    },
                }
            })
            .collect();
        let primary = input.beats.iter().map(|b| b.time_s).collect::<Vec<_>>();
        let result = generate_active_region_candidates(&input, &primary).unwrap();
        assert_eq!(result.silence_regions, vec![[6.0, 10.0]]);
        assert_eq!(result.proposals.len(), 2);
        assert!(
            result
                .proposals
                .iter()
                .flat_map(|p| p.proposal_times_s.as_ref().unwrap())
                .all(|t| !(6.0..=10.0).contains(t))
        );
        assert_eq!(primary.len(), 33);
    }

    #[test]
    fn insufficient_evidence_returns_fallback_not_an_empty_replacement() {
        let mut input = fixture((0..16).map(|i| f64::from(i) * 0.5).collect(), 8.0);
        input.harmonic_changes.clear();
        let primary = vec![0.0, 0.5];
        let result = generate_active_region_candidates(&input, &primary).unwrap();
        assert_eq!(
            result.proposals[0].status,
            ActiveRegionStatus::FallbackNoValidPath
        );
        assert_eq!(result.proposals[0].original_times_s, primary);
        assert!(result.proposals[0].proposal_times_s.is_none());
        input.beat_candidates.truncate(1);
        let result = generate_active_region_candidates(&input, &[]).unwrap();
        assert_eq!(
            result.proposals[0].status,
            ActiveRegionStatus::FallbackNoValidPath
        );
        input.beat_candidates.clear();
        let result = generate_active_region_candidates(&input, &primary).unwrap();
        assert!(result.proposals.is_empty());
        assert_eq!(result.unknown_gaps, vec![[0.0, 8.0]]);
    }

    #[test]
    fn unscorable_primary_stays_unscored_even_with_valid_alternative() {
        let input = fixture((0..16).map(|i| f64::from(i) * 0.5).collect(), 8.0);
        let result = generate_active_region_candidates(&input, &[0.0, 0.5]).unwrap();
        assert_eq!(result.proposals[0].status, ActiveRegionStatus::Proposal);
        assert_eq!(result.proposals[0].original_times_s, vec![0.0, 0.5]);
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("score"));
        assert!(!json.contains("confidence"));
    }

    #[test]
    fn real_tempo_change_retained_and_sparse_storage_is_bounded_by_edges() {
        let ts = (0..8)
            .map(|i| f64::from(i) * 0.5)
            .chain((0..8).map(|i| 4.0 + f64::from(i) * 0.25))
            .collect::<Vec<_>>();
        assert_eq!(
            path(
                &ts,
                &[1.0; 16],
                0.0,
                ts[15],
                &mut ActiveRegionWork::default()
            ),
            Some(ts)
        );
        let ts = (0..4000).map(|i| f64::from(i) * 0.25).collect::<Vec<_>>();
        let mut work = ActiveRegionWork::default();
        assert!(path(&ts, &vec![1.0; 4000], 0.0, ts[3999], &mut work).is_some());
        assert!(work.pair_states <= 4000 * 6);
        assert!(work.peak_dp_capacity_bytes < 1_000_000);
    }

    #[test]
    fn shared_anchors_and_one_sided_disagreements_match_existing_semantics() {
        let p = [0.0, 1.0, 2.0, 3.0, 4.0];
        let q = [1.0, 2.5, 3.0, 5.0];
        let r = disagreement_regions(&p, &q, 0.0, 6.0);
        assert_eq!(
            r.iter()
                .map(|r| (
                    r.left_anchored,
                    r.right_anchored,
                    r.primary_only_beat_count,
                    r.alternative_only_beat_count
                ))
                .collect::<Vec<_>>(),
            vec![(false, true, 1, 0), (true, true, 1, 1), (true, false, 1, 1)]
        );
        assert_eq!(
            r.iter().map(|r| [r.start_s, r.end_s]).collect::<Vec<_>>(),
            vec![[0.0, 1.0], [1.0, 3.0], [3.0, 6.0]]
        );
        let r = disagreement_regions(&[0.0], &[1.0], 0.0, 1.0);
        assert_eq!(
            (
                r[0].primary_only_beat_count,
                r[0].alternative_only_beat_count
            ),
            (1, 1)
        );
        assert!(!r[0].left_anchored && !r[0].right_anchored);
    }

    #[test]
    fn nearest_harmonic_ties_and_exact_tolerance_are_preserved() {
        let mut input = fixture(vec![0.5], 1.0);
        input.harmonic_changes = vec![
            AudioHarmonicChangePoint {
                time_s: 0.484_375,
                strength: 0.2,
            },
            AudioHarmonicChangePoint {
                time_s: 0.515_625,
                strength: 0.8,
            },
        ];
        assert!((reward(&input, &input.beat_candidates[0]) - 0.85).abs() < 1e-12);
        input.harmonic_changes = vec![AudioHarmonicChangePoint {
            time_s: 0.520_001,
            strength: 1.0,
        }];
        assert!((reward(&input, &input.beat_candidates[0]) + 0.15).abs() < 1e-12);
    }

    #[test]
    fn invalid_inputs_fail_before_search() {
        let input = fixture(vec![0.0, 0.5], 1.0);
        for times in [
            vec![f64::NAN],
            vec![-0.1],
            vec![1.1],
            vec![0.5, 0.0],
            vec![0.0, 0.0],
        ] {
            assert!(generate_active_region_candidates(&input, &times).is_err());
        }
        let mut bad = input.clone();
        bad.beat_candidates.reverse();
        assert!(generate_active_region_candidates(&bad, &[]).is_err());
        let mut bad = input.clone();
        bad.harmonic_changes[0].strength = f64::NAN;
        assert!(generate_active_region_candidates(&bad, &[]).is_err());
        let mut bad = input;
        bad.duration_s = f64::INFINITY;
        assert!(generate_active_region_candidates(&bad, &[]).is_err());
    }
}
