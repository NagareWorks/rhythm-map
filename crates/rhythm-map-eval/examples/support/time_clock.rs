//! Controlled time-exposure intervention; frozen joint graph and emissions.
#[path = "time_prior.rs"]
pub mod time_prior;
use anyhow::{Result, ensure};
use serde::Serialize;

const NONE: usize = usize::MAX;
const NEG: f64 = f64::NEG_INFINITY;

#[derive(Clone, Copy)]
pub struct Domain {
    pub min_period: usize,
    pub max_period: usize,
    pub min_meter: usize,
    pub max_meter: usize,
}

impl Default for Domain {
    fn default() -> Self {
        Self {
            min_period: 10,
            max_period: 75,
            min_meter: 2,
            max_meter: 7,
        }
    }
}

#[derive(Clone, Copy)]
struct Node {
    mass: f64,
    reference: f64,
    best: f64,
    trace: u64,
    origin: usize,
}

impl Node {
    const EMPTY: Self = Self {
        mass: NEG,
        reference: NEG,
        best: NEG,
        trace: 0,
        origin: NONE,
    };

    fn shift(mut self, prior: f64, evidence: f64) -> Self {
        self.mass += prior + evidence;
        self.reference += prior;
        self.best += prior + evidence;
        self
    }

    fn merge(&mut self, other: Self) {
        self.mass = log_add(self.mass, other.mass);
        self.reference = log_add(self.reference, other.reference);
        if other.best > self.best {
            self.best = other.best;
            self.trace = other.trace;
            self.origin = other.origin;
        }
    }
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == NEG {
        return b;
    }
    if b == NEG {
        return a;
    }
    let maximum = a.max(b);
    maximum + (-(a - b).abs()).exp().ln_1p()
}

/// Exact off-diagonal L1 sweeps plus a separate self-survival term.
fn transitions(input: &[Node], prior: &time_prior::Prior, output: &mut [Node]) {
    let coordinates = &prior.coordinates;
    let mut carry = Node::EMPTY;
    for j in 0..input.len() {
        if j > 0 {
            carry = carry.shift(coordinates[j - 1] - coordinates[j], 0.);
        }
        output[j] = carry;
        carry.merge(input[j].shift(prior.log_jump_base[j], 0.));
    }
    carry = Node::EMPTY;
    for j in (0..input.len()).rev() {
        if j + 1 < input.len() {
            carry = carry.shift(coordinates[j] - coordinates[j + 1], 0.);
        }
        output[j].merge(carry);
        output[j].merge(input[j].shift(prior.log_survival[j], 0.));
        carry.merge(input[j].shift(prior.log_jump_base[j], 0.));
    }
}

/// Same cyclic [1,2,1]/4 statistic as `phase_likelihood::cell`, phase zero.
/// Extending the cell changes only its two endpoints and one interior term.
/// This builds all lengths in linear time without approximating the normalizer.
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
fn cell_table(values: &[f32], max_length: usize) -> Vec<f64> {
    let stride = max_length + 1;
    let mut result = vec![0.; values.len() * stride];
    for start in 0..values.len() {
        let mut interior = NEG;
        let mut varies = false;
        for length in 3..=max_length.min(values.len() - start) {
            let end = start + length;
            varies |= values[start + 1] != values[start] || values[end - 1] != values[start];
            let stat = |a: usize, b: usize, c: usize| {
                0.25 * f64::from(values[a])
                    + 0.5 * f64::from(values[b])
                    + 0.25 * f64::from(values[c])
            };
            interior = log_add(interior, stat(end - 3, end - 2, end - 1));
            let first = stat(end - 1, start, start + 1);
            let last = stat(end - 2, end - 1, start);
            result[start * stride + length] =
                first - log_add(log_add(interior, first), last) + (length as f64).ln();
            if !varies {
                result[start * stride + length] = 0.;
            }
        }
    }
    result
}

#[derive(Debug, Serialize)]
pub struct Tick {
    pub frame: usize,
    pub period_frames: usize,
    pub meter: usize,
    pub beat_in_bar: usize,
    pub beat_cell_log_ratio: f64,
    /// A locally flat head supplies no timing evidence; this is not a detection.
    pub prior_only: bool,
}

#[derive(Debug, Serialize)]
pub struct Run {
    pub start: usize,
    pub end: usize,
    pub clock_log_ratio_to_null: Option<f64>,
    pub log_reference_partition: Option<f64>,
    pub map_log_probability_given_clock: Option<f64>,
    pub clock_supported: bool,
    /// MAP diagnostics only, even when the null wins. Never observed beats.
    pub map_ticks: Vec<Tick>,
    pub map_complete_bar_span: Option<[usize; 2]>,
    pub edge_reference_frames: usize,
}

#[derive(Debug, Serialize)]
pub struct Decoded {
    pub runs: Vec<Run>,
    pub unavailable_spans: Vec<[usize; 2]>,
}

/// Missing frames split the graph, not just emissions. No path bridges a gap.
/// The null is the same product reference on the entire available frame domain;
/// unknown prefix/suffix frames have likelihood ratio one, not fabricated ticks.
pub fn decode(
    beat: &[f32],
    bar: &[f32],
    available: Option<&[bool]>,
    domain: Domain,
) -> Result<Decoded> {
    ensure!(
        !beat.is_empty() && beat.len() == bar.len(),
        "head length mismatch or empty input"
    );
    ensure!(
        beat.iter().chain(bar).all(|v| v.is_finite()),
        "non-finite heads"
    );
    ensure!(
        available.is_none_or(|a| a.len() == beat.len()),
        "availability length mismatch"
    );
    ensure!(
        domain.min_period >= 3
            && domain.max_period >= domain.min_period
            && domain.max_period <= 255,
        "invalid period domain"
    );
    ensure!(
        domain.min_meter >= 2 && domain.max_meter >= domain.min_meter && domain.max_meter <= 7,
        "invalid meter domain"
    );
    let mut result = Decoded {
        runs: Vec::new(),
        unavailable_spans: Vec::new(),
    };
    let mut start = 0;
    while start < beat.len() {
        let present = available.is_none_or(|a| a[start]);
        let mut end = start + 1;
        while end < beat.len() && available.is_none_or(|a| a[end]) == present {
            end += 1;
        }
        if present {
            result
                .runs
                .push(search(&beat[start..end], &bar[start..end], start, domain));
        } else {
            result.unavailable_spans.push([start, end]);
        }
        start = end;
    }
    Ok(result)
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines, clippy::float_cmp)]
fn search(beat: &[f32], bar: &[f32], offset: usize, domain: Domain) -> Run {
    let n = beat.len();
    let count = domain.max_period - domain.min_period + 1;
    let max_bar = domain.max_period * domain.max_meter;
    let mut result = Run {
        start: offset,
        end: offset + n,
        clock_log_ratio_to_null: None,
        log_reference_partition: None,
        map_log_probability_given_clock: None,
        clock_supported: false,
        map_ticks: Vec::new(),
        map_complete_bar_span: None,
        edge_reference_frames: n,
    };
    if n < domain.min_period * domain.min_meter {
        return result;
    }
    let prior =
        time_prior::Prior::new(&(domain.min_period..=domain.max_period).collect::<Vec<_>>());
    let beat_table = cell_table(beat, domain.max_period);
    let bar_table = cell_table(bar, max_bar);
    let start_count = max_bar.min(n - domain.min_period * domain.min_meter + 1);
    let start_prior = -((start_count * count) as f64).ln();
    let meter_prior = -((domain.max_meter - domain.min_meter + 1) as f64).ln();
    let tail_prior = -(max_bar as f64).ln();
    let mut global = vec![Node::EMPTY; (n + 1) * count];
    let mut current = vec![Node::EMPTY; (max_bar + 1) * count];
    let mut next = current.clone();
    let mut source = vec![Node::EMPTY; count];
    let mut transformed = source.clone();
    for start in 0..=n - domain.min_period * domain.min_meter {
        current.fill(Node::EMPTY);
        for (p, value) in source.iter_mut().enumerate() {
            *value = global[start * count + p];
            value.origin = start * count + p;
            value.trace = 0;
        }
        transitions(&source, &prior, &mut transformed);
        for (p, value) in transformed.iter_mut().enumerate() {
            if start < start_count {
                value.merge(Node {
                    mass: start_prior,
                    reference: start_prior,
                    best: start_prior,
                    trace: 0,
                    origin: NONE,
                });
            }
            let duration = domain.min_period + p;
            if start + duration <= n {
                let mut child =
                    value.shift(0., beat_table[start * (domain.max_period + 1) + duration]);
                child.trace = duration as u64;
                current[duration * count + p] = child;
            }
        }
        for beats in 1..=domain.max_meter {
            let min_elapsed = beats * domain.min_period;
            let max_elapsed = (beats * domain.max_period).min(n - start);
            for elapsed in min_elapsed..=max_elapsed {
                if beats >= domain.min_meter {
                    let evidence = bar_table[start * (max_bar + 1) + elapsed];
                    for p in 0..count {
                        global[(start + elapsed) * count + p]
                            .merge(current[elapsed * count + p].shift(meter_prior, evidence));
                    }
                }
            }
            if beats == domain.max_meter {
                break;
            }
            next.fill(Node::EMPTY);
            for elapsed in min_elapsed..=max_elapsed {
                let row = &current[elapsed * count..(elapsed + 1) * count];
                if row.iter().all(|v| v.mass == NEG) {
                    continue;
                }
                transitions(row, &prior, &mut transformed);
                for (p, value) in transformed.iter().enumerate() {
                    let duration = domain.min_period + p;
                    if start + elapsed + duration <= n {
                        let mut child = value.shift(
                            0.,
                            beat_table[(start + elapsed) * (domain.max_period + 1) + duration],
                        );
                        child.trace |= (duration as u64) << (8 * beats);
                        next[(elapsed + duration) * count + p].merge(child);
                    }
                }
            }
            std::mem::swap(&mut current, &mut next);
        }
    }
    let mut terminal = Node::EMPTY;
    let mut best_index = NONE;
    for end in n.saturating_sub(max_bar - 1)..=n {
        for p in 0..count {
            let index = end * count + p;
            let candidate = global[index].shift(tail_prior + prior.log_survival[p], 0.);
            if candidate.best > terminal.best {
                best_index = index;
            }
            terminal.merge(candidate);
        }
    }
    if best_index == NONE {
        return result;
    }
    let evidence = terminal.mass - terminal.reference;
    result.clock_log_ratio_to_null = Some(evidence);
    result.log_reference_partition = Some(terminal.reference);
    result.map_log_probability_given_clock = Some(terminal.best - terminal.mass);
    result.clock_supported = evidence > 1e-9;
    let span_end = best_index / count;
    let mut index = best_index;
    let mut reversed_bars = Vec::new();
    loop {
        let node = global[index];
        let durations: Vec<usize> = (0..7)
            .map(|i| usize::try_from((node.trace >> (8 * i)) & 255).unwrap())
            .take_while(|&p| p != 0)
            .collect();
        let start = index / count - durations.iter().sum::<usize>();
        reversed_bars.push((start, durations));
        if node.origin == NONE {
            break;
        }
        index = node.origin;
    }
    let span_start = reversed_bars.last().unwrap().0;
    result.edge_reference_frames = span_start + n - span_end;
    result.map_complete_bar_span = Some([offset + span_start, offset + span_end]);
    for (start, durations) in reversed_bars.into_iter().rev() {
        let mut frame = start;
        let meter = durations.len();
        for (beat_in_bar, period_frames) in durations.into_iter().enumerate() {
            let values = &beat[frame..frame + period_frames];
            result.map_ticks.push(Tick {
                frame: frame + offset,
                period_frames,
                meter,
                beat_in_bar,
                beat_cell_log_ratio: beat_table[frame * (domain.max_period + 1) + period_frames],
                prior_only: values.iter().all(|v| *v == values[0]),
            });
            frame += period_frames;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Domain {
        Domain {
            min_period: 3,
            max_period: 5,
            min_meter: 2,
            max_meter: 3,
        }
    }

    #[test]
    fn incremental_cell_normalizers_match_the_frozen_scorer() {
        let values = [-8., -2., -2., -2., -8., 8., -8., -7., -4., -8.];
        let table = cell_table(&values, values.len());
        for start in 0..values.len() {
            for length in 3..=values.len() - start {
                let slow = crate::reference::cell(&values[start..start + length], 0).unwrap();
                assert!(
                    (table[start * (values.len() + 1) + length] - slow.log_ratio_to_null).abs()
                        < 1e-12
                );
            }
        }
        for value in [-f32::MAX, f32::MAX, 0.] {
            let table = cell_table(&[value; 8], 8);
            assert!(table.iter().all(|v| v.abs() < 1e-12));
        }
    }

    struct Enumeration<'a> {
        beat: &'a [f32],
        bar: &'a [f32],
        domain: Domain,
        total: Node,
    }

    impl Enumeration<'_> {
        #[allow(clippy::cast_precision_loss)]
        fn bars(&mut self, start: usize, previous: Option<usize>, weight: Node) {
            for meter in self.domain.min_meter..=self.domain.max_meter {
                self.beats(
                    start,
                    start,
                    previous,
                    meter,
                    weight.shift(
                        -((self.domain.max_meter - self.domain.min_meter + 1) as f64).ln(),
                        0.,
                    ),
                );
            }
        }

        #[allow(clippy::cast_precision_loss)]
        fn beats(
            &mut self,
            bar_start: usize,
            cursor: usize,
            previous: Option<usize>,
            remaining: usize,
            weight: Node,
        ) {
            if remaining == 0 {
                let evidence = crate::reference::cell(&self.bar[bar_start..cursor], 0)
                    .unwrap()
                    .log_ratio_to_null;
                let finished = weight.shift(0., evidence);
                let max_bar = self.domain.max_period * self.domain.max_meter;
                if self.beat.len() - cursor < max_bar {
                    let prior = time_prior::Prior::new(
                        &(self.domain.min_period..=self.domain.max_period).collect::<Vec<_>>(),
                    );
                    self.total.merge(finished.shift(
                        -(max_bar as f64).ln()
                            + prior.log_survival[previous.unwrap() - self.domain.min_period],
                        0.,
                    ));
                }
                if cursor + self.domain.min_period * self.domain.min_meter <= self.beat.len() {
                    self.bars(cursor, previous, finished);
                }
                return;
            }
            for duration in self.domain.min_period..=self.domain.max_period {
                if cursor + duration > self.beat.len() {
                    continue;
                }
                let prior = previous.map_or_else(
                    || -((self.domain.max_period - self.domain.min_period + 1) as f64).ln(),
                    |p| {
                        let prior = time_prior::Prior::new(
                            &(self.domain.min_period..=self.domain.max_period).collect::<Vec<_>>(),
                        );
                        prior.transition(
                            p - self.domain.min_period,
                            duration - self.domain.min_period,
                        )
                    },
                );
                let evidence = crate::reference::cell(&self.beat[cursor..cursor + duration], 0)
                    .unwrap()
                    .log_ratio_to_null;
                self.beats(
                    bar_start,
                    cursor + duration,
                    Some(duration),
                    remaining - 1,
                    weight.shift(prior, evidence),
                );
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn both_partitions_and_map_match_exhaustive_unknown_path_enumeration() {
        let beat = [
            -8., -2., -2., -8., -8., -1., -8., -8., 3., -4., -8., -8., -1., -2., -8.,
        ];
        let bar = [
            -8., -2., -3., -8., -8., -8., -8., -8., 2., -4., -8., -8., -8., -8., -8.,
        ];
        let domain = small();
        let starts = (domain.max_period * domain.max_meter)
            .min(beat.len() - domain.min_period * domain.min_meter + 1);
        let mut enumeration = Enumeration {
            beat: &beat,
            bar: &bar,
            domain,
            total: Node::EMPTY,
        };
        for start in 0..starts {
            let prior = -(starts as f64).ln();
            enumeration.bars(
                start,
                None,
                Node {
                    mass: prior,
                    reference: prior,
                    best: prior,
                    trace: 0,
                    origin: NONE,
                },
            );
        }
        let run = search(&beat, &bar, 0, domain);
        let total = enumeration.total;
        assert!((run.log_reference_partition.unwrap() - total.reference).abs() < 1e-11);
        assert!(
            (run.clock_log_ratio_to_null.unwrap() - (total.mass - total.reference)).abs() < 1e-11
        );
        assert!(
            (run.map_log_probability_given_clock.unwrap() - (total.best - total.mass)).abs()
                < 1e-11
        );
        // Check the returned traceback, not merely the internal max score.
        let mut reconstructed =
            -(starts as f64).ln() - ((domain.max_period * domain.max_meter) as f64).ln();
        let mut previous = None;
        let mut bar_start = 0;
        for tick in &run.map_ticks {
            if tick.beat_in_bar == 0 {
                bar_start = tick.frame;
                reconstructed -= ((domain.max_meter - domain.min_meter + 1) as f64).ln();
            }
            let duration = tick.period_frames;
            reconstructed += previous.map_or_else(
                || -((domain.max_period - domain.min_period + 1) as f64).ln(),
                |p: usize| {
                    let prior = time_prior::Prior::new(
                        &(domain.min_period..=domain.max_period).collect::<Vec<_>>(),
                    );
                    prior.transition(p - domain.min_period, duration - domain.min_period)
                },
            );
            reconstructed += crate::reference::cell(&beat[tick.frame..tick.frame + duration], 0)
                .unwrap()
                .log_ratio_to_null;
            if tick.beat_in_bar + 1 == tick.meter {
                reconstructed += crate::reference::cell(&bar[bar_start..tick.frame + duration], 0)
                    .unwrap()
                    .log_ratio_to_null;
            }
            previous = Some(duration);
        }
        let prior =
            time_prior::Prior::new(&(domain.min_period..=domain.max_period).collect::<Vec<_>>());
        reconstructed += prior.log_survival[previous.unwrap() - domain.min_period];
        assert!((reconstructed - total.best).abs() < 1e-11);
    }

    #[test]
    fn flat_heads_cannot_turn_hypothesis_count_into_evidence() {
        for length in [12, 24, 37] {
            let r = decode(&vec![-8.; length], &vec![8.; length], None, small()).unwrap();
            let run = &r.runs[0];
            assert!(run.clock_log_ratio_to_null.unwrap().abs() < 1e-10);
            assert!(!run.clock_supported);
            assert!(run.map_ticks.iter().all(|t| t.prior_only));
        }
    }

    #[test]
    fn optimized_transform_matches_all_pairs_without_double_counting() {
        let prior = time_prior::Prior::new(&[3, 4, 5, 6]);
        let source: Vec<Node> = [0.3, -4., 1.2, -2.]
            .into_iter()
            .enumerate()
            .map(|(i, mass)| Node {
                mass,
                reference: -mass,
                best: mass,
                trace: i as u64,
                origin: i,
            })
            .collect();
        let mut fast = vec![Node::EMPTY; 4];
        transitions(&source, &prior, &mut fast);
        for (j, actual) in fast.iter().enumerate() {
            let mut slow = Node::EMPTY;
            for (i, value) in source.iter().enumerate() {
                slow.merge(value.shift(prior.transition(i, j), 0.));
            }
            assert!((actual.mass - slow.mass).abs() < 1e-12);
            assert!((actual.reference - slow.reference).abs() < 1e-12);
            assert!((actual.best - slow.best).abs() < 1e-12);
            assert_eq!(actual.origin, slow.origin);
        }
    }

    #[test]
    fn gaps_split_paths_and_invalid_input_fails_closed() {
        let x = vec![-8.; 60];
        let mut available = vec![true; 60];
        available[25..35].fill(false);
        let r = decode(&x, &x, Some(&available), small()).unwrap();
        assert_eq!(r.unavailable_spans, [[25, 35]]);
        assert_eq!(r.runs.len(), 2);
        for run in r.runs {
            for tick in run.map_ticks {
                assert!(tick.frame >= run.start && tick.frame + tick.period_frames <= run.end);
            }
        }
        assert!(decode(&x, &[], None, small()).is_err());
        assert!(decode(&[f32::NAN], &[0.], None, small()).is_err());
        assert!(decode(&x, &x, Some(&[]), small()).is_err());
    }
}
