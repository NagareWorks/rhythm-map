//! Evaluation-only, full-frame renewal clock. No labels or selected events.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

const MIN_PERIOD: usize = 10;
const MAX_PERIOD: usize = 75;
const PERIODS: usize = MAX_PERIOD - MIN_PERIOD + 1;
const RING: usize = MAX_PERIOD + 1;
const START: u8 = u8::MAX;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frames {
    pub beat_logits: Vec<f32>,
    pub downbeat_logits: Vec<f32>,
    /// Explicit absence, not a low activation or a silence detector.
    pub available: Option<Vec<bool>>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Tick {
    pub frame: usize,
    pub period_frames: usize,
    pub bar_phase: usize,
    pub missing_component: bool,
    pub positive_pulse_window: bool,
    pub pulse_contrast: f64,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct Component {
    pub start_frame: usize,
    pub end_frame: usize,
    pub meter_hypothesis_not_estimate: usize,
    pub score_not_confidence: f64,
    pub ticks: Vec<Tick>,
}

#[derive(Serialize)]
pub struct Decoding {
    pub components: Vec<Component>,
    pub unavailable_frames: usize,
    pub uninformative_frames: usize,
    pub max_backpointer_bytes: usize,
    pub period_limits_frames: [usize; 2],
    pub meter_hypotheses: [usize; 6],
}

#[allow(clippy::cast_precision_loss)]
fn real(value: usize) -> f64 {
    value as f64
}

fn log_sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        -(-value).exp().ln_1p()
    } else {
        value - value.exp().ln_1p()
    }
}

fn max_window(values: &[f32], center: usize, radius: usize) -> f64 {
    values[center.saturating_sub(radius)..(center + radius + 1).min(values.len())]
        .iter()
        .map(|&v| f64::from(v))
        .fold(f64::NEG_INFINITY, f64::max)
}

#[derive(Clone, Copy, Default)]
struct Emission {
    pulse: f64,
    bar: f64,
    contrast: f64,
    missing: bool,
    positive: bool,
}

fn emission(beat: &[f32], downbeat: &[f32], time: usize, period: usize) -> Emission {
    let radius = (period / 16).clamp(1, 2);
    let peak = max_window(beat, time, radius);
    // The comparison uses actual frames in the preceding proposed interval.
    // Rounded integer quarters are part of the fixed 50 Hz model, not labels.
    let background = |values: &[f32]| {
        (1..=3)
            .map(|q| max_window(values, time - period + (period * q + 2) / 4, radius))
            .sum::<f64>()
            / 3.0
    };
    let contrast = peak - background(beat);
    let visible = log_sigmoid(peak);
    let missing = 0.1_f64.ln() + log_sigmoid(-peak);
    Emission {
        pulse: contrast + visible.max(missing),
        bar: max_window(downbeat, time, radius) - background(downbeat),
        contrast,
        missing: missing > visible,
        positive: peak > 0.0,
    }
}

/// Exact max-plus L1 distance transform on log-period coordinates. Two sweeps
/// replace the exhaustive predecessor search; deterministic ties keep lower p.
#[allow(clippy::float_cmp)] // Exact ties, not approximate posterior equality.
fn transitions(scores: &[f64], positions: &[f64], out: &mut [f64], args: &mut [u8]) {
    let penalty = 100.0_f64.ln();
    for i in 0..scores.len() {
        out[i] = scores[i];
        args[i] = u8::try_from(i).expect("period index fits u8");
        if i > 0 {
            let prior = usize::from(args[i - 1]);
            let value = scores[prior] - penalty * (positions[i] - positions[prior]);
            if value > out[i] || (value == out[i] && prior < usize::from(args[i])) {
                out[i] = value;
                args[i] = args[i - 1];
            }
        }
    }
    for i in (0..scores.len() - 1).rev() {
        let prior = usize::from(args[i + 1]);
        let value = scores[prior] - penalty * (positions[i] - positions[prior]).abs();
        if value > out[i] || (value == out[i] && prior < usize::from(args[i])) {
            out[i] = value;
            args[i] = args[i + 1];
        }
    }
}

#[allow(clippy::too_many_lines)]
fn clock(pulse: &[f32], downbeat: &[f32], meter: usize) -> Component {
    let frames = pulse.len();
    let width = PERIODS * meter;
    let mut incoming = vec![f64::NEG_INFINITY; RING * width];
    let mut incoming_arg = vec![START; RING * width];
    let mut backs = vec![START; frames * width];
    let mut row = vec![f64::NEG_INFINITY; width];
    let positions: Vec<f64> = (MIN_PERIOD..=MAX_PERIOD).map(|p| real(p).log2()).collect();
    let mut best = f64::NEG_INFINITY;
    let (mut end, mut end_phase, mut end_period) = (0, 0, 0);
    for time in 0..frames {
        let evidence: Vec<Emission> = (MIN_PERIOD..=MAX_PERIOD)
            .map(|p| {
                if time >= p {
                    emission(pulse, downbeat, time, p)
                } else {
                    Emission::default()
                }
            })
            .collect();
        for phase in 0..meter {
            for (index, e) in evidence.iter().enumerate() {
                let period = MIN_PERIOD + index;
                let state = phase * PERIODS + index;
                row[state] = if time < period {
                    // Free phase at the start, not a required detected anchor.
                    0.0
                } else {
                    let source = (time - period) % RING * width
                        + (phase + meter - 1) % meter * PERIODS
                        + index;
                    backs[time * width + state] = incoming_arg[source];
                    incoming[source] + e.pulse + if phase == 0 { e.bar } else { 0.0 }
                };
                // Every terminal hypothesis accounts for the final clock phase;
                // its partial tail is retained as prior-only, not a beat event.
                if time + period >= frames && row[state] > best {
                    best = row[state];
                    (end, end_phase, end_period) = (time, phase, index);
                }
            }
        }
        let target = time % RING * width;
        for phase in 0..meter {
            let start = phase * PERIODS;
            transitions(
                &row[start..start + PERIODS],
                &positions,
                &mut incoming[target + start..target + start + PERIODS],
                &mut incoming_arg[target + start..target + start + PERIODS],
            );
        }
    }
    let mut ticks = Vec::new();
    loop {
        let period = MIN_PERIOD + end_period;
        let e = if end >= period {
            emission(pulse, downbeat, end, period)
        } else {
            Emission::default()
        };
        ticks.push(Tick {
            frame: end,
            period_frames: period,
            bar_phase: end_phase,
            missing_component: end < period || e.missing,
            positive_pulse_window: end >= period && e.positive,
            pulse_contrast: e.contrast,
        });
        let previous = backs[end * width + end_phase * PERIODS + end_period];
        if previous == START {
            break;
        }
        end -= period;
        end_phase = (end_phase + meter - 1) % meter;
        end_period = usize::from(previous);
    }
    ticks.reverse();
    Component {
        start_frame: 0,
        end_frame: frames,
        meter_hypothesis_not_estimate: meter,
        score_not_confidence: best,
        ticks,
    }
}

#[allow(clippy::float_cmp)] // Only exactly constant heads lack all variation.
pub fn decode(input: &Frames) -> Result<Decoding> {
    let n = input.beat_logits.len();
    ensure!(
        n > 0 && n == input.downbeat_logits.len(),
        "empty or mismatched heads"
    );
    ensure!(
        input
            .beat_logits
            .iter()
            .chain(&input.downbeat_logits)
            .all(|v| v.is_finite()),
        "non-finite head"
    );
    ensure!(
        input.available.as_ref().is_none_or(|a| a.len() == n),
        "availability length mismatch"
    );
    let available = |i: usize| input.available.as_ref().is_none_or(|a| a[i]);
    let mut result = Decoding {
        components: vec![],
        unavailable_frames: 0,
        uninformative_frames: 0,
        max_backpointer_bytes: 0,
        period_limits_frames: [MIN_PERIOD, MAX_PERIOD],
        meter_hypotheses: [2, 3, 4, 5, 6, 7],
    };
    let mut start = 0;
    while start < n {
        if !available(start) {
            result.unavailable_frames += 1;
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < n && available(end) {
            end += 1;
        }
        let pulse = &input.beat_logits[start..end];
        let downbeat = &input.downbeat_logits[start..end];
        if pulse.len() < MIN_PERIOD * 2
            || (pulse.iter().all(|v| *v == pulse[0]) && downbeat.iter().all(|v| *v == downbeat[0]))
        {
            result.uninformative_frames += end - start;
        } else {
            let mut best: Option<Component> = None;
            for meter in 2..=7 {
                let candidate = clock(pulse, downbeat, meter);
                if best
                    .as_ref()
                    .is_none_or(|b| candidate.score_not_confidence > b.score_not_confidence)
                {
                    best = Some(candidate);
                }
            }
            let mut best = best.expect("six meter hypotheses");
            best.start_frame = start;
            best.end_frame = end;
            for tick in &mut best.ticks {
                tick.frame += start;
            }
            result.max_backpointer_bytes = result
                .max_backpointer_bytes
                .max((end - start) * PERIODS * 7);
            result.components.push(best);
        }
        start = end;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_transform_matches_exhaustive_predecessors() {
        let x: Vec<f64> = (10..=75).map(|p| real(p).log2()).collect();
        for seed in 0..20 {
            let scores: Vec<f64> = (0..x.len())
                .map(|i| {
                    if (i + seed) % 11 == 0 {
                        f64::NEG_INFINITY
                    } else {
                        real((i * 17 + seed * 13) % 31) - 15.0
                    }
                })
                .collect();
            let mut values = vec![0.0; x.len()];
            let mut args = vec![0; x.len()];
            transitions(&scores, &x, &mut values, &mut args);
            for i in 0..x.len() {
                let (mut best, mut arg) = (f64::NEG_INFINITY, 0);
                for j in 0..x.len() {
                    let v = scores[j] - 100.0_f64.ln() * (x[i] - x[j]).abs();
                    if v > best {
                        (best, arg) = (v, j);
                    }
                }
                assert!((values[i] - best).abs() < 1e-12);
                assert_eq!(usize::from(args[i]), arg);
            }
        }
    }

    fn authored() -> Frames {
        let mut b = vec![-8.0; 300];
        let mut d = b.clone();
        for i in (0..300).step_by(25) {
            b[i] = 8.0;
        }
        for i in (0..300).step_by(100) {
            d[i] = 8.0;
        }
        Frames {
            beat_logits: b,
            downbeat_logits: d,
            available: None,
        }
    }

    #[test]
    fn flat_input_is_not_a_confident_clock() {
        let mut input = authored();
        input.beat_logits.fill(-8.0);
        input.downbeat_logits.fill(-8.0);
        let result = decode(&input).unwrap();
        assert!(result.components.is_empty());
        assert_eq!(result.uninformative_frames, 300);
    }

    #[test]
    fn explicit_absence_splits_components_without_bridging() {
        let mut input = authored();
        let mut available = vec![true; 300];
        available[125..175].fill(false);
        input.available = Some(available);
        let result = decode(&input).unwrap();
        assert_eq!(result.unavailable_frames, 50);
        assert_eq!(result.components.len(), 2);
        assert!(result.components[0].ticks.iter().all(|t| t.frame < 125));
        assert!(result.components[1].ticks.iter().all(|t| t.frame >= 175));
    }

    #[test]
    fn bad_heads_and_hidden_label_fields_fail_closed() {
        let mut input = authored();
        input.beat_logits[0] = f32::NAN;
        assert!(decode(&input).is_err());
        assert!(
            serde_json::from_str::<Frames>(
                r#"{"beat_logits":[0],"downbeat_logits":[0],"truth":[0]}"#
            )
            .is_err()
        );
        let mut input = authored();
        input.available = Some(vec![true]);
        assert!(decode(&input).is_err());
    }

    #[test]
    fn deterministic_and_read_only() {
        let input = authored();
        let before = input.beat_logits.clone();
        let first = decode(&input).unwrap();
        let second = decode(&input).unwrap();
        assert_eq!(first.components, second.components);
        assert_eq!(input.beat_logits, before);
        for component in first.components {
            for ticks in component.ticks.windows(2) {
                assert_eq!(ticks[1].frame - ticks[0].frame, ticks[1].period_frames);
                assert_eq!(
                    ticks[1].bar_phase,
                    (ticks[0].bar_phase + 1) % component.meter_hypothesis_not_estimate
                );
            }
        }
    }
}
