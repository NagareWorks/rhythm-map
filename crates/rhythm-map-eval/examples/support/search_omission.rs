//! Bounded exact search: unknown clock, changing meter, and integrated omissions.
//! This is a reference implementation on short feature arrays, not a decoder.
use super::frames::{Table, add};
#[path = "time_prior.rs"]
#[allow(dead_code)]
mod time_prior;
use anyhow::{Result, ensure};
use serde::Serialize;
use std::collections::BTreeMap;

const NEG: f64 = f64::NEG_INFINITY;

#[derive(Clone, Copy, Serialize)]
pub struct Domain {
    pub min_period: usize,
    pub max_period: usize,
    pub min_meter: usize,
    pub max_meter: usize,
    /// Safety budget, not a beam size. Exceeding it returns no inference.
    pub max_states: usize,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Counts {
    trials: usize,
    pulses: usize,
    bar_pulses: usize,
    accents: usize,
    meter_trials: usize,
    meter_changes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    period: usize,
    meter: usize,
    phase: usize,
    counts: Counts,
}

struct Node {
    frame: usize,
    key: Key,
    mass: f64,
    best: f64,
    parent: Option<(usize, Option<u8>)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Tick {
    pub frame: usize,
    pub period_frames: usize,
    pub meter: usize,
    pub beat_in_bar: usize,
    /// None: unavailable; 0/1/2: inferred omission/plain/accent. Never a detection.
    pub inferred_label: Option<u8>,
}

#[derive(Clone, Default, Serialize)]
pub struct Position {
    pub latent_tick_probability: f64,
    pub inferred_label_probabilities: [f64; 3],
    pub unavailable_tick_probability: f64,
    pub tempo_change_probability: f64,
    pub meter_change_probability: f64,
}

#[derive(Serialize)]
pub struct Inference {
    pub log_ratio: f64,
    pub joint_map_log_weight: f64,
    pub joint_map_probability: f64,
    pub inferred_ticks: Vec<Tick>,
    pub positions: Vec<Position>,
    pub states: usize,
    pub transitions: usize,
}

#[allow(clippy::cast_precision_loss)]
fn beta(success: usize, failure: usize) -> f64 {
    let factorial = |n: usize| (1..=n).map(|i| (i as f64).ln()).sum::<f64>();
    factorial(success) + factorial(failure) - factorial(success + failure + 1)
}

fn terminal(table: &Table, c: Counts) -> Result<f64> {
    Ok(beta(c.pulses, c.trials - c.pulses)
        + beta(c.accents, c.bar_pulses - c.accents)
        + beta(c.meter_changes, c.meter_trials - c.meter_changes)
        - table.normalizer(c.pulses - c.accents, c.accents)?)
}

fn labels(table: &Table, frame: usize, key: Key) -> Vec<(Option<u8>, Counts, f64)> {
    let Some(pair) = table.centered[frame] else {
        return vec![(None, key.counts, 0.)];
    };
    (0..=if key.phase == 0 { 2 } else { 1 })
        .map(|label| {
            let mut c = key.counts;
            c.trials += 1;
            c.pulses += usize::from(label > 0);
            c.bar_pulses += usize::from(label > 0 && key.phase == 0);
            c.accents += usize::from(label == 2);
            let score = if label == 0 {
                0.
            } else {
                pair[0] + if label == 2 { pair[1] } else { 0. }
            };
            (Some(label), c, score)
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn successors(
    key: Key,
    counts: Counts,
    domain: Domain,
    prior: &time_prior::Prior,
) -> Vec<(Key, f64)> {
    let wrapped = key.phase + 1 == key.meter;
    let meters = if wrapped {
        domain.min_meter..=domain.max_meter
    } else {
        key.meter..=key.meter
    };
    let mut out = Vec::new();
    for meter in meters {
        let mut c = counts;
        let changed = meter != key.meter;
        if wrapped && domain.min_meter != domain.max_meter {
            c.meter_trials += 1;
            c.meter_changes += usize::from(changed);
        }
        let meter_prior = if changed {
            -((domain.max_meter - domain.min_meter) as f64).ln()
        } else {
            0.
        };
        for period in domain.min_period..=domain.max_period {
            let from = key.period - domain.min_period;
            let to = period - domain.min_period;
            let transition = if from == to {
                prior.log_survival[from]
            } else {
                prior.log_jump_base[from] - (prior.coordinates[from] - prior.coordinates[to]).abs()
            };
            out.push((
                Key {
                    period,
                    meter,
                    phase: if wrapped { 0 } else { key.phase + 1 },
                    counts: c,
                },
                transition + meter_prior,
            ));
        }
    }
    out
}

fn tick(node: &Node, label: Option<u8>) -> Tick {
    Tick {
        frame: node.frame,
        period_frames: node.key.period,
        meter: node.key.meter,
        beat_in_bar: node.key.phase + 1,
        inferred_label: label,
    }
}

fn insert(
    nodes: &mut Vec<Node>,
    layers: &mut [BTreeMap<Key, usize>],
    node: Node,
    budget: usize,
) -> Result<()> {
    if let Some(&id) = layers[node.frame].get(&node.key) {
        nodes[id].mass = add(nodes[id].mass, node.mass);
        if node.best > nodes[id].best {
            nodes[id].best = node.best;
            nodes[id].parent = node.parent;
        }
    } else {
        ensure!(
            nodes.len() < budget,
            "exact search state budget exceeded; no partial inference returned"
        );
        layers[node.frame].insert(node.key, nodes.len());
        nodes.push(node);
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn infer(table: &Table, domain: Domain) -> Result<Inference> {
    let n = table.centered.len();
    ensure!(
        (2..=32).contains(&n),
        "reference supports 2..32 feature frames only"
    );
    ensure!(
        domain.min_period >= 2 && domain.min_period <= domain.max_period && domain.max_period <= n,
        "invalid period domain"
    );
    ensure!(
        domain.min_meter >= 2 && domain.min_meter <= domain.max_meter && domain.max_meter <= 7,
        "invalid meter domain"
    );
    let periods: Vec<_> = (domain.min_period..=domain.max_period).collect();
    let prior = time_prior::Prior::new(&periods);
    let mut layers = vec![BTreeMap::new(); n];
    let mut nodes = Vec::new();
    for &period in &periods {
        for meter in domain.min_meter..=domain.max_meter {
            let initial =
                -((periods.len() * period * (domain.max_meter - domain.min_meter + 1) * meter)
                    as f64)
                    .ln();
            for frame in 0..period {
                for phase in 0..meter {
                    insert(
                        &mut nodes,
                        &mut layers,
                        Node {
                            frame,
                            key: Key {
                                period,
                                meter,
                                phase,
                                counts: Counts::default(),
                            },
                            mass: initial,
                            best: initial,
                            parent: None,
                        },
                        domain.max_states,
                    )?;
                }
            }
        }
    }
    let mut partition = NEG;
    let mut best = NEG;
    let mut ending = (0, None);
    let mut transitions = 0;
    for frame in 0..n {
        let ids: Vec<_> = layers[frame].values().copied().collect();
        for id in ids {
            let key = nodes[id].key;
            let next_frame = frame + key.period;
            for (label, counts, emission) in labels(table, frame, key) {
                if next_frame >= n {
                    let weight = emission + terminal(table, counts)?;
                    partition = add(partition, nodes[id].mass + weight);
                    if nodes[id].best + weight > best {
                        best = nodes[id].best + weight;
                        ending = (id, label);
                    }
                    transitions += 1;
                } else {
                    for (next, transition) in successors(key, counts, domain, &prior) {
                        let node = Node {
                            frame: next_frame,
                            key: next,
                            mass: nodes[id].mass + emission + transition,
                            best: nodes[id].best + emission + transition,
                            parent: Some((id, label)),
                        };
                        insert(&mut nodes, &mut layers, node, domain.max_states)?;
                        transitions += 1;
                    }
                }
            }
        }
    }
    let mut backward = vec![NEG; nodes.len()];
    let mut positions = vec![Position::default(); n];
    for frame in (0..n).rev() {
        for &id in layers[frame].values() {
            let key = nodes[id].key;
            let next_frame = frame + key.period;
            for (label, counts, emission) in labels(table, frame, key) {
                let mut rest = NEG;
                if next_frame >= n {
                    rest = terminal(table, counts)?;
                } else {
                    for (next, transition) in successors(key, counts, domain, &prior) {
                        let next_id = layers[next_frame][&next];
                        let suffix = transition + backward[next_id];
                        rest = add(rest, suffix);
                        let probability = (nodes[id].mass + emission + suffix - partition).exp();
                        if next.period != key.period {
                            positions[next_frame].tempo_change_probability += probability;
                        }
                        if next.meter != key.meter {
                            positions[next_frame].meter_change_probability += probability;
                        }
                    }
                }
                backward[id] = add(backward[id], emission + rest);
                let probability = (nodes[id].mass + emission + rest - partition).exp();
                positions[frame].latent_tick_probability += probability;
                if let Some(label) = label {
                    positions[frame].inferred_label_probabilities[usize::from(label)] +=
                        probability;
                } else {
                    positions[frame].unavailable_tick_probability += probability;
                }
            }
        }
    }
    let mut path = Vec::new();
    let (mut id, mut label) = ending;
    loop {
        path.push(tick(&nodes[id], label));
        if let Some(parent) = nodes[id].parent {
            (id, label) = parent;
        } else {
            break;
        }
    }
    path.reverse();
    Ok(Inference {
        log_ratio: partition,
        joint_map_log_weight: best,
        joint_map_probability: (best - partition).exp(),
        inferred_ticks: path,
        positions,
        states: nodes.len(),
        transitions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain() -> Domain {
        Domain {
            min_period: 2,
            max_period: 3,
            min_meter: 2,
            max_meter: 3,
            max_states: 100_000,
        }
    }

    #[test]
    fn flat_and_unavailable_integrate_to_one_without_terminal_edge_penalties() {
        for values in [
            vec![Some([0., 0.]); 8],
            vec![None; 8],
            vec![
                Some([0., 0.]),
                None,
                None,
                Some([0., 0.]),
                None,
                None,
                Some([0., 0.]),
                None,
            ],
        ] {
            let count = values.iter().flatten().count();
            let result = infer(&Table::new(&values, count, count).unwrap(), domain()).unwrap();
            assert!(result.log_ratio.abs() < 1e-11);
            for (value, p) in values.iter().zip(result.positions) {
                let total = p.inferred_label_probabilities.iter().sum::<f64>()
                    + p.unavailable_tick_probability;
                assert!((p.latent_tick_probability - total).abs() < 1e-12);
                assert!((0. ..=1. + 1e-12).contains(&p.latent_tick_probability));
                if value.is_none() {
                    assert_eq!(p.inferred_label_probabilities, [0.; 3]);
                }
            }
        }
    }

    #[test]
    fn invalid_domains_and_state_exhaustion_return_errors_not_partial_paths() {
        let table = Table::new(&[Some([0., 0.]); 8], 8, 8).unwrap();
        for d in [
            Domain {
                min_period: 0,
                ..domain()
            },
            Domain {
                max_period: 9,
                ..domain()
            },
            Domain {
                max_meter: 8,
                ..domain()
            },
            Domain {
                min_meter: 4,
                ..domain()
            },
            Domain {
                max_states: 1,
                ..domain()
            },
        ] {
            assert!(infer(&table, d).is_err());
        }
    }

    #[test]
    fn singleton_and_full_meter_domains_preserve_prior_mass() {
        let table = Table::new(&[Some([0., 0.]); 8], 8, 8).unwrap();
        for d in [
            Domain {
                min_period: 3,
                max_period: 3,
                max_meter: 2,
                ..domain()
            },
            Domain {
                max_meter: 7,
                ..domain()
            },
        ] {
            let result = infer(&table, d).unwrap();
            assert!(result.log_ratio.abs() < 1e-11);
            if d.min_period == d.max_period {
                assert!(
                    result
                        .positions
                        .iter()
                        .all(|p| p.tempo_change_probability == 0.
                            && p.meter_change_probability == 0.)
                );
            }
        }
    }

    #[test]
    fn joint_search_keeps_the_paired_permutation_reference_normalized() {
        let values = [
            Some([0.2, 1.]),
            Some([-1., 0.7]),
            Some([1.4, -0.3]),
            Some([0.5, 1.5]),
        ];
        let mut total = 0.;
        let mut permutations = 0.;
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        if a == b || a == c || a == d || b == c || b == d || c == d {
                            continue;
                        }
                        let table = Table::new(&[values[a], values[b], values[c], values[d]], 4, 4)
                            .unwrap();
                        total += infer(&table, domain()).unwrap().log_ratio.exp();
                        permutations += 1.;
                    }
                }
            }
        }
        assert!((total / permutations - 1.).abs() < 1e-11);
    }
}
