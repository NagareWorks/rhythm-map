//! Time-exposure duration prior for an evaluation-only controlled intervention.

pub struct Prior {
    pub coordinates: Vec<f64>,
    pub log_survival: Vec<f64>,
    pub log_jump_base: Vec<f64>,
    pub rate_per_frame: f64,
}

impl Prior {
    /// Preserve the old prior's uniform-domain mean log-survival cost per frame.
    /// No tempo from a fixture, recording, truth path, or parameter sweep enters.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(periods: &[usize]) -> Self {
        assert!(!periods.is_empty() && periods.iter().all(|&p| p > 0));
        let coordinates: Vec<f64> = periods
            .iter()
            .map(|&p| 100_f64.ln() * (p as f64).log2())
            .collect();
        let off_diagonal: Vec<f64> = coordinates
            .iter()
            .enumerate()
            .map(|(i, x)| {
                coordinates
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| i != j)
                    .map(|(_, y)| (-(x - y).abs()).exp())
                    .sum()
            })
            .collect();
        let rate_per_frame = off_diagonal
            .iter()
            .zip(periods)
            .map(|(mass, &p)| mass.ln_1p() / p as f64)
            .sum::<f64>()
            / periods.len() as f64;
        let log_survival: Vec<f64> = periods
            .iter()
            .map(|&p| -rate_per_frame * p as f64)
            .collect();
        let log_jump_base = log_survival
            .iter()
            .zip(off_diagonal)
            .map(|(&survival, mass)| {
                if mass == 0. {
                    f64::NEG_INFINITY
                } else {
                    (-survival.exp_m1()).ln() - mass.ln()
                }
            })
            .collect();
        Self {
            coordinates,
            log_survival,
            log_jump_base,
            rate_per_frame,
        }
    }

    #[cfg(test)]
    pub fn transition(&self, from: usize, to: usize) -> f64 {
        if from == to {
            self.log_survival[from]
        } else {
            self.log_jump_base[from] - (self.coordinates[from] - self.coordinates[to]).abs()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_rows_are_normalized_without_self_jump_double_counting() {
        for periods in [vec![24], vec![10, 24, 48, 75], (10..=75).collect()] {
            let prior = Prior::new(&periods);
            for from in 0..periods.len() {
                let mass: f64 = (0..periods.len())
                    .map(|to| prior.transition(from, to).exp())
                    .sum();
                assert!((mass - 1.).abs() < 1e-12);
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn duration_survival_is_partition_invariant_including_the_terminal_cell() {
        let periods: Vec<usize> = (10..=75).collect();
        let prior = Prior::new(&periods);
        for duration in [12, 24, 48] {
            let cells = 1152 / duration;
            let complete = cells as f64 * prior.log_survival[duration - 10];
            assert!((complete + 1152. * prior.rate_per_frame).abs() < 1e-12);
            let uncensored = (cells - 1) as f64 * prior.log_survival[duration - 10];
            assert!((uncensored - complete).abs() > 1e-3);
        }
    }

    #[test]
    fn changing_time_units_preserves_probabilities_for_the_same_hypotheses() {
        let first = Prior::new(&[10, 24, 48, 75]);
        let doubled = Prior::new(&[20, 48, 96, 150]);
        assert!((first.rate_per_frame - 2. * doubled.rate_per_frame).abs() < 1e-12);
        for from in 0..4 {
            for to in 0..4 {
                assert!((first.transition(from, to) - doubled.transition(from, to)).abs() < 1e-12);
            }
        }
    }
}
