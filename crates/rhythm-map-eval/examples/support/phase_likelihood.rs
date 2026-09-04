//! Rotation-normalized contextual evidence; evaluation only, not beat confidence.
use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Clone, Copy)]
pub struct Cell {
    pub start: usize,
    pub end: usize,
    pub phase: usize,
}

#[derive(Debug, Serialize)]
pub struct CellEvidence {
    pub log_ratio_to_null: f64,
    pub varies: bool,
}

/// A cyclic [1,2,1]/4 pulse statistic is normalized against every cyclic phase
/// in the SAME cell. `exp(log_ratio)` averages to one over all rotations, giving
/// a likelihood ratio against a rotation-invariant reference distribution.
/// This reference assumption need not hold for real neural frame backgrounds.
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
pub fn cell(values: &[f32], phase: usize) -> Result<CellEvidence> {
    let n = values.len();
    ensure!(n >= 3 && phase < n, "invalid cell or phase");
    ensure!(values.iter().all(|v| v.is_finite()), "non-finite evidence");
    let varies = values.iter().any(|v| *v != values[0]);
    if !varies {
        return Ok(CellEvidence {
            log_ratio_to_null: 0.0,
            varies: false,
        });
    }
    let stats: Vec<f64> = (0..n)
        .map(|i| {
            f64::from(values[(i + n - 1) % n]) * 0.25
                + f64::from(values[i]) * 0.5
                + f64::from(values[(i + 1) % n]) * 0.25
        })
        .collect();
    let maximum = stats.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_mean_exp = (stats.iter().map(|s| (s - maximum).exp()).sum::<f64>() / n as f64).ln();
    Ok(CellEvidence {
        log_ratio_to_null: stats[phase] - maximum - log_mean_exp,
        varies,
    })
}

#[derive(Debug, Serialize)]
pub struct PathEvidence {
    pub log_ratio_to_null: f64,
    pub scored_frames: usize,
    pub unavailable_frames: usize,
    pub available_frames_in_unscored_cells: usize,
    pub neutral_flat_cells: usize,
    pub scored_cells: usize,
}

/// Cells must tile the whole frame domain without overlap. Any unavailable
/// frame makes its WHOLE cell unscored, since its phase normalizer is unknown.
/// Report that coverage loss; comparing differently covered paths is invalid.
pub fn score(values: &[f32], cells: &[Cell], available: Option<&[bool]>) -> Result<PathEvidence> {
    ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "invalid evidence"
    );
    ensure!(
        available.is_none_or(|a| a.len() == values.len()),
        "availability length mismatch"
    );
    let mut result = PathEvidence {
        log_ratio_to_null: 0.0,
        scored_frames: 0,
        unavailable_frames: 0,
        available_frames_in_unscored_cells: 0,
        neutral_flat_cells: 0,
        scored_cells: 0,
    };
    let mut cursor = 0;
    for c in cells {
        ensure!(
            c.start == cursor
                && c.end <= values.len()
                && c.end >= c.start + 3
                && c.phase < c.end - c.start,
            "cells must tile the full domain with valid phases"
        );
        cursor = c.end;
        let missing = available.map_or(0, |a| a[c.start..c.end].iter().filter(|&&v| !v).count());
        if missing > 0 {
            result.unavailable_frames += missing;
            result.available_frames_in_unscored_cells += c.end - c.start - missing;
            continue;
        }
        let evidence = cell(&values[c.start..c.end], c.phase)?;
        result.log_ratio_to_null += evidence.log_ratio_to_null;
        result.scored_frames += c.end - c.start;
        result.scored_cells += 1;
        result.neutral_flat_cells += usize::from(!evidence.varies);
    }
    ensure!(cursor == values.len(), "incomplete frame domain");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_average_is_one_and_phase_rotates_with_data() {
        let x = [-8., -7., -2., -4., -8., -6.];
        let sum: f64 = (0..x.len())
            .map(|p| cell(&x, p).unwrap().log_ratio_to_null.exp())
            .sum();
        assert!((sum / 6.0 - 1.0).abs() < 1e-12);
        let mut rotated = x;
        rotated.rotate_right(2);
        for p in 0..x.len() {
            assert!(
                (cell(&x, p).unwrap().log_ratio_to_null
                    - cell(&rotated, (p + 2) % x.len()).unwrap().log_ratio_to_null)
                    .abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn weak_pulse_supports_phase_without_absolute_positive_logits() {
        let mut x = vec![-8.; 24];
        x[3..=5].fill(-2.);
        assert!(cell(&x, 4).unwrap().log_ratio_to_null > 0.0);
        assert!(cell(&x, 16).unwrap().log_ratio_to_null < 0.0);
        let shifted: Vec<f32> = x.iter().map(|v| v + 16.).collect();
        assert!(
            (cell(&x, 4).unwrap().log_ratio_to_null - cell(&shifted, 4).unwrap().log_ratio_to_null)
                .abs()
                < 1e-12
        );
        for value in [-8., 0., 8.] {
            let evidence = cell(&[value; 24], 4).unwrap();
            assert!(!evidence.varies);
            assert!(evidence.log_ratio_to_null.abs() < 1e-12);
        }
    }

    #[test]
    fn coverage_is_not_fabricated_when_normalizer_has_missing_frames() {
        let x = vec![-8.; 24];
        let cells = [
            Cell {
                start: 0,
                end: 12,
                phase: 4,
            },
            Cell {
                start: 12,
                end: 24,
                phase: 4,
            },
        ];
        let mut available = vec![true; 24];
        available[6] = false;
        let result = score(&x, &cells, Some(&available)).unwrap();
        assert_eq!(
            (
                result.scored_frames,
                result.unavailable_frames,
                result.available_frames_in_unscored_cells
            ),
            (12, 1, 11)
        );
        assert_eq!(result.scored_cells, 1);
    }

    #[test]
    fn overlapping_truncated_or_invalid_cells_fail_closed() {
        let c = Cell {
            start: 0,
            end: 12,
            phase: 4,
        };
        assert!(score(&[-8.; 24], &[c, c], None).is_err());
        assert!(score(&[-8.; 24], &[c], None).is_err());
        assert!(cell(&[f32::NAN; 12], 4).is_err());
        assert!(cell(&[-8.; 12], 12).is_err());
        assert!(score(&[-8.; 12], &[c], Some(&[])).is_err());
    }
}
