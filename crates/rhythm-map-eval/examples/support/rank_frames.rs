//! Paired empirical midrank features; order evidence, not calibrated confidence.
use anyhow::{Result, ensure};

/// Fixed local location removal before the single global rank transform.
/// The odd window is strictly shorter than the minimum allowed beat period;
/// edges clip within available runs, with midpoint medians for even remnants.
pub fn remove_background(
    values: &[Option<[f64; 2]>],
    minimum_period: usize,
) -> Result<Vec<Option<[f64; 2]>>> {
    ensure!(
        !values.is_empty()
            && values.iter().flatten().flatten().all(|v| v.is_finite())
            && (3..=75).contains(&minimum_period),
        "invalid background domain"
    );
    let mut result = vec![None; values.len()];
    let radius = (minimum_period - 2) / 2;
    let mut start = 0;
    while start < values.len() {
        if values[start].is_none() {
            start += 1;
            continue;
        }
        let end = (start..values.len())
            .find(|&i| values[i].is_none())
            .unwrap_or(values.len());
        for t in start..end {
            let left = t.saturating_sub(radius).max(start);
            let right = (t + radius + 1).min(end);
            let mut pair = values[t].unwrap();
            for (h, value) in pair.iter_mut().enumerate() {
                let mut window: Vec<_> =
                    values[left..right].iter().map(|p| p.unwrap()[h]).collect();
                window.sort_by(f64::total_cmp);
                *value -= f64::midpoint(window[(window.len() - 1) / 2], window[window.len() / 2]);
            }
            ensure!(
                pair.iter().all(|v| v.is_finite()),
                "background contrast overflow"
            );
            result[t] = Some(pair);
        }
        start = end;
    }
    Ok(result)
}

/// Each head uses logit(mid-CDF) over the same available full-frame domain.
/// Equal values share a midrank. No epsilon, amplitude cutoff, clock-dependent
/// window or learned scale is introduced. All-missing input stays missing.
#[allow(clippy::cast_precision_loss, clippy::float_cmp)] // Exact ties define the statistic.
pub fn transform(values: &[Option<[f64; 2]>]) -> Result<Vec<Option<[f64; 2]>>> {
    ensure!(
        !values.is_empty() && values.iter().flatten().flatten().all(|v| v.is_finite()),
        "invalid rank features"
    );
    let mut result = values.to_vec();
    for h in 0..2 {
        let mut sorted: Vec<_> = values
            .iter()
            .enumerate()
            .filter_map(|(t, p)| p.map(|p| (p[h], t)))
            .collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut start = 0;
        while start < sorted.len() {
            let end = (start + 1..sorted.len())
                .find(|&i| sorted[i].0 != sorted[start].0)
                .unwrap_or(sorted.len());
            // L values below, E equal, G above: log((L+E/2)/(G+E/2)).
            let rank = ((start + end) as f64).ln() - ((2 * sorted.len() - start - end) as f64).ln();
            for &(_, t) in &sorted[start..end] {
                result[t].as_mut().unwrap()[h] = rank;
            }
            start = end;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_missing_and_flat_have_exact_semantics() {
        let out = transform(&[Some([1., 3.]), None, Some([1., 3.]), Some([5., 3.])]).unwrap();
        assert_eq!(out[1], None);
        assert_eq!(out[0], out[2]);
        assert!((out[0].unwrap()[0] + 2_f64.ln()).abs() < 1e-14);
        assert!((out[3].unwrap()[0] - 5_f64.ln()).abs() < 1e-14);
        assert!(out.iter().flatten().all(|p| p[1] == 0.));
        assert_eq!(transform(&[None; 4]).unwrap(), vec![None; 4]);
        assert!(transform(&[]).is_err());
        assert!(transform(&[Some([f64::NAN, 0.])]).is_err());
        assert!(transform(&[Some([0., f64::INFINITY])]).is_err());
    }

    #[test]
    fn order_not_amplitude_is_the_only_retained_information() {
        let values = [Some([-3., 2.]), Some([0., -2.]), None, Some([5., 2.])];
        let changed: Vec<_> = values
            .iter()
            .map(|p| p.map(|v| [v[0] * v[0] * v[0] + 100., v[1] / 4096.]))
            .collect();
        assert_eq!(transform(&values).unwrap(), transform(&changed).unwrap());
        let reversed: Vec<_> = values.iter().rev().copied().collect();
        let expected: Vec<_> = transform(&values).unwrap().into_iter().rev().collect();
        assert_eq!(transform(&reversed).unwrap(), expected);
    }

    #[test]
    fn background_windows_clip_at_missing_runs_and_even_edges() {
        let values = [
            Some([1., 10.]),
            Some([3., 10.]),
            None,
            Some([100., -8.]),
            Some([100., -8.]),
        ];
        assert_eq!(
            remove_background(&values, 10).unwrap(),
            vec![
                Some([-1., 0.]),
                Some([1., 0.]),
                None,
                Some([0., 0.]),
                Some([0., 0.])
            ]
        );
        assert_eq!(remove_background(&[None; 5], 10).unwrap(), vec![None; 5]);
        assert_eq!(
            remove_background(&values, 3).unwrap(),
            vec![
                Some([0., 0.]),
                Some([0., 0.]),
                None,
                Some([0., 0.]),
                Some([0., 0.])
            ]
        );
        assert!(remove_background(&values, 2).is_err());
        assert!(remove_background(&[Some([f64::NAN, 0.])], 10).is_err());
    }
}
