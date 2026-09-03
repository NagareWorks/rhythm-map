use anyhow::{Context, Result, ensure};
use rhythm_map_eval::{CaseInput, EvaluationSuite, SuitePurpose};

pub fn calibration_audio(
    suite: &EvaluationSuite,
    id: &str,
) -> Result<rhythm_map_eval::AudioReference> {
    suite.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        suite.purpose == SuitePurpose::Calibration,
        "parity traces require calibration; holdout/regression rejected"
    );
    let case = suite
        .cases
        .iter()
        .find(|case| case.id == id)
        .context("unknown case")?;
    ensure!(
        case.provenance.commercial_evaluation_allowed,
        "case does not permit commercial evaluation"
    );
    let CaseInput::External { audio, .. } = &case.input else {
        anyhow::bail!("this trace command requires an external calibration case");
    };
    Ok(audio.clone())
}
