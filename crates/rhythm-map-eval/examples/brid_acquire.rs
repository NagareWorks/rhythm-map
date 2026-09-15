//! Source-only BRID acquisition; reuses the audited ZIP fetcher, never a model.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rhythm_map_eval::{
    PublicDatasetAssetRole, PublicDatasetLock, PublicDatasetZipAssetSelection,
    PublicDatasetZipMember, acquire_public_zip_assets,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf};

const INVENTORY: &[u8] =
    include_bytes!("../../../evaluation/datasets/brid-training-candidate-v1.json");
const INVENTORY_SHA: &str = "5532c49ae45af376803bd2a7529966e26e6a934765c7968ef9d3443320332157";
const SMOKE: [&str; 4] = ["0001", "0003", "0010", "0028"];
const SMOKE_REPORT: &[u8] = include_bytes!("../../../evaluation/datasets/brid-audio-smoke-v1.json");
const SMOKE_REPORT_SHA: &str = "96d9a70f055cfa9573eaab315b80b2bc0a8f14fa54556809912b2ec35aedbe19";
const SOURCE: &str = "https://zenodo.org/records/14051323";
const AUDIO_URL: &str = "https://zenodo.org/api/records/14051323/files/audio.zip/content";
const ANNOTATION_URL: &str =
    "https://zenodo.org/api/records/14051323/files/annotations.zip/content";

#[derive(Parser)]
struct Args {
    /// Fresh external directory; existing data is never overwritten.
    #[arg(long)]
    output: PathBuf,
    /// Fetch all 93 inventory mixtures after the frozen four-style structural smoke.
    /// This is source acquisition only, never training admission.
    #[arg(long)]
    all_mixtures: bool,
}

#[derive(Deserialize)]
struct Inventory {
    records: Vec<Record>,
}

#[derive(Deserialize)]
struct Record {
    recording_id: String,
    audio_candidate: Audio,
    annotation_assets: Vec<Annotation>,
}

#[derive(Deserialize)]
struct Audio {
    member_path: String,
    size_bytes: u64,
}

#[derive(Deserialize)]
struct Annotation {
    member_path: String,
    size_bytes: u64,
    sha256: String,
}

struct AcquisitionPlan {
    members: Vec<PublicDatasetZipAssetSelection>,
    expected: Vec<(u64, Option<String>)>,
}

fn selection(all_mixtures: bool) -> Result<AcquisitionPlan> {
    ensure!(
        format!("{:x}", Sha256::digest(INVENTORY)) == INVENTORY_SHA,
        "inventory drift"
    );
    let inventory: Inventory = serde_json::from_slice(INVENTORY)?;
    if all_mixtures {
        ensure!(
            format!("{:x}", Sha256::digest(SMOKE_REPORT)) == SMOKE_REPORT_SHA,
            "smoke evidence drift"
        );
        let smoke: serde_json::Value = serde_json::from_slice(SMOKE_REPORT)?;
        ensure!(
            smoke["structural_gate_passed"] == true
                && smoke["admitted_training_recordings"] == 0
                && smoke["recording_ids"] == serde_json::json!(SMOKE),
            "full acquisition requires the frozen structural smoke, not fit admission"
        );
    }
    ensure!(inventory.records.len() == 93, "mixture inventory drift");
    let ids: Vec<&str> = if all_mixtures {
        inventory
            .records
            .iter()
            .map(|row| row.recording_id.as_str())
            .collect()
    } else {
        SMOKE.to_vec()
    };
    ensure!(
        ids.windows(2).all(|pair| pair[0] < pair[1]),
        "IDs must be unique and sorted"
    );
    let mut selections = Vec::new();
    let mut expected = Vec::new();
    for id in &ids {
        let record = inventory
            .records
            .iter()
            .find(|row| row.recording_id == *id)
            .context("missing preregistered smoke recording")?;
        selections.push(PublicDatasetZipAssetSelection {
            path: format!("audio/{id}.wav"),
            url: AUDIO_URL.into(),
            role: PublicDatasetAssetRole::Audio,
            zip_member: PublicDatasetZipMember {
                archive_size_bytes: 944_409_073,
                member_path: record.audio_candidate.member_path.clone(),
            },
        });
        expected.push((record.audio_candidate.size_bytes, None));
        for annotation in &record.annotation_assets {
            let extension = annotation
                .member_path
                .rsplit('.')
                .next()
                .context("missing extension")?;
            ensure!(
                matches!(extension, "beats" | "bpm"),
                "unexpected annotation"
            );
            selections.push(PublicDatasetZipAssetSelection {
                path: format!("annotations/{id}.{extension}"),
                url: ANNOTATION_URL.into(),
                role: PublicDatasetAssetRole::AnnotationSource,
                zip_member: PublicDatasetZipMember {
                    archive_size_bytes: 84_989,
                    member_path: annotation.member_path.clone(),
                },
            });
            expected.push((annotation.size_bytes, Some(annotation.sha256.clone())));
        }
    }
    ensure!(
        selections.len() == ids.len() * 3,
        "every mixture must contain exactly three assets"
    );
    Ok(AcquisitionPlan {
        members: selections,
        expected,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let plan = selection(args.all_mixtures)?;
    fs::create_dir(&args.output).context("output must be a fresh external directory")?;
    let assets = acquire_public_zip_assets(&plan.members, &args.output)?;
    ensure!(
        assets.len() == plan.expected.len(),
        "incomplete acquisition"
    );
    for (asset, (size, hash)) in assets.iter().zip(plan.expected) {
        ensure!(
            asset.size_bytes == size,
            "source directory size drift: {}",
            asset.path
        );
        if let Some(hash) = hash {
            ensure!(
                asset.sha256 == hash,
                "annotation bytes drift: {}",
                asset.path
            );
        }
    }
    let lock = PublicDatasetLock {
        schema_version: 1,
        id: if args.all_mixtures { "brid-training-mixtures-v1" } else { "brid-training-smoke-v1" }.into(),
        version: if args.all_mixtures { "14051323-all-acoustic-mixtures" } else { "14051323-four-style-smoke" }.into(),
        homepage_url: SOURCE.into(),
        license: "CC-BY-4.0".into(),
        attribution: "Brazilian Rhythmic Instruments Dataset (BRID), Lucas S. Maia, Pedro D. de Tomaz Junior, Magdalena Fuentes, Martin Rocamora, Luiz W. P. Biscainho, Mauricio V. M. da Costa and Sara Cohen, 2018. Unmodified source audio and annotations; training candidates, not independent evaluation.".into(),
        assets,
    };
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.output.join("lock.json"))?;
    serde_json::to_writer_pretty(&mut file, &lock)?;
    writeln!(file)?;
    file.sync_all()?;
    eprintln!("Source files acquired; no training admission or model inference.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_smoke_ids_and_upstream_annotation_hashes() {
        let AcquisitionPlan {
            members: rows,
            expected,
        } = selection(false).unwrap();
        assert_eq!(rows.len(), 12);
        for (index, id) in SMOKE.iter().enumerate() {
            assert_eq!(rows[index * 3].path, format!("audio/{id}.wav"));
            assert_eq!(rows[index * 3 + 1].path, format!("annotations/{id}.beats"));
            assert_eq!(rows[index * 3 + 2].path, format!("annotations/{id}.bpm"));
            assert!(expected[index * 3].1.is_none());
            assert_eq!(expected[index * 3 + 1].1.as_ref().unwrap().len(), 64);
            assert_eq!(expected[index * 3 + 2].1.as_ref().unwrap().len(), 64);
        }
        assert_eq!(rows[0].zip_member.archive_size_bytes, 944_409_073);
        assert_eq!(rows[1].zip_member.archive_size_bytes, 84_989);
    }

    #[test]
    fn full_pool_requires_frozen_structural_smoke_and_preserves_all_mixtures() {
        let plan = selection(true).unwrap();
        assert_eq!(plan.members.len(), 279);
        assert_eq!(plan.expected.len(), 279);
        assert_eq!(
            plan.members
                .iter()
                .filter(|row| row.path.ends_with(".wav"))
                .count(),
            93
        );
        assert_eq!(
            plan.expected
                .iter()
                .step_by(3)
                .map(|(size, _)| size)
                .sum::<u64>(),
            469_990_540
        );
        for id in SMOKE {
            assert!(
                plan.members
                    .iter()
                    .any(|row| row.path == format!("audio/{id}.wav"))
            );
        }
    }
}
