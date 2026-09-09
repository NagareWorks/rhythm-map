# Rhythm-support development metadata inventory v1

## Result and boundary

The official FSD50K v1 metadata download succeeded after the earlier gateway
failures. Both the complete metadata ZIP and the small documentation ZIP match
their [published MD5 checksums](https://zenodo.org/records/4060432). New SHA-256
identities pin both archives and the two inspected members in the
[inventory summary](../datasets/rhythm-support-inventory-v1.json).
The earlier [publication-only worksheet](../datasets/rhythm-support-worksheet-v1.json)
and [failure record](rhythm-support-readiness-v1.md) remain unchanged historical
evidence; this result supersedes their acquisition status, not their null checks.

This completes a reproducible **metadata license screen**, not per-clip legal
clearance, provenance adjudication, independent-group admission or human labels.
No audio was downloaded or selected, no features/model were run, and no project
holdout was accessed. The upstream metadata ZIP necessarily contains eval
metadata bytes, but only `FSD50K.metadata/dev_clips_info_FSD50K.json` was opened;
eval, class examples, ratings and collection payloads were not inspected. From
the documentation ZIP, only `FSD50K.doc/LICENSE-DATASET` was opened.

## Recomputed inventory

| Development metadata license URL | Clip count | Initial screen |
| --- | ---: | --- |
| CC0 1.0 | 14,959 | Retain for review |
| CC BY 3.0 | 20,017 | Retain for review |
| CC BY-NC 3.0 | 4,616 | Exclude |
| Sampling+ 1.0 | 1,374 | Exclude |
| Total | 40,966 | 34,976 retained; 5,990 excluded |

Counts now come from verified development metadata, not just the publication.
The exact license URLs, rather than permissive prefix/family matching, define
this screen. An unrecognized URL or changed schema stops the tool for review.
The archive contains exactly the five expected fields per clip: title,
description, tags, license and uploader. It supplies neither physical duration
nor region-support truth, PCM identity or complete recording/work provenance.

There are 4,947 exact uploader strings across development metadata and 4,179
within the initial license screen. Case-folding gives 4,946 and 4,178,
respectively: even spelling normalization changes counts. Preserve the original
attribution rather than silently resolving identities. One exact uploader
string accounts for 3,080 screened clips; 1,986 screened uploader strings have
one clip. Neither a string count nor a singleton proves recording independence.
Creators, works, recordings, shared samples and derived assets still need
review; `independent_source_group_count` remains null, not 4,179 or zero.

The generated local JSONL has 34,976 preliminary rows, 39,781,602 bytes, SHA-256
`6c2c7c8e72115437e6265c684c7b11bbef598e3c6286139ab2518430c80568be`.
Each row retains the source ID/version/link, archive/member hashes, original
title/uploader attribution and clip license, plus explicit unreviewed rights,
provenance/encoder-overlap unknowns, an unassigned split and no windows or labels.
Uploader names remain hints, not asserted performer/creator identities. Tags
and descriptions are not copied into model-facing inputs or converted to truth.
Only the compact summary and generator are committed; upstream ZIPs and the
expanded third-party metadata inventory stay outside the checkout.

## Rights and admission are still separate

The verified `LICENSE-DATASET` identifies dataset-level **CC BY 4.0**. This is
not an NC license. It does not replace individual clip grants or resolve every
rights question. The publication separately asks commercial users to contact
the authors. This inventory records that note as unresolved review, not as a
new asserted commercial prohibition or automatic clearance; nobody was contacted.
Annotation rights and training, evaluation and redistribution permissions remain
separate unreviewed fields. The archive/member identity records the evidence,
not a legal opinion about every downstream use.

Attribution for the dataset: FSD50K by Eduardo Fonseca, Xavier Favory, Jordi
Pons, Frederic Font and Xavier Serra, Music Technology Group, Universitat
Pompeu Fabra. The local JSONL is a filtered/restructured metadata derivative;
it does not redistribute recordings. Any later distribution still needs the
applicable attribution and rights review.

Zero rights-cleared clips, zero admitted groups and zero collected region
labels describe this project's completed admission work, not the absence of
potentially usable material. All eight label-coverage slices in the original
worksheet remain unfilled. The source's sound-event tags cannot establish
whether a listener follows a pulse through a declared query/context region.
No window/context policy has been populated or frozen by this metadata screen.

## Reproduction and checks

Obtain `FSD50K.metadata.zip` and `FSD50K.doc.zip` from the official record into
an explicit external data directory. No audio or ground-truth archive is needed.
The offline tool has no network path, extraction operation or configurable
eval-member option. It verifies full-archive size, published MD5, new SHA-256,
unique member, bounded decoded length and member SHA-256 before JSON parsing.

```sh
python evaluation/parity/rhythm_support_inventory.py \
  --metadata /external-data/FSD50K.metadata.zip \
  --document /external-data/FSD50K.doc.zip \
  --check evaluation/datasets/rhythm-support-inventory-v1.json \
  --rows /external-data/rhythm-support-candidates-v1.jsonl
python -m unittest discover -s evaluation/parity -p 'test_rhythm_support_inventory.py' -v
```

`--rows` is optional and uses exclusive creation; existing files are never
overwritten. Rows use numeric clip-ID order, sorted JSON keys, ASCII escaping
and LF delimiters. The summary records the deterministic row stream's identity.
CI uses authored tiny metadata/ZIP fixtures, not downloaded source data. It
checks hashes/sizes, duplicate/symlink/missing members, no eval payload access,
duplicate JSON keys, schema/attribution/license drift, deterministic ordering,
and the distinctions between screening, rights, provenance, splits and labels.
The real archive-to-summary and JSONL-hash checks were also run locally; fixture
tests alone do not establish that real-data reconstruction.

Next review the unresolved dataset/annotation conditions and provenance on a
bounded candidate pool before any admission or listening pilot. Metadata alone
cannot certify encoder non-overlap, all eight coverage slices or label agreement.
The [training prerequisites](../../docs/RHYTHM-SUPPORT-PROPOSAL.md) remain: no
automatic audio acquisition, human-label coordination, fitting, production
changes or release. Acquisition recovery is progress on data readiness, not
new evidence that training is necessary or that variable-tempo accuracy improved.
