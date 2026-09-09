# Rhythm-support metadata admission review v1

## Outcome: a bounded review, not a training decision

This pass clarifies standard license terms, detects two already-exposed source
IDs, and reviews eight deterministically selected development metadata records.
It admits **zero new independent groups and zero region labels**. There is no
audio download/selection, listening, feature extraction, fitting, model/default
change or sealed-holdout access. The [previous inventory](rhythm-support-inventory-v1.md)
is unchanged historical evidence, including its 34,976 preliminary rows.

The [reproducible sample and exposure comparison](../datasets/rhythm-support-provenance-v1.json)
are separate from the [source/rights review ledger](../datasets/rhythm-support-admission-review-v1.json).
Automated reconstruction checks the former; it does not certify the latter's
legal interpretation, source freshness or human annotation reliability.

## Remove a false prerequisite, retain real obligations

Standard CC licenses do not need an additional clause naming AI training to
provide copyright permission for a new technology, provided their conditions
are respected. The [Creative Commons AI FAQ](https://creativecommons.org/faq/#artificial-intelligence-and-cc-licenses)
states this explicitly and distinguishes privacy and other non-copyright issues.
Do not require a special AI grant solely because these older licenses do not
mention neural networks. This is a review of standard terms, not jurisdiction-
specific legal advice or permission to start this project's training.

The [CC0 1.0 text](https://creativecommons.org/publicdomain/zero/1.0/legalcode)
includes commercial use in its waiver/fallback but does not clear other
people's rights or guarantee ownership. [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/legalcode)
grants reproduction/adaptation/distribution subject to its conditions, including
applicable attribution. The verified dataset-level grant remains
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode), which also
addresses database rights. None is an NC grant. Clip and dataset terms remain
separate; no upstream sound-event annotation labels were imported or cleared.

The [FSD50K publication](https://zenodo.org/records/4060432) separately asks
commercial users to contact its authors. Retain the note, but do not transform
it into an invented NC restriction or a universal AI-specific permission
requirement. No contact or separate agreement occurred. Concrete ownership,
sample/recording provenance, incidental third-party rights and redistribution
obligations still need review for admitted items. Open questions below are not
findings of infringement or proof that the source is unusable.

## Corpus names are not recording-level non-overlap

The [pinned Beat This README](https://github.com/CPJKU/beat_this/blob/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c/README.md)
describes `final0` as trained on all its data except GTZAN and distinguishes
other split-specific checkpoints. The pinned
[v1.0 annotation source README](https://github.com/CPJKU/beat_this_annotations/blob/c3c47fd37d3074d9f8119f18bbf460f909609f22/README.md)
and root listing name sixteen corpora. Neither FSD50K nor FSLD is named.
The ledger pins revisions, README SHA-256 identities and all sixteen names.
Only READMEs/root names were inspected: no annotation, GTZAN payload, training
spectrogram or new checkpoint was downloaded.

This supports **corpus-name absence only**, not a completed membership audit.
A Freesound clip may reuse a recording, composition or sample listed under
another name elsewhere. The eight records have no verified PCM identities,
complete source-asset genealogy or cross-corpus recording mapping. Keep encoder
recording overlap unknown; do not substitute an empty intersection of unrelated
ID namespaces or infer absence from a name search.

## A concrete project-exposure collision

Comparing the fifteen audio IDs in the existing, exposed FSLD calibration lock
with verified FSD50K development metadata finds **210835 and 418991**, both in
the CC0 screen. This is the same Freesound source-ID namespace, so the match is
meaningful as a known project exposure. It does not verify equal decoded PCM
after dataset preprocessing, nor imply Beat This trained on either clip.

These two source IDs must not become new independent acceptance cases merely
because they arrived through another dataset. The review sampler excludes all
fifteen known FSLD calibration IDs before choosing rows. The original calibration
locks, forty identities and all sealed holdouts remain untouched. Same-creator
or derived-asset relations beyond these exact ID matches remain unreviewed.

## Eight fixed metadata-review records

Before inspecting their descriptions, freeze this rule: process CC BY 3.0 then
CC0 1.0; rank IDs by SHA-256 of `rhythm-support-metadata-review-v1\n` followed
by the canonical decimal ID; break ties numerically; take four per license,
excluding known calibration IDs and case-folded uploader hints already selected
in either quota. Insufficient distinct hints fails instead of relaxing the rule.
No tags, titles, descriptions, model scores or expected labels select the sample.
This tiny quota sample is not a representative dataset survey or an audio pilot.

| Source ID | Historical metadata clue | Unresolved review question |
| --- | --- | --- |
| 61856 | Edited fragments of one scratching session | Source material and derived-fragment family |
| 355100 | Good-sounds instrument-note project, original ID 7982 | Project/session/performer provenance |
| 111125 | Refers to another similar recording | Identify the related recording before split assignment |
| 216126 | Keyboard tapping | Creator/recording provenance; a sound tag is not rhythm truth |
| 167533 | Ring-alert collection | Composition, sample assets and collection family |
| 340968 | Crowd at a parade | Recording provenance and possible incidental third-party content |
| 262983 | Audience recorded in a theatre | Session and possible performance/personality rights |
| 107272 | Transit-bus engine | Recording provenance; machinery is not automatically rhythm-negative |

The seven source-page retrievals other than 340968 were unavailable with
non-retryable tool errors; no bypass or replacement sample was attempted.
For [340968](https://freesound.org/people/SoundsAreGr8/sounds/340968/), a primary-page
snapshot agrees with the metadata and CC0 grant, but the crawler reports it as
seven months old. Record it as corroborating cached evidence, not current live
verification. None of the eight gets a cleared independent group or a rhythm
label from this review. Historical metadata claims are not verified authorship.

## Checks and stopping boundary

```sh
python evaluation/parity/rhythm_support_provenance.py \
  --metadata /external-data/FSD50K.metadata.zip \
  --document /external-data/FSD50K.doc.zip \
  --check evaluation/datasets/rhythm-support-provenance-v1.json
python -m unittest discover -s evaluation/parity -p 'test_rhythm_support_provenance.py' -v
```

Fourteen new authored/frozen-boundary tests cover deterministic quotas, semantic
independence of sampling, exposure exclusion, uploader-hint handling, malformed
source IDs, no license fallback, unknown accounting and stale evidence. The real
locked archives also reproduce the retained sample/exposure report locally.
Tests validate bookkeeping and boundaries, not the truth of a license claim,
absence of reused audio or musical accuracy.

Close this bounded metadata pass here. Do not keep enlarging an unlabeled
sound-effects inventory as a substitute for progress on timing. Region support
is still only a possible side component: it cannot distinguish weak/omitted
beats from true tempo changes when both regions remain rhythmic. The next
technical decision returns to that original competing-clock target and its
retained evidence; it is not automatic annotation, fitting or another presence
model search. Any future listening/training pilot still needs a distinct scope,
baseline, rights/independence plan, resource budget, stop rule and authorization.
