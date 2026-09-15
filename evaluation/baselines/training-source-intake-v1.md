# New training-source intake v1

2026-09-15. This starts source expansion after the rejected
[native-offset experiment](../../experiments/tempo_variation/README.md), not
another loss-weight sweep. No model, optimizer, product default or old split
changes. These are **training candidates**, not a newly opened test set.

## Priority: BRID acoustic mixtures

The [official BRID record](https://zenodo.org/records/14051323) and its
[record API](https://zenodo.org/api/records/14051323) identify an open CC BY 4.0
release. Preserve attribution, the license link and notices of modifications;
the project Apache license does not relicense the data. Attribution: Brazilian
Rhythmic Instruments Dataset (BRID), Lucas S. Maia, Pedro D. de Tomaz Junior,
Magdalena Fuentes, Martin Rocamora, Luiz W. P. Biscainho, Mauricio V. M. da Costa
and Sara Cohen, 2018. No NC clause or additional commercial-use prohibition was
found in the inspected release. This is a documented source review, not a
guarantee about every possible downstream use.

The [original paper, sections 1.2 and 2.2](https://www.colibri.udelar.edu.uy/jspui/bitstream/20.500.12008/43550/1/MTFRBCC18.pdf)
distinguishes metronome-guided solos from jointly recorded, non-metronomic
mixtures. Thus mixtures are not simply synthetic sums of the solo files. It also
describes automatic beat/downbeat annotation with manual correction for its
microtiming experiment; that statement alone does not certify the provenance
and accuracy of every beat in the later complete annotation archive. The source
emphasizes duple meter and second-beat accents: this is relevant new rhythmic
material, not evidence that our model already handles it.

The downloaded annotation ZIP and ranged audio directory yield the
[reproducible inventory](../datasets/brid-training-candidate-v1.json):

| Verified property | Value |
| --- | ---: |
| Audio filenames in ZIP directory | 367 |
| Paired `.beats` and `.bpm` files, matching mixture audio names | 93 |
| Solo filenames without released `.beats` files | 274 |
| Beat / downbeat rows | 4,342 / 2,205 |
| Sum of last-minus-first annotated beat spans | 2,527.192 seconds (42.12 min) |
| Mixtures with 2 / 3 / 4 instruments | 45 / 29 / 19 |
| Samba / partido alto / samba-enredo / marcha | 41 / 28 / 21 / 3 |
| Nonincreasing times or invalid positions | 0 |
| Repeated adjacent beat positions | 0 |
| Mixture compressed audio bytes, excluding ZIP headers | 389,308,232 |

The [description table](https://zenodo.org/records/14051323/files/BRID%20-%20Description.pdf)
lists 44:24 for mixture recording duration. That is not the 42.12-minute
annotated-span sum. We have not decoded audio to confirm duration, endpoint
alignment or whether local beat-interval variations reflect sustained tempo
movement, microtiming or annotation noise. Global `.bpm` labels and descriptive
`60 / median(inter-beat interval)` are retained separately; neither becomes a
framewise tempo target or a change-point label by assumption. Missing solo beat
files must not be filled using the nominal BPM and an invented starting phase.

The official Beat This annotation release
[v1.0 tree](https://github.com/CPJKU/beat_this_annotations/tree/v1.0), resolved tree
`c3c47fd37d3074d9f8119f18bbf460f909609f22`, contains GuitarSet/Groove MIDI but not
BRID. This is evidence of absence from that published list, **not proof of no
unpublished encoder exposure or no cross-source recording overlap**.

Use one conservative `brid-corpus-v1` leakage group at intake. The description
shows shared performers across mixtures, and one performer across solos and
mixtures. Ninety-three recordings do not imply ninety-three independent works.
All excerpts, alternate microphone views and later speed augmentations inherit
their parent identity. No random clip split, corpus-internal test claim or
movement of existing development/holdout works into fit is allowed. The old
holdout remains sealed. BRID supplies a new recording source distinct from the
current classical fit collection; it does not by itself establish a broad
work-disjoint acceptance set.

## Next intake step, before any fit

Acquire the first numeric mixture ID in each of the four annotated styles:
**0001, 0003, 0010, 0028**. This is a source-defined smoke slice, selected without
model outputs, not a favorable-score filter. Use the existing ranged ZIP
acquisition and `dataset-fetch` machinery, verify extracted sizes/SHA-256 and
PCM duration, and check beat endpoints and annotation alignment. Retain failures.
Then admit the full 93-mixture pool if the source/annotation review permits it;
do not train on only four conveniently passing examples.

Before fitting, freeze that pool's role and its audio/reference adapter, cache
features once, and register a matched old-fit versus expanded-fit comparison.
Keep architecture, initialization, compute budget and absolute-timing gates
fixed so any outcome can test the data-coverage hypothesis. New data is a reason
to test again, not permission to promote a rejected checkpoint.

## Two genuinely different follow-on training options

| Source | Verified usable scope | Boundary |
| --- | --- | --- |
| [GuitarSet v1.1.0](https://zenodo.org/records/3371780) | CC BY 4.0 audio and JAMS; 360 short excerpts, six players, multiple styles | Beat This exposed. Training-only for our downstream head, not independent encoder acceptance. Group comping/solo, backing material and repeated lead-sheet/progression families; inspect the official timing errata before import. |
| [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) | CC BY 4.0, 13.6 hours of aligned MIDI and electronic-drum audio, human performances | Beat This exposed. Useful fills/microtiming at metronome tempo, not natural changing-tempo truth. Keep session and repeated-pattern relations together. Start with the 3.11 MB MIDI archive, not the 4.76 GB audio archive. |

Encoder exposure is a reason to disallow **independent evaluation**, not an
automatic reason to discard commercially licensed **training** data for a new
head. The earlier evaluation-only exclusions remain correct; their role is not
silently rewritten. [Guitar-TECHS](https://guitar-techs.github.io/) is lower priority: its official description
lists five hours of techniques but only twelve musical excerpts (8:02), with
note/MIDI rather than established full-corpus beat truth. No amount of filenames,
genre tags or automatic BPM pseudolabels counts as new reviewed tempo truth.

## Reproduction and integrity

Audio payloads and third-party annotations stay outside Git. The current
inventory is **not a fetch lock**: audio SHA-256 fields remain null until payload
acquisition. A ZIP directory CRC is not a cryptographic content verification.
Only the 84,989-byte annotation archive, 100,833-byte description and 1 MiB audio
tail were acquired from BRID in this intake. Source metadata/tree snapshots were
also retained privately; no inference, fitting, or old holdout access occurred.

```sh
curl --fail --location 'https://zenodo.org/api/records/14051323/files/annotations.zip/content' -o brid-annotations.zip
curl --fail --location --range 943360497-944409072 --max-filesize 1048576 \
  'https://zenodo.org/api/records/14051323/files/audio.zip/content' -o brid-audio-tail.bin
python evaluation/parity/brid_training_inventory.py \
  --annotations brid-annotations.zip --audio-tail brid-audio-tail.bin \
  --check evaluation/datasets/brid-training-candidate-v1.json
python -m unittest discover -s evaluation/parity -p test_brid_training_inventory.py -v
```

The fetch must return the exact range (`206`, total size 944,409,073). The offline
reader checks frozen byte sizes and SHA-256 before parsing; annotation MD5 also
matches the publisher. It inventories all 186 annotation files, rejects unsafe
paths, duplicate IDs/members, missing pairs and malformed values, and retains
position discontinuities without silently repairing them. CI uses authored
fixtures, not network or corpus downloads.

Additional evidence identities:

- Official record API snapshot SHA-256:
  `fc5dfb4a91c074f6de900c6e87617485b85280400979c291c4c97bcb51c60aa1`.
- Description PDF SHA-256:
  `74c8cd5dab7881d4abef52fc7b14dfc905bb35291ec4d00bfa44daec88296ab5`;
  publisher MD5 `6aff81c939d00a6f9aa27aec92231902`.
- Beat This v1.0 top-level tree API snapshot SHA-256:
  `51ec1c31f2d7f903e66439bb2dfc5170a6bffbe591d2f4b67032e5bfacdacdb4`.
