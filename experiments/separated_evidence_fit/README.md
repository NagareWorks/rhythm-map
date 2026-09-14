# Four-arm musical evidence experiment

This is the registered runner for the [separated evidence component](../separated_evidence/README.md).
It tests whether separately trained tempo/phase CNNs transfer better than a
shared CNN, while retaining independently fitted zero-feature controls. It does
not train Beat This, open new music, return a coherent clock or change defaults.

| Arm | Training features | Trainable parameters | Native optimizer calls |
| --- | --- | ---: | ---: |
| shared-natural | Cached natural features | 24,003 | 100 |
| shared-zero | Same shapes, all zeros | 24,003 | 100 |
| separated-natural | Cached natural features | 47,907 | 100 tempo + 100 phase |
| separated-zero | Same shapes, all zeros | 47,907 | 100 tempo + 100 phase |

Each arm gets the same 20 epochs, complete recordings, work weights, seed and
original task targets. Neither architecture gets favorable crops or a different
data split. Removing shared updates also increases total training capacity;
this experiment cannot isolate that causal explanation by itself.

## Selection is equal, not secretly a fusion layer

Both arms pick tempo by development tempo loss and phase by development phase
loss, after complete epochs, with earliest ties. A shared network can therefore
contribute tempo from epoch A and phase from epoch B. This is explicitly an
independent evidence pair, not a single jointly trained checkpoint. Both exported
pairs contain 47,907 parameters and perform two trunk evaluations. Training and
export cost must not be conflated.

The measurements reuse the existing tempo median/P95, circular phase and
4-second count error implementations and support masks. Count error integrates
the tempo branch; it does not certify agreement with the phase branch. A small
phase-vector magnitude is unavailable phase, not confidence or a silence label.
Every old numerical gate still applies separately to each architecture, and
separation must additionally justify itself against the shared arm. Even all
numerical gates passing only supports a proposal for coherent-fusion validation.

The [protocol](PROTOCOL-v1.md) fixes every arm, budget, checkpoint rule, temporal
intervention, failure boundary and decision before a musical invocation. The
runner rejects modified source/runtime/initialization/population/budget before
input loading, and never resumes a failed run. All earlier experiments and their
source hashes remain untouched.

## Closed musical outcome, 2026-09-14

The once-only registered experiment has now completed all four fits and 400
evidence-pair evaluations. Both architectures pass the fitted-zero audio-signal
and temporal-phase-dependence gates, but neither passes the full musical gate.
Separation improves development phase and all four ARTBeaT aggregate errors
relative to the shared network; development median BPM error nevertheless rises
from 35.4250% to 35.4772%, with 3/5 recordings regressing. It therefore fails the
registered separation nonregression and breadth requirements. Do not promote
either pair or start fusion/another fit automatically.

See the [complete outcome and limitations](../../evaluation/baselines/separated-evidence-learning-v1.md),
[registered plan](plan-v1.json), [execution receipt](execution-v1.json),
[every-case results](results-v1.json), and [epoch/evaluation journal](journal-v1.jsonl).
The nested `checks/` tests recompute every aggregate, decision, native-update
schedule and checkpoint selection without music access or model inference.
Private prediction packets additionally reproduced all 400 measurements exactly.

## Runner implementation and authored verification

- `register.py`: byte-pinned prerequisites, input population, source closure,
  exact target runtime and matched fresh initialization, without musical access.
- `recorder.py`: native shared updates, unchanged isolated TaskTrainer updates,
  symmetric selection, atomic before/after/best/failure snapshots and receipts.
- `run.py`: fixed four-arm order, full-record/work-macro training and development,
  selection/export, receipt schedule audit, deadlines and no-partial-evaluation.
- `evaluation.py`: all ten evidence pairs per recording, unchanged metric
  denominators and original numerical gates, plus the shared/separated comparison.

Authored tests compare all four reduced-budget fits against standalone native
AdamW/TaskTrainer loops, on CPU and CUDA. They cover nonuniform work weights,
independent selections, complete development, returned versus unknown calls,
failure evidence, late export, diagnostic exclusion, temporal controls, invalid
phase/rate support, and refusal to treat independent fields as a unified clock.
These authored tests are not a musical result. The runner-only commit did not
perform fitting; the subsequent outcome above is separate, retained evidence.

```sh
python -m unittest discover -s experiments/separated_evidence_fit -p 'test_*.py' -v
python -m experiments.separated_evidence_fit.register --device cuda --output NEW_PRIVATE_PLAN
timeout --signal=TERM --kill-after=10s 7500s python3 -m experiments.separated_evidence_fit.run \
  --device cuda --inputs PINNED_PRIVATE_INPUTS --plan NEW_PRIVATE_PLAN \
  --expected-plan-sha256 REGISTERED_PLAN_HASH --output NEW_PRIVATE_RUN
```

CUDA needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process startup. Complete
the exact-source authored/private validation and review before materializing
the target plan. The commands above document a separately authorized, once-only
music invocation; neither unit tests nor registration silently start it.
