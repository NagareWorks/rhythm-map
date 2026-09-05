# Common-context conditional meter evidence, v1

The [censored-meter gate](censored-meter-v1.md) exposed an exact emission tie:
an extra predicted downbeat in a flat beat cell cost nothing. This intervention
retains its initial meter/phase prior, integrated meter-change prior, censored
bar boundaries and frozen inferred clock, but replaces its observation factor.
It passes the defined crop/change structural gate. It is **not a production
decoder**, a tempo fix, or a new selectable strategy.

## One observed domain, with count-dependent normalization

For each already inferred beat cell take the same raw cyclic pulse statistic
from the downbeat head, without the former per-cell normalization:

```
x[t] = head[cell_end-1]/4 + head[cell_start]/2 + head[cell_start+1]/4
```

All `N` scores in an available run form one comparison domain. For a proposed
meter path let `z[t]` indicate a downbeat and `K=sum(z)`. Define:

```
E[K] = sum over all subsets A of size K: exp(sum_{t in A} x[t])
Z[K] = E[K] / choose(N,K)
R(z,x) = exp(sum_t z[t]*x[t]) / Z[K]
```

`Z[K]` is the mean numerator under every assignment of the **same observed
score bag** to K selected locations. For each fixed path its ratio averages to
one under uniform permutations. Thus mixing these ratios with a normalized,
data-independent meter-path prior is also normalized under this conditional
exchangeable-score reference. K is latent and summed over, never supplied from
truth. Each path receives its count normalizer once, not once per frame or bar.
This is neither an independent-cell likelihood product nor a post-hoc penalty
chosen to favor four beats. No coefficient/threshold sweep or fitted parameters
are introduced; the raw statistic's scale remains an explicit model assumption.

For the former four-beat counterexample the raw scores repeat `[-2,-8,-8,-8]`.
The two-beat interpretation now selects many background-valued locations. Its
48-beat path has K=24 rather than K=12, and its corresponding common-context
normalizer makes its evidence lower. Merely summing raw scores would also
penalize weak negative-valued *real* marks; the count normalizer is essential.
Adding a constant to all scores leaves every path ratio unchanged. Fully flat
scores give ratio one for all paths and no new rhythmic evidence.

`E[K]` is computed by elementary-symmetric polynomial recurrence in log space:
starting with `E[0]=1`, multiply by `(1+exp(x[t])*u)` for each observed score.
Subtract the maximum score first for numerical stability; the offset cancels
between numerator and denominator at each K. Invalid/non-finite evidence or
overflow is rejected. No clipping of posterior probabilities is used.

## Counted inference and censored boundaries

The forward/backward state is `(beat index, meter, phase, visible downbeat
count)`. Deterministic within-bar progression and boundary transitions match
the previous audit: meters 2..7, initial `1/(6*m)`, and one run-wide unknown
change probability `rho ~ Beta(1,1)`. Changes choose uniformly among the other
meters. Exact-degree quadrature integrates rho as before. Terminal states of
every phase are accepted; the terminal observation factor is `1/Z[K]`.
Results include meter/downbeat marginals, the count distribution and posterior
mean rho. Marginal argmax labels are not a jointly legal MAP path.

Only supplied, observed beat cells contribute scores and K. A first partial
bar may start before the first cell; a final bar need not close. Missing edge
beats are not padded with observed silence. The frozen clock's excluded edge
frames stay excluded, and unavailable gaps still split runs. This conditional
model **does not integrate unobserved neural frame values from a full-audio
generative model**. Conditioning on a different observed score bag changes the
reference: crop scores are not whole-recording marginal evidence, and adding
observed background is not the same as leaving future data unobserved.

For S=27 and Q=O(N), this short-control implementation takes O(N^3*S) time and
O(N^2*S) working storage, including count state. Its bounded controls are fast;
the current implementation is not claimed to scale to full songs. No beam,
pruned states, truth sections or hand-supplied meter choices are used.

## Frozen result and independent checks

The fifteen original beat/downbeat head pairs and all supplied clock traces,
periods, gaps and edge coverage remain identical to the previous audit. The
139 supplied-grid crops and four genuine meter changes also retain their raw
signals and truth. Only their observation representation/scoring changes.

| Structural diagnostic | Previous conditional model | Common-context model |
| --- | --- | --- |
| Every position's most probable meter correct in crop controls | 87/139 | 139/139 |
| Four-beat crops | 0/16 | 16/16 |
| Six-beat crops | 0/36 | 36/36 |
| True 4->3 meter change | 40/48 positions | 48/48 |
| True 3->4 meter change | 48/48 | 48/48 |
| True 4->2 and 2->4 changes | Each 24/48 | Each 48/48 |
| Six main controls excluding weak true doubling | Ambiguous final four, below 50% | Four at every supplied tick; final P(4) about 0.92..0.95 |
| Flat-middle control | Final four, P(4)=0.4786 | Four at every supplied tick, including the seven prior-only beat ticks |
| Weak true doubling | Incorrect 47-tick clock retained | Same incorrect clock; middle 16 ticks favor meter two, other 31 favor four |
| Flat heads | Conditional log ratio 0 | 0 |
| One fixed-seed noise draw | Conditional log ratio +0.5532 | Slightly negative; not a false-positive-rate estimate |

These are authored structural checks, not music accuracy or calibrated
confidence. The improved numeric probabilities depend on this reference and
prior. In particular, the beat clock was already selected from the same neural
heads, so these post-selection conditional ratios must not be combined with
the former clock evidence or compared across clocks having different sampled
score bags. Neural backgrounds need not be exchangeable, especially across
sections, varying beat-cell lengths or production textures.

Run `cargo run --locked --profile evaluation -p rhythm-map-eval --example
common_meter` for `evaluation/parity/common-meter-v1.json`. It identifies the
decoder, runner and previous frozen reports by SHA-256. Rust tests exhaustively
enumerate small state paths and score subsets with Beta-integrated change/stay
counts, independently of the recurrence and quadrature. They also check unit
reference mass, offsets, flat inputs and count/marginal consistency.
`python -m unittest discover -s evaluation/parity -p test_common_meter.py -v`
reconstructs raw inputs and every saved probability using normal-space
polynomial multiplication and count-augmented dense matrix operations.

## Next gate

Keep the common-context treatment of unsupported starts and censored meter
state as prerequisites. Next give competing beat/tempo interpretations a shared
observation domain and account for observation reuse exactly once, then test
weak real doubling against missing-beat constant tempo. The current conditional
result cannot decide between those clocks and must not be used to manufacture
new detector events. Integration must also address complexity and unavailable
frame semantics before any full-cohort replay or production promotion.
No production defaults, external API, user parameter, model weight, private
music, holdout, training or release is changed. The remaining timing failure
still does not establish that a neural model needs training.
