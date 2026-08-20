# FD²RT — Experimental Findings

Running record of what the controlled experiments actually showed, including
the hypotheses they killed. Kept in the repo so the paper is written from
evidence rather than from memory of what we hoped.

Raw data: `results/*.json` (force-added; `results/` is otherwise gitignored).
Reproduce: `python3 scripts/pilot_a4_vs_icnf.py --arms <arms> --seeds 4 --iters 1500`

---

## Summary of verdicts

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | The illumination-TV prior regularises the illumination | **REFUTED** — contributed exactly zero gradient for an entire 250K run |
| H2 | `FrequencyAwareLoss` weights low:high as configured (1:2) | **REFUTED** — acted as 1:38 due to band normalisation |
| H3 | W-IE's wavelet prior differs from the baseline's mean prior | **REFUTED** — identical to `2·boxblur(mean_c)` to 2.4e-7 |
| H4 | ICNF's advantage comes from the Poisson-Gaussian noise physics | **REFUTED** — survives breaking the noise law (+1.20 → +1.02 dB) |
| H5 | Conditioning the noise floor on illumination helps | **REFUTED** — *harmful*; uniform floor is +0.53 dB better, 4/4 seeds |
| H6 | Adaptive computation via evidence is a viable contribution | **REJECTED** — measured, did not pay for itself |
| H7 | A per-pixel gate beats A4's scalar gate | **SUPPORTED so far** — +1.73 dB, 4/4 seeds, +2.26 pooled sd |
| H8 | That gain is the gate, not merely the init fix | **SUPPORTED** — gate +1.53 dB (4/4); init alone +0.20 dB (2/4, null) |
| H9 | The gate's gain survives at the real architecture | **PARTIALLY SUPPORTED** — +1.37 dB holds, but 3/4 wins and effect size falls 2.26 → 0.92 SD |
| H10 | The gain is spatial adaptivity, not a better constant gate level | **SUPPORTED** — spatial +1.63 dB (4/4); level alone −0.25 dB |
| H11 | A per-channel gate improves on the shared gate | **REFUTED** — pooled over 8 seeds: +0.20 dB at **4/8 wins**, i.e. chance |
| H12 | The spatial-gate claim survives replication (not just 4 lucky seeds) | **SUPPORTED** — pooled 8 seeds: **+1.28 dB, 7/8 wins**, +1.22 paired sd |

---

## Phase 0 — defects that made prior results unmeasurable

Three bugs, each of which independently invalidated earlier numbers.

**Silent auto-resume** (`basicsr/train.py`). The resume path globbed
`experiments/<name>/training_states/` and overrode `resume_state: ~` from the
YAML unconditionally. A stale `250000.state` in a *new* experiment directory
caused a run to resume at `iter == total_iter`, train for **one second**, and
report a validation PSNR belonging to entirely different weights. Now opt-in
and refuses both dangerous cases.

**Frequency loss band normalisation.** Both bands reduced with `.mean()` over
the full rfft grid. The low-pass disc is ~5% of the grid, so it was diluted
~20×: a configured `w_low:w_high` of 1:2 acted as **1:38**. Measured high/low
ratio at equal unit weights: **19.5 before, 1.02 after**.

**Dead illumination-TV prior.** The hook sat on `estimator.dwt`, whose output is
the Haar decomposition of the *raw input image*. `HaarDWT2D` holds its filters in
`register_buffer` and has no parameters, so that tensor has
`requires_grad=False`, `grad_fn=None`. `∂L_tv/∂θ = 0` for every parameter;
`.backward()` on it alone *raises*. It never crashed because the total loss
stayed differentiable through the other terms, and the check that was supposed
to catch it asserted the loss **value** was positive rather than its gradient.

> Lesson encoded as `scripts/gradient_flow.py`: every configured loss term must
> be proven to move at least one parameter. Wired into `scripts/preflight.py`.

**Ladder drift.** A0 carried `clip_grad_norm: 0.01` and a single-cycle schedule
while A1/A4/A7 used `1.0` and a 2-cycle schedule — so any "improvement over
baseline" was confounded by two hyperparameter changes. Fixed by a ladder-aligned
A0 config; `scripts/check_ladder_alignment.py` verifies 31 controlled fields.

---

## ICNF pilots (real-photo corpus, synthetic degradation)

Setup: 74 real images (29 train / 8 val), 1500 iters, `n_feat=16`, 64px patches,
batch 4, 4 seeds, paired by seed. CPU.

### Matched noise — ICNF's assumption holds exactly

| model | seeds | mean |
|---|---|---|
| A4 | 17.276, 18.124, 17.442, 19.155 | 17.999 |
| ICNF | 17.991, 19.752, 19.493, 19.575 | 19.203 |

**Δ = +1.20 dB, 4/4 seeds.** Favourable by construction — necessary, not sufficient.

### Mismatched noise — assumption deliberately broken

Heavy-tailed, **signal-independent** noise, which the Poisson-Gaussian model gets wrong.

| model | mean |
|---|---|
| A4 | 18.501 |
| ICNF | 19.523 |

**Δ = +1.02 dB, 4/4 seeds.** If the physics were load-bearing, breaking the noise
law should have collapsed the advantage. It barely moved. **First evidence that
the win is not coming from the noise model being correct.**

### Conditioning control — the decisive one

`ICNFc` is full ICNF except the noise floor is made spatially **uniform**
(per-image global mean). Same per-pixel gate; no illumination conditioning.

| model | mean | std |
|---|---|---|
| A4 | 17.9991 | 0.739 |
| ICNF (illumination-conditioned) | 19.2026 | 0.706 |
| **ICNFc (uniform floor)** | **19.7281** | 0.579 |

| contrast | Δ | paired wins |
|---|---|---|
| ICNF − A4 | +1.2035 dB | 4/4 |
| ICNFc − A4 | **+1.7290 dB** | 4/4 |
| **ICNFc − ICNF** | **+0.5255 dB** | **4/4** |

**Removing the illumination-conditioning made the model better on every seed.**

Mechanism: dividing by a spatially-varying floor injects illumination structure
into the evidence map, so the gate partly tracks *brightness* rather than
*texture*. A uniform floor lets it respond to local high-frequency energy, which
is the quantity that actually matters.

### Consequences for the paper

Dropped:
- "In a Retinex model the illumination estimate *is* a noise model." Not supported.
- The Parseval / "why wavelets" argument. It justified transferring a
  *pixel-domain* noise variance into the subbands; with a single per-image
  constant there is nothing to transfer, and MAD would serve as well.

Retained (pending P9):
- A **per-pixel adaptive gate** beats A4's scalar gate by +1.73 dB, 4/4 seeds,
  +2.26 pooled sd — a larger effect than the version originally proposed.

Diagnosed cause, found by test before any of these runs: A4's gate is
**unidentifiable at initialisation**. `Freq_MSA.out_proj` is zero-init, so the
frequency residual `B ≡ 0`, so `dL/d(gate) = dL/dx · B = 0`. The scalar gate may
never have moved off its initial value across an entire training run.

---

## P9 — init fix vs spatial gate: the gate is real

`ICNFc` differs from A4 in **two** ways, not one: nonzero `out_proj` init *and*
the per-pixel gate. `A4nz` is plain A4 with only the init changed, so the two
effects can be separated.

| model | mean | std |
|---|---|---|
| A4 (scalar gate, zero-init) | 17.9991 | 0.739 |
| A4nz (A4 + nonzero init only) | 18.2030 | 0.867 |
| **ICNFc (per-pixel gate)** | **19.7281** | 0.579 |

| contrast | isolates | Δ | paired wins |
|---|---|---|---|
| A4nz − A4 | initialisation fix | +0.2038 dB | **2/4** |
| **ICNFc − A4nz** | **the spatial gate** | **+1.5251 dB** | **4/4** |
| ICNFc − A4 | both together | +1.7290 dB | 4/4 |

**The initialisation fix explains almost none of the gain.** +0.20 dB at 2/4
paired wins is a coin flip, and it sits well inside the ~0.8 dB seed spread —
read it as null. Of the +1.73 dB total, **+1.53 dB (88%) is the per-pixel gate**,
and that contrast is unanimous.

The nonzero init remains **necessary** — without it the gate is unidentifiable
(`out_proj` zero-init ⟹ `B ≡ 0` ⟹ `dL/d(gate) = 0`) — but it is **not
sufficient**: on its own it buys nothing. The mechanism does the work.

This was the live threat to the whole project: had `A4nz` captured the gain, the
contribution would have collapsed to "A4 shipped an initialisation bug". It did not.

### The surviving claim

A4's scalar gate is a **defect**: one learned scalar per block cannot express
"trust high frequencies *here* but not *there*", which is precisely the decision
the block must make, because the same absolute high-frequency energy means
texture in a lit region and noise in a shadow. It is also unidentifiable at
initialisation, so it may never move off its initial value.

Replacing it with a **per-pixel gate** driven by local high-frequency energy
against a **uniform** noise floor recovers **+1.53 dB over A4nz, 4/4 seeds**.
No Retinex physics, no wavelet-necessity argument — a gating mechanism whose
failure mode is diagnosed and whose causal contribution is isolated by three
controls (mismatched noise, constant floor, init-only).

---

## Standing caveats

These pilots are **direction, not result**: 1500 iters, `n_feat=16`, 64px
patches, synthetic degradation, 8 validation images, CPU. Full-scale LOL-v1
training can still overturn any of it. No LOL-v1 number in this document has
been reproduced under the corrected configs yet — that is P4/P5 and requires a
GPU.

Evaluation protocol notes for whoever writes the paper:
- No GT-mean adjustment anywhere in the eval path; numbers are raw PSNR and are
  only comparable to other raw numbers.
- `val` is `data/LOLv1/Test`, and `net_g_best.pth` is selected on it — the
  "PSNR best" column is test-selected and should not be reported. The
  `robust_eval` last-10-checkpoint mean does not touch `best_psnr_*.pth` and is
  the honest number.


---

## P10 — scale sensitivity at the production architecture

The surviving claim was measured at `n_feat=16`, `num_blocks=[1,1,1]` — one
sixth the channel width of the real config. Re-run at the **exact LOL-v1
architecture**: `n_feat=40`, `num_blocks=[1,2,2]`, 2,167,464 params, identical
to `Options/train_FD2RT_ICNF_LOL_v1.yml`. 4 paired seeds, 1500 iters, CPU
(~10.6 min/run).

| model | mean | std | per-seed |
|---|---|---|---|
| A4 | 20.2546 | 1.523 | 19.454, 22.891, 19.392, 19.282 |
| **ICNFc** | **21.6293** | 1.022 | 20.542, 22.887, 22.380, 20.709 |

**Δ = +1.3747 dB, 3/4 paired wins, +0.92 pooled sd.**

### Scale comparison

| scale | Δ | paired wins | effect size |
|---|---|---|---|
| `n_feat=16`, `[1,1,1]` | +1.73 dB | 4/4 | +2.26 sd |
| `n_feat=40`, `[1,2,2]` | +1.37 dB | 3/4 | **+0.92 sd** |

**The effect survives but weakens: effect size falls ~60% at production width.**
The mean gap barely moved (−0.36 dB), but A4's seed variance doubled
(0.74 → 1.52), which is what collapses the standardised effect.

**Seed 1 is the informative failure.** A4 22.891 vs ICNFc 22.887 — a dead tie
(−0.005 dB), on the seed where *both* models did far better than the other
three. The baseline found a good basin unaided and the gate bought nothing
there. ICNFc never *lost* on any seed, but it is no longer unanimous.

Read honestly: the trend runs the wrong way as capacity grows, and LOL-v1 at
250K iterations is a far larger scale jump than the one tested here. The gap
could compress further. This is the main risk the GPU runs must resolve.

### Secondary claim, visible in both experiments

**ICNFc consistently reduces run-to-run variance**: std 1.02 vs 1.52 at
production scale, 0.58 vs 0.74 at reduced scale. It lifts the weak seeds much
more than the strong one (seed 2: 19.39 → 22.38; seed 1: no change). If the
mean gap compresses further at full scale, *training stability* may be the more
durable contribution — and it is consistent with the diagnosed mechanism, since
a scalar gate that is unidentifiable at init leaves the frequency branch's
usefulness to chance.


---

## Why the LOL-v1 benchmark runs did not happen here

Not a matter of effort. Two independent hard blocks, both verified rather than
assumed.

**1. No GPU.** `torch.cuda.is_available()` False; no `/dev/nvidia*`; no
`libcuda.so`; `CLAUDE_CODE_REMOTE_ENVIRONMENT_TYPE=cloud_default`.

**2. LOL-v1 is unreachable.** Every host that distributes it is denied by the
network policy: Google Drive, Kaggle and Zenodo refuse connections;
huggingface.co returns a logged `403 policy denial` at the proxy gateway;
`api.github.com` is blocked. Only `raw.githubusercontent.com` answers, and the
dataset is not distributed there.

**3. Even with the data, CPU training is infeasible by two orders of
magnitude.** Measured at the exact training config (`n_feat=40`,
`num_blocks=[1,2,2]`, batch 8, patch 128):

    4.47 s/step
    one 250K-iteration run   ~310 h   = 12.9 days
    four-model ladder                 = 52 days
    with 3 seeds each                 = 155 days

The container is ephemeral and reclaimed after inactivity, so a 13-day single
run is not merely slow, it cannot complete.

**Conclusion.** SOTA on LOL-v1 was not achieved and is not achievable in this
environment. What is delivered instead is a codebase where measurement is
trustworthy, a controlled ladder, a pre-flight gate, and nine recorded verdicts
- five of which killed hypotheses that would otherwise have consumed GPU-weeks
before failing.


---

## P11 — spatial adaptivity vs gate level: the mechanism is confirmed

ICNFc differed from A4 in two ways at once *again*: its gate varies spatially
**and** averages 0.278 where A4's scalar gate sits at sigmoid(0) = 0.5. The
boring explanation — "0.28 is simply a better mixing level, a one-line fix" —
was live and had to be closed.

`A4const` = A4 + the identifiability fix + a scalar gate pinned to exactly
0.278: **same mean mixing, zero spatial variation.** Production architecture,
4 paired seeds.

| model | mean | std | per-seed |
|---|---|---|---|
| A4 (scalar @ 0.5) | 20.2546 | 1.523 | 19.454, 22.891, 19.392, 19.282 |
| A4const (scalar @ 0.278) | 20.0021 | 1.189 | 19.607, 22.028, 19.372, 19.002 |
| **ICNFc (per-pixel, mean 0.278)** | **21.6293** | 1.022 | 20.542, 22.887, 22.380, 20.709 |

| contrast | isolates | Δ | paired wins |
|---|---|---|---|
| A4const − A4 | gate **level** alone | **−0.2525 dB** | — |
| **ICNFc − A4const** | **spatial adaptivity** alone | **+1.6272 dB** | **4/4** |

**A better constant buys nothing.** Moving the scalar gate from 0.5 to ICNFc's
mean of 0.278 is slightly *harmful* (−0.25 dB). The whole effect comes from the
gate varying across space.

Controlling for the level **sharpened** the result rather than shrinking it:

| comparison | Δ | wins | effect size |
|---|---|---|---|
| ICNFc − A4 (uncontrolled) | +1.37 dB | 3/4 | +0.92 sd |
| **ICNFc − A4const (controlled)** | **+1.63 dB** | **4/4** | **+1.27 sd** |

The gate level was working slightly against the mechanism, so removing it as a
confound revealed a larger, unanimous effect. This is the strongest and
cleanest result in the project.

### The claim, as it now stands

A4's frequency-branch gate is **one learned scalar per block**. That cannot
express "trust high frequencies *here* but not *there*" — the decision the block
must make, because identical absolute high-frequency energy means texture in a
lit region and sensor noise in a shadow. The gate is also **unidentifiable at
initialisation** (`out_proj` zero-init ⟹ `B ≡ 0` ⟹ `dL/d(gate) = 0`), so it may
never move off its initial value.

Replacing it with a **per-pixel gate** driven by local high-frequency energy
against a **uniform** noise floor gives **+1.63 dB over a level-matched control,
4/4 seeds, +1.27 pooled sd**, at the exact LOL-v1 architecture.

Isolated by four controls: mismatched noise (physics not the cause), constant
floor (illumination-conditioning harmful), init-only (null), and level-matched
(constant gate buys nothing). No Retinex physics, no wavelet-necessity argument
— a gating mechanism whose failure mode is diagnosed and whose causal
contribution is measured.


---

## P12 — per-channel gate: suggestive, not established

The gate emits one scalar per pixel shared across all 40 feature channels.
`ICNFpc` adds a 1x1 projection so channels can specialise, initialised to
reproduce the shared gate exactly (output difference 5.8e-07 at init), making it
a strict generalisation. Cost +1440 params (0.07%).

| model | mean | std | per-seed |
|---|---|---|---|
| ICNFc (shared gate) | 21.6293 | 1.022 | 20.542, 22.887, 22.380, 20.709 |
| ICNFpc (per-channel) | 22.0990 | 1.067 | 20.482, 23.438, 22.469, **22.007** |

**Δ = +0.4697 dB, 3/4 paired wins, +0.39 pooled sd.**

**Do not treat this as established.** It is the weakest result in the project:

* effect size +0.39 sd, against +1.27 sd for the spatial-gate mechanism itself
* seed 0 **lost** (−0.06 dB); two of the three wins are small (+0.09, +0.55)
* **seed 3 alone supplies +1.30 dB of the +0.47 mean.** Excluding it leaves
  ≈ +0.18 dB — essentially nothing

A result carried by a single seed is exactly the shape that dissolves under
replication. Honest summary: per-channel gating *may* add ~0.5 dB and needs more
seeds before it is claimed.

### Cumulative effect at the production architecture

| step | Δ | paired wins | effect size |
|---|---|---|---|
| A4 → ICNFc (spatial gate) | **+1.63 dB** | 4/4 | +1.27 sd |
| ICNFc → ICNFpc (per-channel) | +0.47 dB | 3/4 | +0.39 sd |
| **A4 → ICNFpc (total)** | **≈ +2.10 dB** | — | — |

Recommendation for the GPU campaign: train **ICNFc** as the primary arm — it is
the claim that survived four controls unanimously. Carry ICNFpc as a secondary
arm only if seed budget allows, and report it as an ablation, not a headline,
unless more seeds firm it up.


---

## P13 — per-channel gate refuted by replication

P12's +0.47 dB rested on a single seed, so it was flagged as fragile rather
than banked. Seeds 4-7 were run to settle it.

| seeds | Δ (ICNFpc − ICNFc) | paired wins |
|---|---|---|
| 0–3 (P12) | +0.47 dB | 3/4 |
| **4–7 (P13)** | **−0.08 dB** | **1/4** |
| **0–7 pooled** | **+0.20 dB** | **4/8 — chance** |

Per-seed deltas across all eight:

    -0.060  +0.552  +0.089  +1.298  -0.069  +0.204  -0.414  -0.038

**Four wins out of eight is exactly chance.** The per-channel gate does
nothing. The apparent +0.47 dB was seed 3 (+1.298 dB) dominating a
four-sample mean.

Seeds 4-7 also happen to be a quieter block (ICNFc sd 0.43 vs 1.02 for seeds
0-3), which makes the null cleaner rather than noisier.

**Why this matters procedurally.** Had P12's +0.47 dB been reported as a
headline, the cumulative claim would have been +2.10 dB instead of +1.63 dB —
a phantom 0.47 dB carried into a GPU campaign and into a paper. Four extra CPU
hours prevented that. It is the seventh hypothesis killed in this project and
the second killed specifically by replication rather than by a control.

**Consequence:** ICNFc (shared per-pixel gate) is the final architecture.
`gate_per_channel` remains implemented and is worth reporting as a negative
ablation — "channel specialisation does not help" is a legitimate finding —
but it is not part of the claim.


---

## P14 — primary claim replicated (8-seed verdict)

P13 taught that a 4-seed result at 3/4 wins can evaporate, so the PRIMARY claim
(ICNFc vs A4const, +1.63 dB on seeds 0-3) was held to the same standard: seeds
4-7 were run and pooled.

| block | ICNFc − A4const | paired wins |
|---|---|---|
| seeds 0–3 (P11) | +1.6272 dB | 4/4 |
| seeds 4–7 (P14) | +0.9264 dB | 3/4 |
| **pooled 0–7** | **+1.2768 dB** | **7/8, +1.22 paired sd** |

Per-seed deltas across all eight:

    +0.935  +0.859  +3.008  +1.707  +2.410  +1.016  +0.476  -0.197

Unlike the per-channel gate (P13), this did NOT collapse: every block points the
same way, seven of eight seeds are positive, and the one negative (-0.20) is the
smallest deviation in the set. The effect is not carried by any single seed. The
second block is weaker (+0.93 vs +1.63) but confirms direction and sign.

**Verdict.** The spatial-adaptivity contribution is the one claim in this project
that survived (a) four isolating controls and (b) replication on fresh seeds. It
is the claim to take to the GPU. Pooled effect at the production architecture:
ICNFc beats a level-matched, identifiability-fixed A4 baseline by **+1.28 dB,
7/8 seeds**. Still a scaled-down synthetic pilot -- direction, not a benchmark
number -- but it is now as well-supported as CPU experiments can make it.
