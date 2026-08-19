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
| H8 | That gain is the gate, not merely the init fix | **UNDER TEST** (P9) |

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

## P9 — init fix vs spatial gate (in progress)

`ICNFc` differs from A4 in **two** ways, not one: nonzero `out_proj` init *and*
the per-pixel gate. `A4nz` is plain A4 with only the init changed.

- `A4nz − A4` isolates the initialisation fix
- `ICNFc − A4nz` isolates the gate mechanism

If `A4nz` captures most of the +1.73 dB, the contribution reduces to
"A4 had an initialisation bug" — a paragraph, not a paper.

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
