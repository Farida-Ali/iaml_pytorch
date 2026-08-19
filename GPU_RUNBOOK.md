# GPU Runbook

What is needed to finish this project, and the exact order to do it in.
Everything here is committed and pre-flighted; none of it has been executed,
because the development container has no GPU and cannot reach LOL-v1.

---

## 1. What I need from you

### Hardware
One CUDA GPU. VRAM is not the constraint — the model is 2.17M parameters and
trains at batch 8 / patch 128, which fits comfortably in 8–12 GB. Any card that
ran the earlier A4 experiments is sufficient.

### Data
| dataset | path expected by the configs | needed for |
|---|---|---|
| LOL-v1 train (485 pairs) | `data/LOLv1/Train/{input,target}` | **required** — all ladder runs |
| LOL-v1 test (15 pairs) | `data/LOLv1/Test/{input,target}` | **required** — validation + eval |
| LOL-v2-real | `data/LOL_v2_real/...` | for the paper's dataset suite |
| LOL-v2-synthetic | `data/LOL_v2_synthetic/...` | for the paper's dataset suite |

The container cannot download these: Drive, Kaggle, Zenodo and HuggingFace are
all blocked by the network policy. They must already be on the GPU machine, or
be placed there by you.

### Time budget
Assume ~12–20 h per 250K-iteration run on a modern GPU.

| stage | runs | purpose |
|---|---|---|
| A0 anchor | 1 | gates everything; nothing is interpretable without it |
| ladder A1 / A4 / ICNFc | 3 | the ablation |
| seed replication | ×3 | **not optional** — see §4 |

Minimum credible campaign: **4 runs** to get a first read, **12 runs** for a
result that can survive review.

### Environment note
If this is run from a Claude Code session, the environment must be GPU-backed
**at container creation** — a grant applied to an already-running session is not
picked up. Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. Run order

Nothing starts until pre-flight passes. It refuses stale experiment state, dead
loss terms, and ladder drift — each of which has already destroyed a result in
this project.

```bash
git pull
bash scripts/run_all_tests.sh          # expect: ALL SUITES GREEN
```

### Step 1 — A0 anchor, alone

```bash
python scripts/preflight.py --opt Options/train_Retinexformer_A0_ladder_LOL_v1.yml
python -m basicsr.train      --opt Options/train_Retinexformer_A0_ladder_LOL_v1.yml
```

**Do not start anything else until this finishes.** Every other number is
expressed relative to A0, and the previous A0 was trained with a different
grad-clip and LR schedule, which is why the old comparisons were unusable.

**Exit criterion:** a stable A0 PSNR under the corrected config. Report it back
before proceeding.

### Step 2 — the ladder

```bash
for cfg in Options/train_FD2RT_A1_LOL_v1_fixed.yml \
           Options/train_FD2RT_A4_LOL_v1_fixed.yml \
           Options/train_FD2RT_ICNF_LOL_v1.yml; do
  python scripts/preflight.py --opt $cfg || break
  python -m basicsr.train      --opt $cfg
done
```

### Step 3 — evaluation

```bash
python scripts/robust_eval.py --arch RetinexFormer --ckpt_dir experiments/train_Retinexformer_A0_ladder_LOL_v1/models --label A0
python scripts/robust_eval.py --arch FD2RT_V1      --ckpt_dir experiments/train_FD2RT_A1_LOL_v1_fixed/models        --label A1
python scripts/robust_eval.py --arch FD2RT_A4      --ckpt_dir experiments/train_FD2RT_A4_LOL_v1_fixed/models        --label A4
python scripts/robust_eval.py --arch FD2RT_ICNF    --ckpt_dir experiments/train_FD2RT_ICNF_LOL_v1/models            --label ICNF
```

Report the **`PSNR robust` mean ± std** column (last-10-checkpoint average).
Do **not** report `PSNR best` — see §3.

---

## 3. Two protocol facts that must not be forgotten

**No GT-mean adjustment.** There is none anywhere in the eval path, so these are
raw PSNR numbers. Many published LOL results *are* GT-mean adjusted and sit 2–4 dB
higher for that reason alone. Comparing across that line is the single most
common apples-to-oranges error in this subfield.

**`net_g_best.pth` is selected on the test set.** Every config validates against
`data/LOLv1/Test` and saves the best checkpoint by that metric. The "PSNR best"
column is therefore test-selected and a reviewer will reject it. `robust_eval`
globs `net_g_*.pth` and excludes `best_psnr_*.pth`, so its last-10 mean is clean —
that is the number to publish.

---

## 4. Why seeds are not optional

At production scale the measured effect size is **+0.92 pooled sd** with A4's
seed spread at **1.52 dB**. A single run per arm cannot separate the effect from
seed noise — the difference you are trying to detect is smaller than the
variation between two runs of the same model.

Set `manual_seed` to at least three values per arm and report mean ± std across
seeds. If only a partial budget exists, spend it on **seeds for A4 vs ICNFc**
rather than on breadth across A0/A1.

---

## 5. Open question the GPU must settle

Whether the gate advantage survives at full scale and full training length.

| scale | Δ (ICNFc − A4) | paired wins | effect size |
|---|---|---|---|
| `n_feat=16`, `[1,1,1]`, 1500 it | +1.73 dB | 4/4 | +2.26 sd |
| `n_feat=40`, `[1,2,2]`, 1500 it | +1.37 dB | 3/4 | +0.92 sd |
| `n_feat=40`, `[1,2,2]`, 250K it | **unknown** | — | — |

The trend runs the wrong way as capacity grows. LOL-v1 at 250K iterations is a
far larger jump than the one tested. **This is the main risk**, and it is the
question the campaign exists to answer.

A secondary claim strengthened under scrutiny and is worth measuring
explicitly: ICNFc consistently **reduces run-to-run variance** (std 1.02 vs 1.52
at production scale; 0.58 vs 0.74 at reduced scale), lifting weak seeds far more
than strong ones. If the mean gap compresses at full scale, training stability
may be the more durable contribution — but it can only be shown with multiple
seeds, which is a second reason §4 matters.

---

## 6. What to send back

- The A0 anchor number, before anything else runs.
- `results/robust_eval/comparison_table.csv`.
- Per-seed PSNRs for A4 and ICNFc, not just the means.
- Any pre-flight failure, verbatim — it is designed to catch exactly the
  failures that have already cost this project results.
