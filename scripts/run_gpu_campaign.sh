#!/usr/bin/env bash
# =============================================================================
# FD²RT GPU campaign — the full LOL-v1 ladder, one command.
#
# RUN THIS ON THE GPU BOX (it has the GPU + data; the dev sandbox has neither).
#   1. git clone / git pull this branch onto the GPU machine
#   2. put LOL-v1 at  data/LOLv1/{Train,Test}/{input,target}
#   3. bash scripts/run_gpu_campaign.sh   2>&1 | tee campaign.log
#
# It is idempotent-ish and safe:
#   * pre-flights every config before training (refuses stale state, dead loss,
#     ladder drift) — the guards that already saved this project twice
#   * trains A0 ANCHOR FIRST and STOPS if it fails — nothing downstream is
#     interpretable without it
#   * trains the ladder A1 -> A4 -> ICNFc, ICNFc and A4 across N_SEEDS seeds
#     (default 3) because the production-scale effect is ~+1 pooled sd and a
#     single run per arm cannot separate it from seed noise
#   * evaluates every run with robust_eval (last-10-checkpoint mean; excludes
#     the test-selected net_g_best.pth) and appends to a comparison CSV
#   * writes a final summary you can paste back
#
# Config knobs (env vars):
#   N_SEEDS=3     seeds for the A4-vs-ICNFc comparison (A0/A1 use 1)
#   SKIP_A1=0     set 1 to skip the A1 rung if the budget is tight
#   DRY=0         set 1 to print the plan and pre-flight only, no training
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

N_SEEDS="${N_SEEDS:-3}"
SKIP_A1="${SKIP_A1:-0}"
DRY="${DRY:-0}"
PYTHON="${PYTHON:-python}"
TRAIN="$PYTHON -m basicsr.train --opt"
PRE="$PYTHON scripts/preflight.py --opt"
EVAL="$PYTHON scripts/robust_eval.py"

log(){ echo -e "\n\033[1m[$(date +%H:%M:%S)] $*\033[0m"; }
die(){ echo -e "\n\033[31mFATAL: $*\033[0m" >&2; exit 1; }

# --- 0. environment sanity -------------------------------------------------
log "0. Environment"
$PYTHON -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
print('  GPU:', torch.cuda.get_device_name(0))" \
  || die "CUDA not available. Run this on the GPU box, not the dev sandbox."
for d in data/LOLv1/Train/input data/LOLv1/Train/target \
         data/LOLv1/Test/input  data/LOLv1/Test/target; do
  n=$(ls "$d" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || die "no images in $d — place LOL-v1 on disk first"
  echo "  $d : $n files"
done
log "0b. Full regression suite (must be green before any training)"
bash scripts/run_all_tests.sh || die "regression suite failed — do not train"

# --- helper: train one config into a uniquely-named experiment -------------
# args: <config.yml> <run_name> <seed>
train_one(){
  local cfg="$1" name="$2" seed="$3"
  local tmp="Options/_campaign_${name}.yml"
  # derive a per-run config: override name + seed, keep everything else
  sed -e "s/^name:.*/name: ${name}/" \
      -e "s/^manual_seed:.*/manual_seed: ${seed}/" "$cfg" > "$tmp"
  log "PRE-FLIGHT  $name  (seed $seed)"
  $PRE "$tmp" || { echo "  pre-flight FAILED for $name"; return 1; }
  if [ "$DRY" = "1" ]; then echo "  [DRY] would train $name"; return 0; fi
  log "TRAIN  $name  (seed $seed)"
  $TRAIN "$tmp" || { echo "  training FAILED for $name"; return 1; }
  return 0
}

# args: <arch> <run_name> <label>
eval_one(){
  local arch="$1" name="$2" label="$3"
  [ "$DRY" = "1" ] && { echo "  [DRY] would eval $label"; return 0; }
  log "EVAL  $label"
  $EVAL --arch "$arch" --ckpt_dir "experiments/${name}/models" \
        --data_root data/LOLv1/Test --label "$label" \
        --out_dir results/robust_eval || echo "  eval FAILED for $label"
}

# --- 1. A0 anchor — alone, first, gates everything -------------------------
log "1. A0 anchor (ladder-aligned RetinexFormer)"
train_one Options/train_Retinexformer_A0_ladder_LOL_v1.yml \
          campaign_A0_s100 100 \
  || die "A0 anchor failed. STOP: every downstream number is relative to A0."
eval_one RetinexFormer campaign_A0_s100 A0

# --- 2. A1 rung (single seed; optional) ------------------------------------
if [ "$SKIP_A1" = "0" ]; then
  log "2. A1 rung (W-IE)"
  train_one Options/train_FD2RT_A1_LOL_v1_fixed.yml campaign_A1_s100 100 \
    && eval_one FD2RT_V1 campaign_A1_s100 A1
else
  log "2. A1 rung SKIPPED (SKIP_A1=1)"
fi

# --- 3. A4 vs ICNFc — the headline comparison, N_SEEDS each ----------------
log "3. A4 vs ICNFc across $N_SEEDS seeds (the claim under test)"
SEEDS=(); for i in $(seq 0 $((N_SEEDS-1))); do SEEDS+=( $((100+i)) ); done
for s in "${SEEDS[@]}"; do
  train_one Options/train_FD2RT_A4_LOL_v1_fixed.yml "campaign_A4_s${s}"   "$s" \
    && eval_one FD2RT_A4   "campaign_A4_s${s}"   "A4_s${s}"
  # ICNFc: the FD2RT_ICNF config already carries illum_source: constant
  train_one Options/train_FD2RT_ICNF_LOL_v1.yml   "campaign_ICNFc_s${s}" "$s" \
    && eval_one FD2RT_ICNF "campaign_ICNFc_s${s}" "ICNFc_s${s}"
done

# --- 4. summary ------------------------------------------------------------
log "4. Campaign complete — summary"
CSV=results/robust_eval/comparison_table.csv
if [ -f "$CSV" ]; then
  echo "  comparison table: $CSV"; echo; column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
fi
$PYTHON - <<'PY' || true
import csv, os, statistics as st, collections
p='results/robust_eval/comparison_table.csv'
if not os.path.exists(p): raise SystemExit
rows=list(csv.DictReader(open(p)))
def col(r):
    for k in ('PSNR_mean','psnr_mean','PSNR_robust','psnr'):
        if k in r: return float(r[k])
    return None
by=collections.defaultdict(list)
for r in rows:
    lbl=r.get('label') or r.get('Label') or ''
    base=lbl.split('_s')[0] if '_s' in lbl else lbl
    v=col(r)
    if v is not None: by[base].append(v)
print("\n  arm         n   PSNR mean   std")
print("  " + "-"*40)
for k in ('A0','A1','A4','ICNFc'):
    if by.get(k):
        xs=by[k]; sd=st.pstdev(xs) if len(xs)>1 else 0.0
        print(f"  {k:<10} {len(xs):>2}   {st.mean(xs):>8.4f}   {sd:>5.3f}")
if by.get('A4') and by.get('ICNFc'):
    d=st.mean(by['ICNFc'])-st.mean(by['A4'])
    print(f"\n  ICNFc - A4 : {d:+.4f} dB  (report PSNR_robust, NOT PSNR_best)")
PY
echo
echo "  Send back: results/robust_eval/comparison_table.csv and this summary."
echo "  Reminders: these are RAW PSNR (no GT-mean adjustment) -> compare only to"
echo "  raw published numbers; PSNR_best is test-selected -> do not report it."
