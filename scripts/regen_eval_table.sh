#!/usr/bin/env bash
# =============================================================================
# Regenerate results/robust_eval/comparison_table.csv from trained experiments.
#
# Run on the GPU box AFTER training, when the campaign finished but the CSV is
# missing (usually because an eval step errored — the campaign continues past
# eval failures so training still completes).
#
#   git pull                       # get the robust_eval FD2RT_ICNF fix first
#   bash scripts/regen_eval_table.sh
#
# It discovers every experiments/*/models/ dir that has net_g_*.pth, infers the
# architecture from the experiment name, evaluates each (last-10-checkpoint
# robust mean, excluding the test-selected net_g_best.pth), and writes a fresh
# comparison table. Idempotent: it backs up any existing CSV first so re-runs
# don't pile up duplicate rows.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
OUT="${OUT:-results/robust_eval}"
DATA="${DATA:-data/LOLv1/Test}"
mkdir -p "$OUT"

echo "=== 0. sanity ==="
$PYTHON -c "import torch;print('  cuda:',torch.cuda.is_available())" 2>/dev/null || true
[ -d "$DATA/input" ] && echo "  test data: $(ls "$DATA"/input 2>/dev/null | wc -l) input files in $DATA" \
                     || echo "  WARNING: $DATA/input not found — pass DATA=... to point at LOL-v1 Test"

echo
echo "=== 1. experiments present ==="
found=0
for d in experiments/*/models; do
  [ -d "$d" ] || continue
  n=$(ls "$d"/net_g_*.pth 2>/dev/null | wc -l)
  exp=$(basename "$(dirname "$d")")
  printf "  %-38s  %s net_g_*.pth\n" "$exp" "$n"
  [ "$n" -gt 0 ] && found=1
done
if [ "$found" = 0 ]; then
  echo
  echo "  No net_g_*.pth checkpoints found. Either training did not complete, or"
  echo "  checkpoints are named differently. Raw listing of what IS there:"
  ls -R experiments/*/models 2>/dev/null | head -60
  echo
  echo "  (robust_eval globs net_g_*.pth; best checkpoints are net_g_best.pth and"
  echo "   are intentionally excluded. If you only see best_psnr_*.pth, training"
  echo "   was interrupted before a periodic net_g_<iter>.pth was saved.)"
  exit 1
fi

# infer arch from experiment name (order matters: check ICNF/A0/A1 before A4)
arch_for(){
  case "$1" in
    *ICNF*)                     echo FD2RT_ICNF ;;   # illum_source defaults to constant
    *A0*|*[Rr]etinexformer*)    echo RetinexFormer ;;
    *A1*|*FD2RT_V1*)            echo FD2RT_V1 ;;
    *A4*)                       echo FD2RT_A4 ;;
    *)                          echo "" ;;
  esac
}

# fresh table so re-runs don't accumulate duplicate rows
CSV="$OUT/comparison_table.csv"
[ -f "$CSV" ] && mv -f "$CSV" "$CSV.$(date +%s).bak" && echo "  (backed up old CSV)"

echo
echo "=== 2. evaluating each experiment ==="
for d in experiments/*/models; do
  [ -d "$d" ] || continue
  ls "$d"/net_g_*.pth >/dev/null 2>&1 || continue
  exp=$(basename "$(dirname "$d")")
  arch=$(arch_for "$exp")
  if [ -z "$arch" ]; then echo "  SKIP $exp  (cannot infer arch from name)"; continue; fi
  echo "  -> $exp   as $arch"
  $PYTHON scripts/robust_eval.py --arch "$arch" --ckpt_dir "$d" \
      --data_root "$DATA" --label "$exp" --out_dir "$OUT" \
    || echo "     EVAL FAILED for $exp (see error above)"
done

echo
echo "=== 3. comparison table ==="
if [ -f "$CSV" ]; then
  column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
  echo
  echo "  wrote $CSV"
  echo "  Report the PSNR_robust (mean) column, NOT PSNR_best (test-selected)."
  echo "  These are RAW PSNR (no GT-mean adjustment) — compare only to raw numbers."
else
  echo "  CSV still not created — every eval failed. Paste the first error above."
fi
