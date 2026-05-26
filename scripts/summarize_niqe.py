"""
scripts/summarize_niqe.py
─────────────────────────
Aggregate per-image NIQE CSVs produced by run_niqe_eval.py into a
summary table saved as niqe_summary.md and printed to stdout.

Discovers all niqe_results.csv files under --results_dir automatically,
or accepts an explicit list via --csv_files.

Usage
─────
# Auto-discover all CSVs under the non_reference output tree
python scripts/summarize_niqe.py \
    --results_dir qualitative_outputs/non_reference \
    --out_dir     qualitative_outputs/non_reference

# Or specify CSVs explicitly
python scripts/summarize_niqe.py \
    --csv_files \
        qualitative_outputs/non_reference/LIME/IAML_A4/niqe_results.csv \
        qualitative_outputs/non_reference/LIME/Retinexformer_A0/niqe_results.csv \
        qualitative_outputs/non_reference/NPE/IAML_A4/niqe_results.csv \
        qualitative_outputs/non_reference/NPE/Retinexformer_A0/niqe_results.csv \
    --out_dir qualitative_outputs/non_reference

Output
──────
  <out_dir>/niqe_summary.md   — markdown table + per-dataset delta vs A0
  <out_dir>/niqe_all.csv      — single aggregated CSV with all per-image rows
"""

import sys
import os
import csv
import argparse
from glob import glob
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description='Aggregate NIQE CSVs into a summary markdown table.')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--results_dir', default=None,
                   help='Root directory to search recursively for niqe_results.csv files.')
    g.add_argument('--csv_files', nargs='+', default=None,
                   help='Explicit list of niqe_results.csv file paths.')
    p.add_argument('--out_dir', required=True,
                   help='Directory to save niqe_summary.md and niqe_all.csv.')
    p.add_argument('--baseline_model', default=None,
                   help='Model name to use as baseline for delta computation '
                        '(e.g. "Retinexformer_A0"). Auto-detected if not set.')
    return p.parse_args()


def load_csv(path: str) -> list:
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                score = float(row['niqe_score'])
            except (ValueError, KeyError):
                score = float('nan')
            rows.append({
                'dataset':    row.get('dataset', ''),
                'model':      row.get('model', ''),
                'filename':   row.get('filename', ''),
                'niqe_score': score,
            })
    return rows


def main():
    args = parse_args()

    # Collect CSV paths
    if args.results_dir:
        csv_paths = sorted(
            glob(os.path.join(args.results_dir, '**', 'niqe_results.csv'),
                 recursive=True))
        if not csv_paths:
            print(f'ERROR: no niqe_results.csv files found under {args.results_dir}')
            return 1
    else:
        csv_paths = args.csv_files

    print(f'[Input] {len(csv_paths)} CSV file(s):')
    for p in csv_paths:
        print(f'  {p}')

    # Load all rows
    all_rows = []
    for path in csv_paths:
        if not os.path.isfile(path):
            print(f'  WARNING: not found — {path}')
            continue
        all_rows.extend(load_csv(path))

    if not all_rows:
        print('ERROR: no data rows loaded.')
        return 1

    # Save aggregated CSV
    os.makedirs(args.out_dir, exist_ok=True)
    all_csv_path = os.path.join(args.out_dir, 'niqe_all.csv')
    with open(all_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'model', 'filename', 'niqe_score'])
        w.writeheader()
        w.writerows(all_rows)
    print(f'\n[Saved] Aggregated CSV → {all_csv_path}')

    # Group by (dataset, model)
    groups = defaultdict(list)
    for row in all_rows:
        key = (row['dataset'], row['model'])
        if not np.isnan(row['niqe_score']):
            groups[key].append(row['niqe_score'])

    datasets = sorted({k[0] for k in groups})
    models   = sorted({k[1] for k in groups})

    # Detect baseline model (Retinexformer_A0 or first alphabetically)
    baseline = args.baseline_model
    if baseline is None:
        for candidate in ('Retinexformer_A0', 'RetinexFormer_A0'):
            if candidate in models:
                baseline = candidate
                break
        if baseline is None:
            baseline = models[0]
    print(f'\n[Baseline model for delta computation] {baseline}')

    # Build summary table data
    # Columns: Dataset | Model | NIQE Mean ± Std | N | Δ vs baseline
    table_rows = []
    for ds in datasets:
        for mdl in models:
            scores = groups.get((ds, mdl), [])
            if not scores:
                continue
            mean_s = float(np.mean(scores))
            std_s  = float(np.std(scores))
            n      = len(scores)

            # Delta vs baseline for same dataset
            base_scores = groups.get((ds, baseline), [])
            if base_scores and mdl != baseline:
                base_mean = float(np.mean(base_scores))
                delta = mean_s - base_mean
                delta_str = f'{delta:+.4f}'
                better = '↓ better' if delta < 0 else ('↑ worse' if delta > 0 else '—')
                delta_cell = f'{delta_str} ({better})'
            elif mdl == baseline:
                delta_cell = '— (baseline)'
            else:
                delta_cell = 'N/A'

            table_rows.append({
                'dataset':   ds,
                'model':     mdl,
                'mean':      mean_s,
                'std':       std_s,
                'n':         n,
                'delta':     delta_cell,
            })

    # Print table to stdout
    print(f'\n{"Dataset":12s}  {"Model":22s}  {"NIQE Mean":>12}  {"± Std":>8}  '
          f'{"N":>5}  {"Δ vs " + baseline}')
    print('─' * 90)
    for r in table_rows:
        print(f'{r["dataset"]:12s}  {r["model"]:22s}  '
              f'{r["mean"]:>12.4f}  {r["std"]:>8.4f}  '
              f'{r["n"]:>5d}  {r["delta"]}')

    # Build markdown
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    md_lines = []
    md_lines.append('# NIQE Evaluation Summary — Non-Reference Datasets')
    md_lines.append(f'\nGenerated: {timestamp}')
    md_lines.append('\n> **Lower NIQE = better perceptual quality** (no reference image needed)')
    md_lines.append(f'\nBaseline for Δ computation: **{baseline}**')

    # One table per dataset
    for ds in datasets:
        ds_rows = [r for r in table_rows if r['dataset'] == ds]
        if not ds_rows:
            continue
        md_lines.append(f'\n## Dataset: {ds}\n')
        md_lines.append(f'| Model | NIQE Mean | ± Std | N images | Δ vs {baseline} |')
        md_lines.append('|-------|-----------|-------|----------|---------|')
        for r in ds_rows:
            md_lines.append(
                f'| {r["model"]} | {r["mean"]:.4f} | {r["std"]:.4f} '
                f'| {r["n"]} | {r["delta"]} |')

    # Overall summary table (all datasets combined)
    md_lines.append('\n## Overall Summary (all datasets combined)\n')
    md_lines.append(f'| Dataset | Model | NIQE Mean ± Std | N images | Δ vs {baseline} |')
    md_lines.append('|---------|-------|-----------------|----------|---------|')
    for r in table_rows:
        md_lines.append(
            f'| {r["dataset"]} | {r["model"]} | '
            f'{r["mean"]:.4f} ± {r["std"]:.4f} | {r["n"]} | {r["delta"]} |')

    md_lines.append('\n---')
    md_lines.append(f'\n*Note: NIQE scores computed on Y-channel (YCbCr) of enhanced images.*  ')
    md_lines.append('*Protocol: convert RGB → BGR → Y channel, crop_border=0, block_size=96×96.*')

    md_text = '\n'.join(md_lines) + '\n'

    md_path = os.path.join(args.out_dir, 'niqe_summary.md')
    with open(md_path, 'w') as f:
        f.write(md_text)
    print(f'\n[Saved] Summary → {md_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
