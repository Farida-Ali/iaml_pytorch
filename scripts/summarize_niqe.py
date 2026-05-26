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


def _safe_float(row, key):
    try:
        return float(row[key])
    except (ValueError, KeyError, TypeError):
        return float('nan')


def load_csv(path: str) -> list:
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({
                'dataset':      row.get('dataset', ''),
                'model':        row.get('model', ''),
                'filename':     row.get('filename', ''),
                'niqe_score':   _safe_float(row, 'niqe_score'),
                'brisque_score': _safe_float(row, 'brisque_score'),
                'piqe_score':   _safe_float(row, 'piqe_score'),
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
    fieldnames = ['dataset', 'model', 'filename', 'niqe_score', 'brisque_score', 'piqe_score']
    with open(all_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    print(f'\n[Saved] Aggregated CSV → {all_csv_path}')

    # Group by (dataset, model) — separate lists per metric
    groups_niqe    = defaultdict(list)
    groups_brisque = defaultdict(list)
    groups_piqe    = defaultdict(list)
    for row in all_rows:
        key = (row['dataset'], row['model'])
        if not np.isnan(row['niqe_score']):
            groups_niqe[key].append(row['niqe_score'])
        if not np.isnan(row.get('brisque_score', float('nan'))):
            groups_brisque[key].append(row['brisque_score'])
        if not np.isnan(row.get('piqe_score', float('nan'))):
            groups_piqe[key].append(row['piqe_score'])
    groups = groups_niqe  # keep for baseline detection

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

    def mean_std_str(score_list):
        if score_list:
            return float(np.mean(score_list)), float(np.std(score_list)), len(score_list)
        return float('nan'), float('nan'), 0

    # Build summary table data
    table_rows = []
    for ds in datasets:
        for mdl in models:
            niqe_list    = groups_niqe.get((ds, mdl), [])
            brisque_list = groups_brisque.get((ds, mdl), [])
            piqe_list    = groups_piqe.get((ds, mdl), [])

            if not niqe_list and not brisque_list and not piqe_list:
                continue

            n = max(len(niqe_list), len(brisque_list), len(piqe_list))
            mean_niqe, std_niqe, _       = mean_std_str(niqe_list)
            mean_brisque, std_brisque, _ = mean_std_str(brisque_list)
            mean_piqe, std_piqe, _       = mean_std_str(piqe_list)

            # Delta vs baseline (NIQE only)
            base_scores = groups_niqe.get((ds, baseline), [])
            if base_scores and mdl != baseline and niqe_list:
                base_mean = float(np.mean(base_scores))
                delta = mean_niqe - base_mean
                delta_str = f'{delta:+.4f}'
                better = '↓ better' if delta < 0 else ('↑ worse' if delta > 0 else '—')
                delta_cell = f'{delta_str} ({better})'
            elif mdl == baseline:
                delta_cell = '— (baseline)'
            else:
                delta_cell = 'N/A'

            table_rows.append({
                'dataset':      ds,
                'model':        mdl,
                'mean':         mean_niqe,
                'std':          std_niqe,
                'mean_brisque': mean_brisque,
                'std_brisque':  std_brisque,
                'mean_piqe':    mean_piqe,
                'std_piqe':     std_piqe,
                'n':            n,
                'delta':        delta_cell,
            })

    def fmt_metric(m, s):
        if np.isnan(m):
            return f'{"n/a":>12}  {"":>8}'
        return f'{m:>12.4f}  {s:>8.4f}'

    # Print table to stdout
    print(f'\n{"Dataset":12s}  {"Model":22s}  {"NIQE Mean":>12}  {"±Std":>8}  '
          f'{"BRISQUE":>12}  {"±Std":>8}  {"PIQE":>12}  {"±Std":>8}  '
          f'{"N":>5}  {"Δ vs " + baseline}')
    print('─' * 130)
    for r in table_rows:
        print(f'{r["dataset"]:12s}  {r["model"]:22s}  '
              f'{fmt_metric(r["mean"], r["std"])}  '
              f'{fmt_metric(r["mean_brisque"], r["std_brisque"])}  '
              f'{fmt_metric(r["mean_piqe"], r["std_piqe"])}  '
              f'{r["n"]:>5d}  {r["delta"]}')

    # Build markdown
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    md_lines = []
    md_lines.append('# NIQE Evaluation Summary — Non-Reference Datasets')
    md_lines.append(f'\nGenerated: {timestamp}')
    md_lines.append('\n> **Lower NIQE = better perceptual quality** (no reference image needed)')
    md_lines.append(f'\nBaseline for Δ computation: **{baseline}**')

    def md_metric(m, s):
        return f'{m:.4f} ± {s:.4f}' if not np.isnan(m) else 'n/a'

    # One table per dataset
    for ds in datasets:
        ds_rows = [r for r in table_rows if r['dataset'] == ds]
        if not ds_rows:
            continue
        md_lines.append(f'\n## Dataset: {ds}\n')
        md_lines.append(f'| Model | NIQE Mean ± Std | BRISQUE Mean ± Std | PIQE Mean ± Std | N images | Δ NIQE vs {baseline} |')
        md_lines.append('|-------|-----------------|--------------------|-----------------|---------:|------|')
        for r in ds_rows:
            md_lines.append(
                f'| {r["model"]} '
                f'| {md_metric(r["mean"], r["std"])} '
                f'| {md_metric(r["mean_brisque"], r["std_brisque"])} '
                f'| {md_metric(r["mean_piqe"], r["std_piqe"])} '
                f'| {r["n"]} | {r["delta"]} |')

    # Overall summary table (all datasets combined)
    md_lines.append('\n## Overall Summary (all datasets combined)\n')
    md_lines.append(f'| Dataset | Model | NIQE Mean ± Std | BRISQUE Mean ± Std | PIQE Mean ± Std | N images | Δ NIQE vs {baseline} |')
    md_lines.append('|---------|-------|-----------------|--------------------|-----------------|---------:|------|')
    for r in table_rows:
        md_lines.append(
            f'| {r["dataset"]} | {r["model"]} '
            f'| {md_metric(r["mean"], r["std"])} '
            f'| {md_metric(r["mean_brisque"], r["std_brisque"])} '
            f'| {md_metric(r["mean_piqe"], r["std_piqe"])} '
            f'| {r["n"]} | {r["delta"]} |')

    md_lines.append('\n---')
    md_lines.append(f'\n*Note: All scores are no-reference (lower = better perceptual quality).*  ')
    md_lines.append('*NIQE: Y-channel (YCbCr), crop_border=0, block_size=96×96 (basicsr implementation).*  ')
    md_lines.append('*BRISQUE / PIQE: piq library (pip install piq). Shown as n/a if not installed.*')

    md_text = '\n'.join(md_lines) + '\n'

    md_path = os.path.join(args.out_dir, 'niqe_summary.md')
    with open(md_path, 'w') as f:
        f.write(md_text)
    print(f'\n[Saved] Summary → {md_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
