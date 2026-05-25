"""
Script 2: select_qualitative_examples.py
Analyse per-image metrics CSVs from A1 (FD2RT_A4) and Retinexformer runs,
apply selection criteria to find good qualitative comparison candidates, and
rank/save results.

Usage example:
  python scripts/select_qualitative_examples.py \
      --lolv1_a1      qualitative_outputs/LOL-v1/A1/metrics.csv \
      --lolv1_retinex qualitative_outputs/LOL-v1/Retinexformer_A0/metrics.csv \
      --lolv2_a1      qualitative_outputs/LOL-v2-Real/A1/metrics.csv \
      --lolv2_retinex qualitative_outputs/LOL-v2-Real/Retinexformer_A0/metrics.csv \
      --out_dir       qualitative_outputs/reports/
"""

import sys
import os
import argparse
import csv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> dict:
    """Load metrics.csv into a dict keyed by filename."""
    records = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records[row["filename"]] = {
                "psnr_rgb": float(row["psnr_rgb"]),
                "psnr_y": float(row["psnr_y"]),
                "ssim_y": float(row["ssim_y"]),
                "inference_time_ms": float(row["inference_time_ms"]),
            }
    return records


def join_metrics(a1_records: dict, retinex_records: dict) -> list:
    """Inner-join on filename; return list of merged dicts."""
    common = set(a1_records.keys()) & set(retinex_records.keys())
    rows = []
    for fname in sorted(common):
        a1 = a1_records[fname]
        rx = retinex_records[fname]
        rows.append({
            "filename": fname,
            "a1_psnr_rgb": a1["psnr_rgb"],
            "a1_psnr_y": a1["psnr_y"],
            "a1_ssim_y": a1["ssim_y"],
            "retinex_psnr_rgb": rx["psnr_rgb"],
            "retinex_psnr_y": rx["psnr_y"],
            "retinex_ssim_y": rx["ssim_y"],
            "delta_psnr": a1["psnr_y"] - rx["psnr_y"],
            "delta_ssim": a1["ssim_y"] - rx["ssim_y"],
        })
    return rows


def apply_criteria(rows: list, delta_ssim_threshold: float) -> list:
    """Return rows that pass all four selection criteria.

    A: delta_ssim >= delta_ssim_threshold
    B: a1_psnr_y  >= retinex_psnr_y
    C: retinex_psnr_y > 18.0
    D: a1_psnr_y  > 20.0
    """
    passing = []
    for row in rows:
        a = row["delta_ssim"] >= delta_ssim_threshold
        b = row["a1_psnr_y"] >= row["retinex_psnr_y"]
        c = row["retinex_psnr_y"] > 18.0
        d = row["a1_psnr_y"] > 20.0
        if a and b and c and d:
            passing.append(row)
    return passing


def rank_rows(rows: list) -> list:
    """Sort by delta_ssim descending, then a1_ssim_y descending."""
    return sorted(rows, key=lambda r: (-r["delta_ssim"], -r["a1_ssim_y"]))


def print_full_table(all_rows: list, passing_set: set, dataset_name: str):
    """Print the full ranked table for a dataset."""
    header = (
        f"{'filename':>30}  {'a1_psnr_y':>10}  {'a1_ssim_y':>10}  "
        f"{'retinex_psnr_y':>14}  {'retinex_ssim_y':>14}  "
        f"{'delta_psnr':>10}  {'delta_ssim':>10}  {'selected':>8}"
    )
    print(f"\n{'=' * 110}")
    print(f"  Dataset: {dataset_name}")
    print(header)
    print("-" * 110)
    for row in all_rows:
        selected = row["filename"] in passing_set
        print(
            f"  {row['filename']:>28}  {row['a1_psnr_y']:>10.4f}  {row['a1_ssim_y']:>10.4f}  "
            f"{row['retinex_psnr_y']:>14.4f}  {row['retinex_ssim_y']:>14.4f}  "
            f"{row['delta_psnr']:>+10.4f}  {row['delta_ssim']:>+10.4f}  "
            f"{'YES' if selected else 'no':>8}"
        )
    print(f"{'=' * 110}\n")


def select_for_dataset(
    dataset_name: str,
    a1_path: str,
    retinex_path: str,
) -> tuple:
    """Run the full selection pipeline for one dataset.

    Returns (all_ranked_rows, passing_rows, top3, best1).
    """
    print(f"\n[{dataset_name}] Loading A1 metrics from     '{a1_path}'")
    print(f"[{dataset_name}] Loading Retinex metrics from '{retinex_path}'")

    a1_records = load_csv(a1_path)
    retinex_records = load_csv(retinex_path)

    joined = join_metrics(a1_records, retinex_records)
    print(f"[{dataset_name}] Joined on {len(joined)} common filenames.")

    # --- First pass: strict criterion A (delta_ssim >= 0.010) ---------------
    threshold = 0.010
    passing = apply_criteria(joined, threshold)
    if len(passing) < 3:
        print(
            f"[{dataset_name}] Only {len(passing)} images pass with delta_ssim >= {threshold:.3f}. "
            f"Relaxing to >= 0.005 …"
        )
        threshold = 0.005
        passing = apply_criteria(joined, threshold)

    if len(passing) == 0:
        print(
            f"[{dataset_name}] STOP: 0 images pass even with relaxed threshold. "
            "Check your metric CSVs."
        )
        # Return all rows ranked but no passing images
        all_ranked = rank_rows(joined)
        passing_set = set()
        print_full_table(all_ranked, passing_set, dataset_name)
        return all_ranked, [], [], None

    # Rank passing images
    passing_ranked = rank_rows(passing)
    passing_set = {r["filename"] for r in passing_ranked}

    # Rank ALL images the same way for the full table
    all_ranked = rank_rows(joined)

    print_full_table(all_ranked, passing_set, dataset_name)

    top3 = passing_ranked[:3]
    best1 = passing_ranked[0] if passing_ranked else None

    print(f"[{dataset_name}] Passing images ({len(passing)}, threshold={threshold:.3f}):")
    for i, r in enumerate(passing_ranked):
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"  {'TOP-' + str(i+1):6s}  {r['filename']:30s}  "
            f"PSNR_Y(A1)={r['a1_psnr_y']:.4f}  SSIM_Y(A1)={r['a1_ssim_y']:.4f}  "
            f"delta_psnr={r['delta_psnr']:+.4f}  delta_ssim={r['delta_ssim']:+.4f}"
            + marker
        )

    print(f"\n[{dataset_name}] TOP 3:")
    for i, r in enumerate(top3):
        print(f"  {i+1}. {r['filename']}")
    if best1:
        print(f"[{dataset_name}] BEST 1: {best1['filename']}")

    return all_ranked, passing_ranked, top3, best1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Select best qualitative example images from metrics CSVs."
    )
    p.add_argument("--lolv1_a1", required=True,
                   help="Path to LOL-v1 A1 metrics.csv.")
    p.add_argument("--lolv1_retinex", required=True,
                   help="Path to LOL-v1 Retinexformer metrics.csv.")
    p.add_argument("--lolv2_a1", default=None,
                   help="(Optional) Path to LOL-v2-Real A1 metrics.csv.")
    p.add_argument("--lolv2_retinex", default=None,
                   help="(Optional) Path to LOL-v2-Real Retinexformer metrics.csv.")
    p.add_argument("--out_dir", default="qualitative_outputs/reports/",
                   help="Output directory for candidate_rankings.csv.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv_out_rows = []  # accumulates rows for candidate_rankings.csv

    # -----------------------------------------------------------------------
    # LOL-v1
    # -----------------------------------------------------------------------
    all_v1, passing_v1, top3_v1, best1_v1 = select_for_dataset(
        dataset_name="LOL-v1",
        a1_path=args.lolv1_a1,
        retinex_path=args.lolv1_retinex,
    )

    # Build passing set for rank tagging
    passing_fnames_v1 = {r["filename"] for r in passing_v1}
    for rank_idx, row in enumerate(all_v1):
        passes = row["filename"] in passing_fnames_v1
        rank_val = (
            passing_v1.index(row) + 1
            if passes
            else None
        )
        csv_out_rows.append({
            "dataset": "LOL-v1",
            "filename": row["filename"],
            "a1_psnr_y": row["a1_psnr_y"],
            "a1_ssim_y": row["a1_ssim_y"],
            "retinex_psnr_y": row["retinex_psnr_y"],
            "retinex_ssim_y": row["retinex_ssim_y"],
            "delta_psnr": row["delta_psnr"],
            "delta_ssim": row["delta_ssim"],
            "passes_criteria": passes,
            "rank": rank_val if rank_val is not None else "",
        })

    # -----------------------------------------------------------------------
    # LOL-v2-Real (optional)
    # -----------------------------------------------------------------------
    if args.lolv2_a1 and args.lolv2_retinex:
        all_v2, passing_v2, top3_v2, best1_v2 = select_for_dataset(
            dataset_name="LOL-v2-Real",
            a1_path=args.lolv2_a1,
            retinex_path=args.lolv2_retinex,
        )
        passing_fnames_v2 = {r["filename"] for r in passing_v2}
        for row in all_v2:
            passes = row["filename"] in passing_fnames_v2
            rank_val = (
                passing_v2.index(row) + 1
                if passes
                else None
            )
            csv_out_rows.append({
                "dataset": "LOL-v2-Real",
                "filename": row["filename"],
                "a1_psnr_y": row["a1_psnr_y"],
                "a1_ssim_y": row["a1_ssim_y"],
                "retinex_psnr_y": row["retinex_psnr_y"],
                "retinex_ssim_y": row["retinex_ssim_y"],
                "delta_psnr": row["delta_psnr"],
                "delta_ssim": row["delta_ssim"],
                "passes_criteria": passes,
                "rank": rank_val if rank_val is not None else "",
            })
    else:
        if args.lolv2_a1 or args.lolv2_retinex:
            print(
                "[WARNING] LOL-v2-Real: both --lolv2_a1 and --lolv2_retinex must be "
                "provided together. Skipping LOL-v2-Real."
            )

    # -----------------------------------------------------------------------
    # Save candidate_rankings.csv
    # -----------------------------------------------------------------------
    out_csv = os.path.join(args.out_dir, "candidate_rankings.csv")
    fieldnames = [
        "dataset", "filename",
        "a1_psnr_y", "a1_ssim_y",
        "retinex_psnr_y", "retinex_ssim_y",
        "delta_psnr", "delta_ssim",
        "passes_criteria", "rank",
    ]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_out_rows)
    print(f"\n[Output] Saved candidate rankings to '{out_csv}'.")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL SELECTION SUMMARY")
    print("=" * 70)
    if best1_v1:
        print(f"  LOL-v1  best image : {best1_v1['filename']}")
        for i, r in enumerate(top3_v1):
            print(f"           top-{i+1}     : {r['filename']}")
    else:
        print("  LOL-v1 : no candidates passed criteria.")

    if args.lolv2_a1 and args.lolv2_retinex:
        if best1_v2:
            print(f"  LOL-v2  best image : {best1_v2['filename']}")
            for i, r in enumerate(top3_v2):
                print(f"           top-{i+1}     : {r['filename']}")
        else:
            print("  LOL-v2-Real : no candidates passed criteria.")
    print("=" * 70)


if __name__ == "__main__":
    main()
