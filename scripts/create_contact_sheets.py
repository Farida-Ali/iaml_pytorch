"""
Script 3: create_contact_sheets.py
Create a contact sheet figure with 3 rows (one per candidate image) x 4 columns
(Input | Retinexformer | Ours | GT) for qualitative comparison in the paper.

Usage example:
  python scripts/create_contact_sheets.py \
      --dataset_name "LOL-v1" \
      --image_ids "00750.png,00723.png,00701.png" \
      --input_dir   qualitative_outputs/LOL-v1/input/ \
      --gt_dir      qualitative_outputs/LOL-v1/gt/ \
      --a1_dir      qualitative_outputs/LOL-v1/A1/ \
      --retinex_dir qualitative_outputs/LOL-v1/Retinexformer_A0/ \
      --a1_metrics      qualitative_outputs/LOL-v1/A1/metrics.csv \
      --retinex_metrics qualitative_outputs/LOL-v1/Retinexformer_A0/metrics.csv \
      --out_dir qualitative_outputs/figures/
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
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COL_WIDTH_PX = 600      # display pixels per column
N_COLS = 4              # Input | Retinexformer | Ours | GT
DPI = 300
MIN_FONT_PT = 10

COL_LABELS = ["Input", "Retinexformer", "Ours", "GT"]
HEADER_BG = "#333333"
HEADER_FG = "white"


def load_csv_metrics(path: str) -> dict:
    """Return dict: filename -> {psnr_y, ssim_y}."""
    records = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            records[row["filename"]] = {
                "psnr_y": float(row["psnr_y"]),
                "ssim_y": float(row["ssim_y"]),
            }
    return records


def load_image_np(directory: str, filename: str) -> np.ndarray:
    """Load image from directory/filename as uint8 RGB numpy array."""
    path = os.path.join(directory, filename)
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    """Resize image to target_w pixels wide, preserving aspect ratio."""
    h, w = img.shape[:2]
    new_h = int(round(h * target_w / w))
    pil = Image.fromarray(img).resize((target_w, new_h), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Create 3×4 contact sheet for qualitative paper figure."
    )
    p.add_argument("--dataset_name", required=True,
                   help='Dataset label, e.g. "LOL-v1".')
    p.add_argument("--image_ids", required=True,
                   help="Comma-separated list of exactly 3 filenames.")
    p.add_argument("--input_dir", required=True,
                   help="Directory of input LQ images.")
    p.add_argument("--gt_dir", required=True,
                   help="Directory of GT images.")
    p.add_argument("--a1_dir", required=True,
                   help="Directory of A1 (Ours) enhanced images.")
    p.add_argument("--retinex_dir", required=True,
                   help="Directory of Retinexformer enhanced images.")
    p.add_argument("--a1_metrics", required=True,
                   help="Path to A1 metrics.csv.")
    p.add_argument("--retinex_metrics", required=True,
                   help="Path to Retinexformer metrics.csv.")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for the contact sheet PNG and PDF.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    image_ids = [s.strip() for s in args.image_ids.split(",")]
    if len(image_ids) != 3:
        print(
            f"[ERROR] --image_ids must contain exactly 3 filenames "
            f"(got {len(image_ids)}: {image_ids})."
        )
        sys.exit(1)

    a1_metrics = load_csv_metrics(args.a1_metrics)
    retinex_metrics = load_csv_metrics(args.retinex_metrics)

    # -----------------------------------------------------------------------
    # Load and resize all images
    # Ordered columns: Input | Retinexformer | Ours | GT
    # -----------------------------------------------------------------------
    rows_data = []
    for fname in image_ids:
        inp_img = resize_to_width(load_image_np(args.input_dir, fname), COL_WIDTH_PX)
        ret_img = resize_to_width(load_image_np(args.retinex_dir, fname), COL_WIDTH_PX)
        a1_img = resize_to_width(load_image_np(args.a1_dir, fname), COL_WIDTH_PX)
        gt_img = resize_to_width(load_image_np(args.gt_dir, fname), COL_WIDTH_PX)

        a1_m = a1_metrics.get(fname, {"psnr_y": float("nan"), "ssim_y": float("nan")})
        rx_m = retinex_metrics.get(fname, {"psnr_y": float("nan"), "ssim_y": float("nan")})

        delta_psnr = a1_m["psnr_y"] - rx_m["psnr_y"]
        delta_ssim = a1_m["ssim_y"] - rx_m["ssim_y"]

        row_label = (
            f"{fname}   "
            f"Retinex: {rx_m['psnr_y']:.2f} dB / {rx_m['ssim_y']:.3f}  |  "
            f"Ours: {a1_m['psnr_y']:.2f} dB / {a1_m['ssim_y']:.3f}  |  "
            f"ΔPSNR: {delta_psnr:+.2f}  |  ΔSSIM: {delta_ssim:+.3f}"
        )

        rows_data.append({
            "filename": fname,
            "images": [inp_img, ret_img, a1_img, gt_img],
            "label": row_label,
            "img_h": inp_img.shape[0],
        })

    # -----------------------------------------------------------------------
    # Figure layout
    #
    # We use a GridSpec where each logical "image row" is preceded by a
    # thin header bar row.  Structure:
    #   row 0 : column header bar   (height = header_h_px)
    #   row 1 : row 0 header bar    (height = row_header_h_px)
    #   row 2 : row 0 images
    #   row 3 : row 1 header bar
    #   row 4 : row 1 images
    #   row 5 : row 2 header bar
    #   row 6 : row 2 images
    # -----------------------------------------------------------------------
    n_rows = len(rows_data)   # 3
    total_w_px = N_COLS * COL_WIDTH_PX  # 2400 px
    fig_w_in = total_w_px / DPI         # 8.0 inches at 300 DPI

    # Estimate per-image heights in inches
    img_heights_in = [r["img_h"] / DPI for r in rows_data]

    col_header_h_in = 0.25          # bar for column labels at very top
    row_header_h_in = 0.20          # bar above each image row

    gs_heights = [col_header_h_in]
    for h in img_heights_in:
        gs_heights.append(row_header_h_in)
        gs_heights.append(h)

    total_h_in = sum(gs_heights) + 0.1  # small buffer

    fig = plt.figure(figsize=(fig_w_in, total_h_in), dpi=DPI)
    n_gs_rows = 1 + n_rows * 2          # 1 col_header + (row_header + img) * 3
    gs = GridSpec(
        n_gs_rows, N_COLS,
        figure=fig,
        height_ratios=gs_heights,
        hspace=0.0,
        wspace=0.0,
    )

    font_size = max(MIN_FONT_PT, fig_w_in * 1.3)

    # -----------------------------------------------------------------------
    # Column header bar (row 0 of GridSpec)
    # -----------------------------------------------------------------------
    ax_col_header = fig.add_subplot(gs[0, :])
    ax_col_header.set_facecolor(HEADER_BG)
    ax_col_header.set_xlim(0, N_COLS)
    ax_col_header.set_ylim(0, 1)
    ax_col_header.axis("off")
    for ci, label in enumerate(COL_LABELS):
        ax_col_header.text(
            ci + 0.5, 0.5, label,
            ha="center", va="center",
            fontsize=font_size,
            fontweight="bold",
            color=HEADER_FG,
            transform=ax_col_header.transData,
        )

    # -----------------------------------------------------------------------
    # Image rows
    # -----------------------------------------------------------------------
    for ri, row_data in enumerate(rows_data):
        gs_row_header = 1 + ri * 2
        gs_row_img = gs_row_header + 1

        # -- Row header bar --------------------------------------------------
        ax_row_header = fig.add_subplot(gs[gs_row_header, :])
        ax_row_header.set_facecolor(HEADER_BG)
        ax_row_header.axis("off")
        ax_row_header.text(
            0.5, 0.5, row_data["label"],
            ha="center", va="center",
            fontsize=max(MIN_FONT_PT - 1, font_size - 1),
            fontweight="bold",
            color=HEADER_FG,
            transform=ax_row_header.transAxes,
        )

        # -- Four image cells ------------------------------------------------
        for ci, img in enumerate(row_data["images"]):
            ax = fig.add_subplot(gs[gs_row_img, ci])
            ax.imshow(img, aspect="auto", interpolation="lanczos")
            ax.axis("off")

    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)

    # -----------------------------------------------------------------------
    # Save PNG and PDF
    # -----------------------------------------------------------------------
    safe_name = args.dataset_name.replace(" ", "_")
    png_path = os.path.join(args.out_dir, f"candidate_sheet_{safe_name}.png")
    pdf_path = os.path.join(args.out_dir, f"candidate_sheet_{safe_name}.pdf")

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", format="png")
    print(f"[Saved] PNG -> '{png_path}'")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", format="pdf")
    print(f"[Saved] PDF -> '{pdf_path}'")

    plt.close(fig)


if __name__ == "__main__":
    main()
