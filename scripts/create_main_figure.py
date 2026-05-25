"""
Script 4: create_main_figure.py
Create the main qualitative comparison figure for the AAAI 2027 paper.

Layout (per dataset row):
  Columns: Input | Retinexformer | Ours | GT | Zoom
  Below images (cols 0-3): metric caption "XX.XX dB / X.XXX" (or "Input"/"GT").
  Left margin: dataset row label.
  Zoom column: 128×128 crop displayed at 384×384 px with a cyan border.
  Cyan rectangle overlay on full images showing crop location.

Saved (both 1-column and 2-column versions):
  qualitative_outputs/figures/qualitative_comparison_1col.png / .pdf
  qualitative_outputs/figures/qualitative_comparison_2col.png / .pdf

Usage example (single dataset):
  python scripts/create_main_figure.py \
      --lolv1_image_id    "00750.png" \
      --lolv1_input_dir   qualitative_outputs/LOL-v1/input/ \
      --lolv1_gt_dir      qualitative_outputs/LOL-v1/gt/ \
      --lolv1_a1_dir      qualitative_outputs/LOL-v1/A1/ \
      --lolv1_retinex_dir qualitative_outputs/LOL-v1/Retinexformer_A0/ \
      --lolv1_a1_metrics      qualitative_outputs/LOL-v1/A1/metrics.csv \
      --lolv1_retinex_metrics qualitative_outputs/LOL-v1/Retinexformer_A0/metrics.csv \
      --out_dir qualitative_outputs/figures/
"""

import sys
import os
import argparse
import csv
import math

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZOOM_WINDOW = 64        # sliding window size for SSIM-based crop search
ZOOM_STRIDE = 16        # stride of the sliding window search
CROP_SIZE = 128         # final crop taken around the best window centre
ZOOM_DISPLAY = 384      # display size in pixels (3× of CROP_SIZE)

CYAN = "#00FFFF"
RECT_LINEWIDTH = 3      # cyan rectangle border width in points

DPI = 300

# Column layout
N_COLS = 5              # Input | Retinexformer | Ours | GT | Zoom
COL_LABELS = ["Input", "Retinexformer", "Ours", "GT", "Zoom"]

# Figure widths (inches)
WIDTH_1COL_IN = 3.5     # single-column journal format
WIDTH_2COL_IN = 7.0     # double-column journal format


# ---------------------------------------------------------------------------
# CSV / Image helpers
# ---------------------------------------------------------------------------

def load_metrics_csv(path: str) -> dict:
    """Return dict: filename -> {psnr_y, ssim_y}."""
    records = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            records[row["filename"]] = {
                "psnr_y": float(row["psnr_y"]),
                "ssim_y": float(row["ssim_y"]),
            }
    return records


def load_image_f32(directory: str, filename: str) -> np.ndarray:
    """Load image as float32 [0,1] numpy array (H,W,3)."""
    path = os.path.join(directory, filename)
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_image_uint8(directory: str, filename: str) -> np.ndarray:
    """Load image as uint8 numpy array (H,W,3)."""
    path = os.path.join(directory, filename)
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Zoom-crop selection via sliding-window SSIM delta
# ---------------------------------------------------------------------------

def patch_ssim(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """Compute mean SSIM between two grayscale float32 patches [H,W]."""
    from skimage.metrics import structural_similarity as sk_ssim
    return float(
        sk_ssim(patch_a, patch_b, data_range=1.0, win_size=11,
                gaussian_weights=True)
    )


def rgb_to_gray_f32(img_f32: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [H,W,3] to grayscale [H,W] via standard weights."""
    return (
        0.2989 * img_f32[:, :, 0]
        + 0.5870 * img_f32[:, :, 1]
        + 0.1140 * img_f32[:, :, 2]
    )


def find_best_crop(
    a1_f32: np.ndarray,
    retinex_f32: np.ndarray,
    gt_f32: np.ndarray,
    win: int = ZOOM_WINDOW,
    stride: int = ZOOM_STRIDE,
    crop: int = CROP_SIZE,
) -> tuple:
    """Find 128×128 crop region where A1 most outperforms Retinexformer.

    Steps:
    1. Slide a win×win window with given stride over all images.
    2. At each position compute SSIM(a1_patch, gt_patch) and
       SSIM(retinex_patch, gt_patch); take delta = ssim_a1 - ssim_retinex.
    3. Pick position with maximum delta.
    4. Centre a crop×crop region on that window's centre (clamped to bounds).

    Returns (y0, y1, x0, x1) for the crop.
    """
    H, W = gt_f32.shape[:2]
    a1_gray = rgb_to_gray_f32(a1_f32)
    ret_gray = rgb_to_gray_f32(retinex_f32)
    gt_gray = rgb_to_gray_f32(gt_f32)

    best_delta = -np.inf
    best_cy, best_cx = H // 2, W // 2   # fallback to image centre

    ys = range(0, H - win + 1, stride)
    xs = range(0, W - win + 1, stride)

    if len(ys) == 0 or len(xs) == 0:
        # Image smaller than window — just use full image centre
        cy, cx = H // 2, W // 2
    else:
        for y in ys:
            for x in xs:
                patch_a1 = a1_gray[y: y + win, x: x + win]
                patch_rx = ret_gray[y: y + win, x: x + win]
                patch_gt = gt_gray[y: y + win, x: x + win]

                ssim_a1 = patch_ssim(patch_a1, patch_gt)
                ssim_rx = patch_ssim(patch_rx, patch_gt)
                delta = ssim_a1 - ssim_rx

                if delta > best_delta:
                    best_delta = delta
                    best_cy = y + win // 2
                    best_cx = x + win // 2

        cy, cx = best_cy, best_cx

    # Centre crop×crop on (cy, cx), clamp to image bounds
    half = crop // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = y0 + crop
    x0_adj = x0
    if y1 > H:
        y1 = H
        y0 = max(0, H - crop)
    x1 = x0_adj + crop
    if x1 > W:
        x1 = W
        x0_adj = max(0, W - crop)

    return int(y0), int(y1), int(x0_adj), int(x1)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_cyan_rect(ax, y0: int, y1: int, x0: int, x1: int):
    """Draw a 3-pixel-wide cyan rectangle on an imshow axes."""
    rect = patches.Rectangle(
        (x0 - 0.5, y0 - 0.5),
        x1 - x0, y1 - y0,
        linewidth=RECT_LINEWIDTH,
        edgecolor=CYAN,
        facecolor="none",
    )
    ax.add_patch(rect)


def make_zoom_display(crop_img: np.ndarray) -> np.ndarray:
    """Resize a crop to ZOOM_DISPLAY×ZOOM_DISPLAY pixels (nearest-neighbour)."""
    pil = Image.fromarray(
        (crop_img * 255.0).clip(0, 255).astype(np.uint8)
    )
    pil = pil.resize((ZOOM_DISPLAY, ZOOM_DISPLAY), Image.NEAREST)
    return np.array(pil, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Core figure builder
# ---------------------------------------------------------------------------

class DatasetRow:
    """All data needed to draw one dataset row in the figure."""

    def __init__(
        self,
        label: str,
        image_id: str,
        input_dir: str,
        gt_dir: str,
        a1_dir: str,
        retinex_dir: str,
        a1_metrics_path: str,
        retinex_metrics_path: str,
    ):
        self.label = label
        self.image_id = image_id

        # Load images as float32 [0,1]
        self.inp_f32 = load_image_f32(input_dir, image_id)
        self.gt_f32 = load_image_f32(gt_dir, image_id)
        self.a1_f32 = load_image_f32(a1_dir, image_id)
        self.retinex_f32 = load_image_f32(retinex_dir, image_id)

        # Load metrics
        a1_m = load_metrics_csv(a1_metrics_path)
        rx_m = load_metrics_csv(retinex_metrics_path)
        self.a1_psnr_y = a1_m[image_id]["psnr_y"]
        self.a1_ssim_y = a1_m[image_id]["ssim_y"]
        self.retinex_psnr_y = rx_m[image_id]["psnr_y"]
        self.retinex_ssim_y = rx_m[image_id]["ssim_y"]

        # Find best zoom crop
        print(
            f"  [{label}] Searching best zoom crop for '{image_id}' …"
        )
        self.y0, self.y1, self.x0, self.x1 = find_best_crop(
            self.a1_f32, self.retinex_f32, self.gt_f32
        )
        print(
            f"  [{label}] Crop region: y=[{self.y0},{self.y1}] x=[{self.x0},{self.x1}]"
        )

        # Build zoom display image
        crop_a1 = self.a1_f32[self.y0: self.y1, self.x0: self.x1]
        zoom_pil = Image.fromarray(
            (crop_a1 * 255.0).clip(0, 255).astype(np.uint8)
        ).resize((ZOOM_DISPLAY, ZOOM_DISPLAY), Image.NEAREST)
        self.zoom_np = np.array(zoom_pil, dtype=np.uint8)

    def get_crops(self):
        """Return (inp_crop, retinex_crop, a1_crop, gt_crop) as uint8 np arrays."""
        def _crop_u8(f32):
            return (
                (f32[self.y0: self.y1, self.x0: self.x1] * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )
        return _crop_u8(self.inp_f32), _crop_u8(self.retinex_f32), \
               _crop_u8(self.a1_f32), _crop_u8(self.gt_f32)


def build_figure(dataset_rows: list, fig_width_in: float) -> plt.Figure:
    """Build the complete qualitative comparison figure.

    Parameters
    ----------
    dataset_rows : list of DatasetRow
    fig_width_in : figure width in inches (3.5 for 1-col, 7.0 for 2-col)

    Returns
    -------
    matplotlib Figure
    """
    n_rows = len(dataset_rows)

    # ------------------------------------------------------------------
    # We need per-row image aspect ratios to size the rows properly.
    # All full images in a row share the same H/W (only the zoom differs).
    # The zoom is always ZOOM_DISPLAY × ZOOM_DISPLAY px → square.
    # ------------------------------------------------------------------
    # We map pixel widths to figure-inch widths:
    #   figure has N_COLS columns each of equal width (fig_width_in / N_COLS)
    #   Each image in cols 0-3 has the same native W.
    #   The aspect ratio of a row = H / W (for the full images).
    # ------------------------------------------------------------------
    col_w_in = fig_width_in / N_COLS

    # Compute row heights (in figure inches)
    row_img_heights = []
    for dr in dataset_rows:
        H, W = dr.inp_f32.shape[:2]
        row_h_in = col_w_in * H / W
        row_img_heights.append(row_h_in)

    # Caption height below each image row, and column-header height
    caption_h_in = col_w_in * 0.12
    col_header_h_in = col_w_in * 0.10

    # Total figure height
    total_h_in = col_header_h_in + n_rows * (max(row_img_heights) + caption_h_in)

    fig = plt.figure(figsize=(fig_width_in, total_h_in), dpi=DPI)

    # GridSpec: 1 col-header row + n_rows * (img row + caption row)
    gs_n_rows = 1 + n_rows * 2
    gs_heights = [col_header_h_in]
    for rh in row_img_heights:
        gs_heights.append(rh)
        gs_heights.append(caption_h_in)

    gs = GridSpec(
        gs_n_rows, N_COLS,
        figure=fig,
        height_ratios=gs_heights,
        hspace=0.02,
        wspace=0.01,
        left=0.08, right=0.99, top=0.99, bottom=0.01,
    )

    font_size = max(5.0, fig_width_in * 1.5)

    # ------------------------------------------------------------------
    # Column headers row (gs row 0)
    # ------------------------------------------------------------------
    for ci, label in enumerate(COL_LABELS):
        ax = fig.add_subplot(gs[0, ci])
        ax.set_facecolor("#333333")
        ax.text(
            0.5, 0.5, label,
            ha="center", va="center",
            fontsize=font_size,
            fontweight="bold",
            color="white",
            transform=ax.transAxes,
        )
        ax.axis("off")

    # ------------------------------------------------------------------
    # Dataset rows
    # ------------------------------------------------------------------
    for ri, dr in enumerate(dataset_rows):
        gs_img_row = 1 + ri * 2
        gs_cap_row = gs_img_row + 1

        # Full images: Input | Retinexformer | Ours | GT
        imgs_f32 = [dr.inp_f32, dr.retinex_f32, dr.a1_f32, dr.gt_f32]
        captions = [
            "Input",
            f"{dr.retinex_psnr_y:.2f} dB / {dr.retinex_ssim_y:.3f}",
            f"{dr.a1_psnr_y:.2f} dB / {dr.a1_ssim_y:.3f}",
            "GT",
        ]

        for ci in range(4):
            # Image cell
            ax = fig.add_subplot(gs[gs_img_row, ci])
            ax.imshow(imgs_f32[ci], aspect="auto", interpolation="lanczos")
            # Cyan crop rectangle on all four full images
            draw_cyan_rect(ax, dr.y0, dr.y1, dr.x0, dr.x1)
            ax.axis("off")

            # Row label on leftmost column
            if ci == 0:
                ax.set_ylabel(
                    dr.label,
                    fontsize=font_size,
                    fontweight="bold",
                    rotation=90,
                    va="center",
                    labelpad=2,
                )
                ax.yaxis.set_visible(True)
                ax.tick_params(left=False, labelleft=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

            # Caption cell
            ax_cap = fig.add_subplot(gs[gs_cap_row, ci])
            ax_cap.set_facecolor("white")
            ax_cap.text(
                0.5, 0.5, captions[ci],
                ha="center", va="center",
                fontsize=max(4.0, font_size - 1.5),
                color="black",
                transform=ax_cap.transAxes,
            )
            ax_cap.axis("off")

        # Zoom column (col 4)
        # Show A1 crop at ZOOM_DISPLAY × ZOOM_DISPLAY with cyan border
        ax_zoom = fig.add_subplot(gs[gs_img_row, 4])
        zoom_display = make_zoom_display(dr.a1_f32[dr.y0: dr.y1, dr.x0: dr.x1])
        ax_zoom.imshow(zoom_display, aspect="equal", interpolation="nearest")
        # Thin cyan border around zoom panel
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor(CYAN)
            spine.set_linewidth(2)
            spine.set_visible(True)
        ax_zoom.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Empty zoom caption cell
        ax_zoom_cap = fig.add_subplot(gs[gs_cap_row, 4])
        ax_zoom_cap.axis("off")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Create main qualitative comparison figure (1-col and 2-col versions)."
    )

    # LOL-v1 (required)
    p.add_argument("--lolv1_image_id", required=True,
                   help="Filename of the best LOL-v1 example image.")
    p.add_argument("--lolv1_input_dir", required=True,
                   help="LOL-v1 input (LQ) image directory.")
    p.add_argument("--lolv1_gt_dir", required=True,
                   help="LOL-v1 GT image directory.")
    p.add_argument("--lolv1_a1_dir", required=True,
                   help="LOL-v1 A1 (Ours) enhanced image directory.")
    p.add_argument("--lolv1_retinex_dir", required=True,
                   help="LOL-v1 Retinexformer enhanced image directory.")
    p.add_argument("--lolv1_a1_metrics", required=True,
                   help="Path to LOL-v1 A1 metrics.csv.")
    p.add_argument("--lolv1_retinex_metrics", required=True,
                   help="Path to LOL-v1 Retinexformer metrics.csv.")

    # LOL-v2-Real (optional)
    p.add_argument("--lolv2_image_id", default=None,
                   help="(Optional) Filename of the best LOL-v2-Real example image.")
    p.add_argument("--lolv2_input_dir", default=None)
    p.add_argument("--lolv2_gt_dir", default=None)
    p.add_argument("--lolv2_a1_dir", default=None)
    p.add_argument("--lolv2_retinex_dir", default=None)
    p.add_argument("--lolv2_a1_metrics", default=None)
    p.add_argument("--lolv2_retinex_metrics", default=None)

    p.add_argument("--out_dir", required=True,
                   help="Output directory for figures.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build dataset rows
    # ------------------------------------------------------------------
    print("[Setup] Loading LOL-v1 data …")
    rows = [
        DatasetRow(
            label="LOL-v1",
            image_id=args.lolv1_image_id,
            input_dir=args.lolv1_input_dir,
            gt_dir=args.lolv1_gt_dir,
            a1_dir=args.lolv1_a1_dir,
            retinex_dir=args.lolv1_retinex_dir,
            a1_metrics_path=args.lolv1_a1_metrics,
            retinex_metrics_path=args.lolv1_retinex_metrics,
        )
    ]

    lolv2_args = [
        args.lolv2_image_id, args.lolv2_input_dir, args.lolv2_gt_dir,
        args.lolv2_a1_dir, args.lolv2_retinex_dir,
        args.lolv2_a1_metrics, args.lolv2_retinex_metrics,
    ]
    if any(a is not None for a in lolv2_args):
        if all(a is not None for a in lolv2_args):
            print("[Setup] Loading LOL-v2-Real data …")
            rows.append(
                DatasetRow(
                    label="LOL-v2-Real",
                    image_id=args.lolv2_image_id,
                    input_dir=args.lolv2_input_dir,
                    gt_dir=args.lolv2_gt_dir,
                    a1_dir=args.lolv2_a1_dir,
                    retinex_dir=args.lolv2_retinex_dir,
                    a1_metrics_path=args.lolv2_a1_metrics,
                    retinex_metrics_path=args.lolv2_retinex_metrics,
                )
            )
        else:
            print(
                "[WARNING] LOL-v2-Real: all seven --lolv2_* arguments must be provided "
                "together. Skipping LOL-v2-Real row."
            )

    # ------------------------------------------------------------------
    # Render and save 1-column and 2-column figures
    # ------------------------------------------------------------------
    for col_label, width_in in [("1col", WIDTH_1COL_IN), ("2col", WIDTH_2COL_IN)]:
        print(f"\n[Figure] Building {col_label} figure ({width_in:.1f} inches wide) …")
        fig = build_figure(rows, width_in)

        for fmt in ("png", "pdf"):
            out_path = os.path.join(
                args.out_dir,
                f"qualitative_comparison_{col_label}.{fmt}",
            )
            fig.savefig(out_path, dpi=DPI, bbox_inches="tight", format=fmt)
            print(f"  [Saved] {out_path}")

        plt.close(fig)

    print("\n[Done] All figures saved.")


if __name__ == "__main__":
    main()
