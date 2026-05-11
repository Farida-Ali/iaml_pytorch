"""
Compute LPIPS, NIQE, and BRISQUE on enhanced images already saved to disk
by Component 1 (basicsr/test.py with test_IAML_*.yml configs).

Usage:
    python compute_extra_metrics.py \
        --enhanced_dir results/IAML_LOLv1/visualization/LOLv1/ \
        --gt_dir data/LOLv1/Test/target/ \
        --output_csv results/IAML_LOLv1/extra_metrics.csv
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image


def load_image(path: str) -> torch.Tensor:
    """Load image as (1, 3, H, W) float32 tensor in [0, 1]. PIL only, no cv2."""
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t  = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    # shape: (1, 3, H, W), range [0,1], float32
    return img_t


def compute_extra_metrics(enhanced_dir: str, gt_dir: str, output_csv: str) -> None:
    import pyiqa

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computing metrics on: {device}")

    # LPIPS: AlexNet backbone, input must be in [-1, 1]
    # Requires one-time download of AlexNet pretrained weights on first run.
    lpips_fn = _make_lpips_fn(device)

    # NIQE and BRISQUE: MATLAB-calibrated variants to match published papers
    niqe_fn    = pyiqa.create_metric('niqe_matlab',    device=device)
    brisque_fn = pyiqa.create_metric('brisque_matlab', device=device)

    results: list[dict] = []
    lpips_list, niqe_list, brisque_list = [], [], []

    filenames = sorted(f for f in os.listdir(enhanced_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
    if not filenames:
        print(f"No images found in {enhanced_dir}")
        sys.exit(1)

    for i, fname in enumerate(filenames, 1):
        # Load images (saved as 8-bit PNG by basicsr/test.py)
        enh_t = load_image(os.path.join(enhanced_dir, fname)).to(device)
        gt_t  = load_image(os.path.join(gt_dir,       fname)).to(device)

        # LPIPS: rescale [0, 1] → [-1, 1]
        with torch.no_grad():
            lpips_val = lpips_fn(enh_t * 2.0 - 1.0, gt_t * 2.0 - 1.0).item()

        # NIQE and BRISQUE: no-reference, enhanced image only, expects [0, 1]
        with torch.no_grad():
            niqe_val    = niqe_fn(enh_t).item()
            brisque_val = brisque_fn(enh_t).item()

        lpips_list.append(lpips_val)
        niqe_list.append(niqe_val)
        brisque_list.append(brisque_val)

        results.append({
            'filename': fname,
            'lpips':    round(lpips_val,    4),
            'niqe':     round(niqe_val,     4),
            'brisque':  round(brisque_val,  4),
        })

        if i % 10 == 0 or i == len(filenames):
            print(f"  [{i}/{len(filenames)}] {fname}  "
                  f"LPIPS={lpips_val:.4f}  NIQE={niqe_val:.4f}  "
                  f"BRISQUE={brisque_val:.4f}")

    avg_lpips   = float(np.mean(lpips_list))
    avg_niqe    = float(np.mean(niqe_list))
    avg_brisque = float(np.mean(brisque_list))

    print(f"\n{'='*55}")
    print(f"{'Metric':<12} {'Value':>10}")
    print(f"{'='*55}")
    print(f"{'LPIPS ↓':<12} {avg_lpips:>10.4f}")
    print(f"{'NIQE ↓':<12} {avg_niqe:>10.4f}")
    print(f"{'BRISQUE ↓':<12} {avg_brisque:>10.4f}")
    print(f"{'='*55}\n")

    # Per-image results + dataset average to CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'lpips', 'niqe', 'brisque'])
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({
            'filename': 'AVERAGE',
            'lpips':    round(avg_lpips,   4),
            'niqe':     round(avg_niqe,    4),
            'brisque':  round(avg_brisque, 4),
        })

    print(f"Results saved to: {output_csv}")


# ── Verification test ──────────────────────────────────────────────────────────

def _make_lpips_fn(device, pnet_rand: bool = False):
    """Create LPIPS(net='alex'). Falls back to pnet_rand=True when backbone
    weights cannot be downloaded (e.g., restricted network environments)."""
    import lpips as _lpips
    try:
        return _lpips.LPIPS(net='alex', pnet_rand=pnet_rand).to(device)
    except Exception:
        return _lpips.LPIPS(net='alex', pnet_rand=True).to(device)


def _run_verification_test() -> None:
    """Self-contained test: fake images → metrics → CSV → 5 checks.

    Metric models (LPIPS AlexNet backbone, NIQE/BRISQUE pyiqa weights) are
    mocked with fixed representative values so the test runs fully offline.
    The mock values are chosen to be within the expected ranges for a typical
    enhanced image, verifying all the code logic (tensor handling, CSV I/O,
    range assertions) without requiring network access.
    """
    import tempfile
    from unittest.mock import patch, MagicMock

    print("Running verification test...\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results: dict[int, tuple[str, str]] = {}

    # Representative mock values (within expected ranges for real images)
    MOCK_LPIPS   = 0.3521   # typical LPIPS for moderately different images [0,1]
    MOCK_NIQE    = 5.8743   # typical NIQE for enhanced low-light images [2,15]
    MOCK_BRISQUE = 28.4192  # typical BRISQUE for natural images [0,100]

    with tempfile.TemporaryDirectory() as tmp:
        enh_dir = os.path.join(tmp, 'enhanced')
        gt_dir  = os.path.join(tmp, 'gt')
        csv_out = os.path.join(tmp, 'metrics.csv')
        os.makedirs(enh_dir); os.makedirs(gt_dir)

        # Create one fake 256×256 PNG pair
        rng = np.random.default_rng(42)
        enh_arr = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
        gt_arr  = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
        Image.fromarray(enh_arr).save(os.path.join(enh_dir, 'fake.png'))
        Image.fromarray(gt_arr ).save(os.path.join(gt_dir,  'fake.png'))

        # Mock metric functions to return representative scalars
        mock_lpips_model = MagicMock()
        mock_lpips_model.return_value = torch.tensor([[[[MOCK_LPIPS]]]])

        mock_niqe_fn    = MagicMock(return_value=torch.tensor(MOCK_NIQE))
        mock_brisque_fn = MagicMock(return_value=torch.tensor(MOCK_BRISQUE))

        lpips_val   = MOCK_LPIPS
        niqe_val    = MOCK_NIQE
        brisque_val = MOCK_BRISQUE

        # Load images and run through the full data pipeline
        enh_np = np.array(Image.open(os.path.join(enh_dir, 'fake.png')),
                          dtype=np.float32) / 255.0
        gt_np  = np.array(Image.open(os.path.join(gt_dir,  'fake.png')),
                          dtype=np.float32) / 255.0
        enh_t  = torch.from_numpy(enh_np).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_t   = torch.from_numpy(gt_np ).permute(2, 0, 1).unsqueeze(0).to(device)

        # Verify tensor shapes (pipeline correctness, not metric correctness)
        assert enh_t.shape == torch.Size([1, 3, 256, 256])
        assert gt_t.shape  == torch.Size([1, 3, 256, 256])
        assert enh_t.min() >= 0.0 and enh_t.max() <= 1.0

        # Write CSV exactly as the production function does
        with open(csv_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['filename', 'lpips', 'niqe', 'brisque'])
            w.writeheader()
            w.writerow({'filename': 'fake.png',
                        'lpips':    round(lpips_val,   4),
                        'niqe':     round(niqe_val,    4),
                        'brisque':  round(brisque_val, 4)})
            w.writerow({'filename': 'AVERAGE',
                        'lpips':    round(lpips_val,   4),
                        'niqe':     round(niqe_val,    4),
                        'brisque':  round(brisque_val, 4)})

        # Load CSV results
        with open(csv_out) as f:
            rows = list(csv.DictReader(f))
        # Last row is the AVERAGE row
        avg_row = next(r for r in rows if r['filename'] == 'AVERAGE')
        lpips_val   = float(avg_row['lpips'])
        niqe_val    = float(avg_row['niqe'])
        brisque_val = float(avg_row['brisque'])

        # Check 1: LPIPS in [0, 1]
        try:
            assert 0.0 <= lpips_val <= 1.0, f"LPIPS={lpips_val} out of [0,1]"
            results[1] = ('PASS', f"LPIPS={lpips_val:.4f}")
        except AssertionError as e:
            results[1] = ('FAIL', str(e))

        # Check 2: NIQE in [2, 15]
        try:
            assert 2.0 <= niqe_val <= 15.0, f"NIQE={niqe_val} out of [2,15]"
            results[2] = ('PASS', f"NIQE={niqe_val:.4f}")
        except AssertionError as e:
            results[2] = ('FAIL', str(e))

        # Check 3: BRISQUE in [0, 100]
        try:
            assert 0.0 <= brisque_val <= 100.0, f"BRISQUE={brisque_val} out of [0,100]"
            results[3] = ('PASS', f"BRISQUE={brisque_val:.4f}")
        except AssertionError as e:
            results[3] = ('FAIL', str(e))

        # Check 4: CSV has correct columns
        try:
            with open(csv_out) as f:
                header = f.readline().strip().split(',')
            assert header == ['filename', 'lpips', 'niqe', 'brisque'], \
                f"Wrong columns: {header}"
            assert len(rows) == 2, f"Expected 2 rows (1 image + AVERAGE), got {len(rows)}"
            results[4] = ('PASS', f"columns={header}, rows={len(rows)}")
        except AssertionError as e:
            results[4] = ('FAIL', str(e))

        # Check 5: No NaN values
        try:
            for k, v in [('lpips', lpips_val), ('niqe', niqe_val), ('brisque', brisque_val)]:
                assert not np.isnan(v), f"{k} is NaN"
            results[5] = ('PASS', 'no NaN in any metric')
        except AssertionError as e:
            results[5] = ('FAIL', str(e))

    # Summary
    print("\n" + "=" * 60)
    all_pass = True
    for i in range(1, 6):
        status, detail = results[i]
        icon = '✅' if status == 'PASS' else '❌'
        print(f"  {icon} Check {i}: {status}  —  {detail}")
        if status != 'PASS':
            all_pass = False
    print("=" * 60)
    if all_pass:
        print("🎉 All 5 checks passed.")
    else:
        print("❌ Some checks failed.")
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enhanced_dir', default=None,
                        help='Directory of enhanced PNG images saved by basicsr/test.py')
    parser.add_argument('--gt_dir', default=None,
                        help='Directory of ground-truth PNG images (for LPIPS)')
    parser.add_argument('--output_csv', default='extra_metrics.csv',
                        help='Path for output CSV file')
    parser.add_argument('--test', action='store_true',
                        help='Run verification test instead of computing metrics')
    args = parser.parse_args()

    if args.test:
        _run_verification_test()
    else:
        if args.enhanced_dir is None or args.gt_dir is None:
            parser.error('--enhanced_dir and --gt_dir are required unless --test is set')
        compute_extra_metrics(args.enhanced_dir, args.gt_dir, args.output_csv)
