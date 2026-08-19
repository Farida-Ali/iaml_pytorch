"""
Illumination-Conditioned Noise Floor (ICNF).

THE IDEA
────────
In a Retinex model the illumination estimate is not merely a brightness prior —
it is a *noise model*. Photon arrival is Poisson and read-out is Gaussian, so
the noise variance at a pixel is an affine function of the underlying signal:

    var_noise(x) = a · I(x) + b

with `a` set by sensor gain and `b` by read noise. The moment the network
estimates illumination, the noise floor at every pixel becomes predictable.

WHY THIS NEEDS A WAVELET BASIS
──────────────────────────────
The Haar transform used here is ORTHONORMAL (verified: Parseval holds to
5.9e-8 in scripts/feasibility_icnf.py). For an orthonormal transform, white
noise of variance s² lands with variance s² in *every* subband. So the
predicted pixel-domain noise variance transfers directly to the high-frequency
subbands with no rescaling — the noise floor is analytically known
per-coefficient. Haar is also spatially localised, so that floor is known
per-location.

Fourier is orthonormal but not localised (no spatially varying floor). A plain
convolution is localised but not orthonormal (variance is not preserved, so the
floor is not analytic). Wavelets are the only basis giving both — which is what
makes the transform *necessary* here rather than decorative.

WHAT IT BUYS
────────────
Observed high-frequency energy is texture + noise, entangled. Subtracting the
predicted floor isolates texture:

    evidence(x) = max(0, E_hf(x) − var_noise(x)) / (var_noise(x) + eps)

Measured separability (AUC, textured vs flat regions), from
scripts/feasibility_icnf2.py:

    illum range   raw HF energy   SNR-Net-style   ICNF
      1:1            0.9025          0.4591      0.9165
      3:1            0.6607          0.4621      0.8651
     19:1            0.6035          0.4633      0.7795

Raw HF energy collapses as illumination range grows — it confuses "brightly-lit
texture" with "texture" and goes blind in shadows. ICNF holds, because it
normalises by the illumination-predicted floor. Wide within-image illumination
range is the defining property of low-light imagery, so the regime where this
wins is exactly the target domain.

LIMITATION (state it in the paper)
──────────────────────────────────
Below a texture-to-noise ratio of ~0.02 no statistic separates texture from
noise — the information is gone. ICNF returns ~0 evidence there, which is the
correct behaviour: it tells the model to suppress rather than hallucinate
detail.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavelet_utils import HaarDWT2D   # repo-root module, as in fd2rt_a4_arch.py


class ICNF(nn.Module):
    """Illumination-Conditioned Noise Floor / texture-evidence map.

    Args:
        a_init (float): initial sensor-gain coefficient (var = a*I + b).
        b_init (float): initial read-noise floor.
        learn_params (bool): learn (a, b) jointly with the network. They are
            stored as log-parameters so they stay strictly positive.
        window (int): box window (in LL-resolution pixels) over which
            high-frequency energy is pooled. Odd.
        eps (float): numerical floor.
        mode (str): 'subtract' -> max(0, E - v) / (v + eps)   [default]
                    'ratio'    -> E / (v + eps)
            Both were measured; 'subtract' scored slightly higher at high
            texture-to-noise ratio (0.9237 vs 0.8697 at TNR 0.708).
        detach_illum (bool): treat the illumination input as a constant when
            computing the floor. Keeps ICNF a *conditioning* signal and stops
            the network from trivially lowering the predicted floor to inflate
            evidence. The evidence map itself still carries gradient through
            the observed energy term.
    """

    def __init__(self, a_init=0.02, b_init=1e-5, learn_params=True,
                 window=9, eps=1e-8, mode='subtract', detach_illum=True):
        super().__init__()
        if window % 2 == 0:
            raise ValueError(f'window must be odd, got {window}')
        if mode not in ('subtract', 'ratio'):
            raise ValueError(f"mode must be 'subtract' or 'ratio', got {mode}")

        self.dwt = HaarDWT2D()
        self.window = window
        self.eps = eps
        self.mode = mode
        self.detach_illum = detach_illum

        log_a = torch.log(torch.tensor(float(a_init)))
        log_b = torch.log(torch.tensor(float(b_init)))
        if learn_params:
            self.log_a = nn.Parameter(log_a)
            self.log_b = nn.Parameter(log_b)
        else:
            self.register_buffer('log_a', log_a)
            self.register_buffer('log_b', log_b)

        # Box-pooling kernel for local energy, registered so it moves with .to()
        k = torch.ones(1, 1, window, window) / float(window * window)
        self.register_buffer('box', k)

    # ------------------------------------------------------------------ #
    @property
    def a(self):
        return self.log_a.exp()

    @property
    def b(self):
        return self.log_b.exp()

    def _local_mean(self, x):
        """Box-filter each channel independently."""
        C = x.shape[1]
        return F.conv2d(x, self.box.expand(C, 1, -1, -1),
                        padding=self.window // 2, groups=C)

    # ------------------------------------------------------------------ #
    def hf_energy(self, img):
        """Mean squared high-frequency wavelet coefficient, pooled locally.

        Returns [B, C, H/2, W/2] at LL resolution.
        """
        _, lh, hl, hh = self.dwt(img)
        return self._local_mean((lh ** 2 + hl ** 2 + hh ** 2) / 3.0)

    def noise_floor(self, illum):
        """Predicted noise variance from the Poisson-Gaussian model.

        Args:
            illum: [B, C, H, W] or [B, 1, H, W] illumination / intensity
                estimate in [0, 1].
        Returns:
            [B, C, H/2, W/2] predicted variance at LL resolution.
        """
        if self.detach_illum:
            illum = illum.detach()
        # Average the illumination over each 2x2 block so the floor is stated at
        # the same resolution as the subbands it will be compared against.
        illum_ll = F.avg_pool2d(illum, 2)
        return self.a * illum_ll + self.b

    def estimate_sigma_mad(self, img):
        """Classical MAD noise-level estimate from the HH subband.

        sigma_hat = median(|HH|) / 0.6745, exact for an orthonormal transform.
        Verified to <1% error for sigma in [0.005, 0.10]. Provided as a
        calibration/diagnostic path: it estimates a GLOBAL sigma with no
        illumination model, so it is what ICNF improves upon, and it is useful
        for initialising (a, b) on a new sensor or dataset.
        """
        _, _, _, hh = self.dwt(img)
        return hh.abs().flatten(1).median(dim=1).values / 0.6745

    # ------------------------------------------------------------------ #
    def forward(self, img, illum=None, out_size=None):
        """Compute the texture-evidence map.

        Args:
            img: [B, C, H, W] observed (low-light) image, values in [0, 1].
            illum: [B, C|1, H, W] illumination estimate. If None, a local mean
                of `img` is used — the inference-time proxy, which is what the
                feasibility study measured, so results are not oracle-inflated.
            out_size: (H, W) to upsample the evidence map to. Defaults to the
                input resolution.

        Returns:
            evidence: [B, 1, H, W] non-negative texture evidence. ~0 where the
                observed high-frequency energy is fully explained by the
                predicted noise floor.
        """
        if img.dim() != 4:
            raise ValueError(f'expected [B,C,H,W], got {tuple(img.shape)}')
        B, C, H, W = img.shape

        if illum is None:
            illum = self._local_mean(img)
        elif illum.shape[1] == 1 and C > 1:
            illum = illum.expand(-1, C, -1, -1)

        E = self.hf_energy(img)                 # [B, C, H/2, W/2]
        v = self.noise_floor(illum)             # [B, C, H/2, W/2]

        if self.mode == 'subtract':
            evidence = (E - v).clamp_min(0.0) / (v + self.eps)
        else:
            evidence = E / (v + self.eps)

        # Collapse colour channels — noise/texture evidence is a scene property.
        evidence = evidence.mean(dim=1, keepdim=True)     # [B, 1, H/2, W/2]

        size = out_size if out_size is not None else (H, W)
        return F.interpolate(evidence, size=size, mode='bilinear',
                             align_corners=False)


class ICNFGate(nn.Module):
    """Turns the ICNF evidence map into a bounded per-pixel gate in (0, 1).

    Replaces the scalar `gates[i]` parameter in DDA_Block_Dual with a
    spatially-varying gate: the frequency branch is opened where there is
    genuine texture evidence and closed where high-frequency content is fully
    explained by noise.

        gate = sigmoid(slope * (log1p(evidence) - bias))

    log1p compresses the heavy tail of the evidence ratio; `slope` and `bias`
    are learnable so the network can choose its own operating point.

    On `bias_init`: the evidence ratio is an absolute quantity ("how far above
    the predicted noise floor is this?"), so the gate's operating point has to
    be set on that absolute scale rather than normalised per image — normalising
    would discard exactly the information the floor provides. Typical evidence
    on textured content puts log1p(evidence) around 2, so bias_init must sit
    comfortably above that for the gate to start closed. The default of 6.0
    yields gate ~= 0.02 at initialisation, so the block reduces to its
    spatial-only behaviour at step 0 — matching how A4's zero-init frequency
    branch preserves A1 behaviour, and keeping A4 checkpoints loadable. The gate
    opens as training pushes `bias` down.
    """

    def __init__(self, slope_init=1.0, bias_init=6.0, learnable=True):
        super().__init__()
        slope = torch.tensor(float(slope_init))
        bias = torch.tensor(float(bias_init))
        if learnable:
            self.slope = nn.Parameter(slope)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer('slope', slope)
            self.register_buffer('bias', bias)

    def forward(self, evidence):
        return torch.sigmoid(self.slope * (torch.log1p(evidence.clamp_min(0)) - self.bias))
