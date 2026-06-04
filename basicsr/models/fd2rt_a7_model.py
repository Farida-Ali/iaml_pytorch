"""
basicsr/models/fd2rt_a7_model.py
────────────────────────────────
FD²RT A7 training model — adds two loss terms on top of the A4 architecture:

    L_total = L1(I_en, I_gt)                [existing pixel loss, weight 1.0]
            + 0.1  * FrequencyAwareLoss     [NEW]
            + 0.01 * IlluminationTVLoss      [NEW]

The A4 architecture (fd2rt_a4_arch.py) is NOT modified. This wrapper:

  • Subclasses ImageCleanModel (the existing training model). Everything about
    the optimizer, scheduler, grad-clip, AMP, EMA and validation is inherited
    unchanged — only `optimize_parameters` is overridden to add the two terms.

  • Captures the W-IE LL subband via a FORWARD HOOK on the estimator's HaarDWT2D
    submodule (`...estimator.dwt`). HaarDWT2D.forward returns (LL, LH, HL, HH);
    the hook stores LL (shape [B, 3, H/2, W/2]). No architecture edit is needed —
    the hook is attached from here at training-setup time.

Config (options/train/LOLv1/train_fd2rt_a7_LOL_v1.yml):
    model_type: FD2RT_A7_Model
    train:
      pixel_opt: { type: L1Loss, loss_weight: 1 }
      freq_opt:  { type: FrequencyAwareLoss, loss_weight: 0.1, w_low: 1.0, w_high: 2.0 }
      tv_opt:    { type: IlluminationTVLoss,  loss_weight: 0.01 }
"""

import importlib
from collections import OrderedDict

import torch

from basicsr.models.image_restoration_model import ImageCleanModel
from basicsr.utils import get_root_logger

try:
    from torch.cuda.amp import autocast
    _HAS_AMP = True
except Exception:                       # pragma: no cover
    _HAS_AMP = False

loss_module = importlib.import_module('basicsr.models.losses')


class FD2RT_A7_Model(ImageCleanModel):
    """A4 architecture + frequency-aware loss + illumination-TV prior."""

    # ── Training setup: instantiate new losses + attach LL-capture hook ──── #
    def init_training_settings(self):
        # Sets up cri_pix, optimizer, scheduler, EMA exactly as A4.
        super().init_training_settings()

        train_opt = self.opt['train']
        logger = get_root_logger()

        # Frequency-aware loss (optional but expected for A7)
        self.cri_freq = None
        if train_opt.get('freq_opt'):
            freq_cfg = dict(train_opt['freq_opt'])
            freq_type = freq_cfg.pop('type')
            self.cri_freq = getattr(loss_module, freq_type)(**freq_cfg).to(self.device)
            logger.info(f'A7: frequency loss enabled — {freq_type} {freq_cfg}')

        # Illumination total-variation prior (optional but expected for A7)
        self.cri_tv = None
        if train_opt.get('tv_opt'):
            tv_cfg = dict(train_opt['tv_opt'])
            tv_type = tv_cfg.pop('type')
            self.cri_tv = getattr(loss_module, tv_type)(**tv_cfg).to(self.device)
            logger.info(f'A7: illumination-TV loss enabled — {tv_type} {tv_cfg}')

        # Forward hook on the W-IE DWT to capture the LL subband (Option A).
        self._captured_LL = None
        if self.cri_tv is not None:
            self._register_ll_hook()

    def _register_ll_hook(self):
        """Attach a forward hook on the W-IE's HaarDWT2D to capture LL.

        The W-IE DWT lives at module path '...estimator.dwt'. The Freq_MSA
        blocks also contain HaarDWT2D modules, but their paths end in
        'freq_blocks.<i>.dwt', so matching on 'estimator.dwt' uniquely selects
        the illumination-estimator DWT. With stage=1 there is exactly one.
        """
        net = self.get_bare_model(self.net_g)
        target_module = None
        target_name = None
        for name, module in net.named_modules():
            if name.endswith('estimator.dwt'):
                target_module = module
                target_name = name
                break
        if target_module is None:
            raise RuntimeError(
                'FD2RT_A7_Model: could not locate W-IE "estimator.dwt" module '
                'to attach the LL-capture hook.')

        def _hook(_module, _inp, out):
            # HaarDWT2D.forward returns (LL, LH, HL, HH); keep LL only.
            self._captured_LL = out[0]

        target_module.register_forward_hook(_hook)
        get_root_logger().info(
            f'A7: LL-capture forward hook attached to "{target_name}".')

    # ── Optimization step: pixel + frequency + TV ────────────────────────── #
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self._captured_LL = None

        with autocast(enabled=self.use_amp):
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]
            self.output = preds[-1]

            loss_dict = OrderedDict()

            # Existing pixel loss (L1 on the enhanced output)
            l_pix = 0.
            for pred in preds:
                l_pix = l_pix + self.cri_pix(pred, self.gt)
            loss_dict['l_pix'] = l_pix
            l_total = l_pix

            # NEW: frequency-aware restoration loss
            if self.cri_freq is not None:
                l_freq = self.cri_freq(self.output, self.gt)
                loss_dict['l_freq'] = l_freq
                l_total = l_total + l_freq

            # NEW: illumination smoothness (TV) prior on the W-IE LL subband
            if self.cri_tv is not None:
                if self._captured_LL is None:
                    raise RuntimeError(
                        'FD2RT_A7_Model: LL subband was not captured during '
                        'forward — the hook did not fire.')
                l_tv = self.cri_tv(self._captured_LL)
                loss_dict['l_tv'] = l_tv
                l_total = l_total + l_tv

        # Backward + grad-clip + step — identical to A4 except l_total.
        self.amp_scaler.scale(l_total).backward()
        self.amp_scaler.unscale_(self.optimizer_g)

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(
                self.net_g.parameters(),
                self.opt['train'].get('clip_grad_norm', 1.0))

        self.amp_scaler.step(self.optimizer_g)
        self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
