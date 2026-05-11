import torch
import torch.nn as nn

# Import from our arch module (in basicsr/archs/, not basicsr/models/archs/)
from basicsr.archs.iaml_arch import IAMLNet as _IAMLNet


class IAMLNet(_IAMLNet):
    """BasicSR-compatible inference wrapper around IAMLNet.

    BasicSR's test pipeline calls net_g(lq_image) with a single tensor argument.
    This wrapper routes that single-argument call to the student-only inference
    path, which is all that is needed at test time.

    The full two-argument forward(x_low, x_clean) and ema_update() from the
    parent class remain available for use with train_iaml.py.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Called by BasicSR's nonpad_test() with the padded low-light input.
        # Uses student encoder + student decoder only (teacher is discarded at inference).
        return self.inference(x)
