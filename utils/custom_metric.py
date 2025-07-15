from tsl.metrics.torch.functional import rmse, mse, mae
from tsl.metrics.torch import MaskedMetric
from typing import Any
import torch

class MaskedRMSE(MaskedMetric):
    """Root Mean Squared Error Metric."""

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, mask_nans=False, mask_inf=False, at=None, **kwargs: Any):
        # Use MSE as the base metric, not RMSE
        super(MaskedRMSE, self).__init__(
            metric_fn=mse,  # ← Use MSE instead of RMSE
            mask_nans=mask_nans,
            mask_inf=mask_inf,
            metric_fn_kwargs={'reduction': 'none'},
            at=at,
            **kwargs,
        )

    def compute(self):
        """Override to take square root of the mean squared error."""
        if self.numel > 0:
            mse_value = self.value / self.numel  # This gives MSE
            return torch.sqrt(mse_value)         # Take sqrt to get RMSE
        return torch.sqrt(self.value)