"""
Exponential Smoothing Module for Gaze Direction Sequences.

Applies exponential moving average (EMA) to smooth gaze direction predictions
across time, reducing frame-to-frame jitter while preserving real gaze shifts.

Two modes:
  1. Learnable alpha (default): alpha = sigmoid(alpha_raw), trained end-to-end
  2. Fixed alpha: traditional EMA, alpha set at initialization

Applied at two positions in the pipeline:
  - Intermediate: after LSTM, before TWIESN (optional)
  - Final: after TWIESN, producing the final prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExponentialSmoothing(nn.Module):
    """
    Exponential Moving Average smoothing for gaze direction sequences.

    smoothed[t] = alpha * gaze[t] + (1 - alpha) * smoothed[t-1]
    smoothed[0] = gaze[0]

    The smoothing is applied per-sequence, maintaining the temporal order.
    """

    def __init__(self, alpha: float = 0.3, learnable: bool = True):
        """
        Args:
            alpha:    initial smoothing factor in (0, 1)
                       - Higher alpha → more weight to current frame (less smoothing)
                       - Lower alpha → more weight to history (more smoothing)
            learnable: if True, alpha is trained via sigmoid(raw_alpha)
        """
        super().__init__()
        self.learnable = learnable

        if learnable:
            # Initialize raw_alpha so sigmoid(raw_alpha) ≈ alpha
            import numpy as np
            raw_alpha = np.log(alpha / (1.0 - alpha + 1e-8))
            self.alpha_raw = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

    def get_alpha(self) -> torch.Tensor:
        """Get the current smoothing factor."""
        if self.learnable:
            return torch.sigmoid(self.alpha_raw)
        return self.alpha

    def forward(self, gaze_sequence: torch.Tensor) -> dict:
        """
        Apply exponential smoothing to a gaze direction sequence.

        Args:
            gaze_sequence: [B, T, D] sequence of gaze direction vectors

        Returns:
            dict with:
                'smoothed': [B, T, D] smoothed sequence
                'alpha':    scalar, current smoothing factor
                'raw':      [B, T, D] original input (for residual connections)
        """
        B, T, D = gaze_sequence.shape
        alpha = self.get_alpha()

        smoothed = []
        s_prev = gaze_sequence[:, 0]  # Initial state = first frame

        for t in range(T):
            x_t = gaze_sequence[:, t]
            s_t = alpha * x_t + (1.0 - alpha) * s_prev
            # Re-normalize to maintain unit vector property
            s_t = s_t / (torch.norm(s_t, dim=-1, keepdim=True) + 1e-8)
            smoothed.append(s_t)
            s_prev = s_t

        smoothed = torch.stack(smoothed, dim=1)  # [B, T, D]

        return {
            'smoothed': smoothed,
            'alpha': alpha,
            'raw': gaze_sequence,
        }


class AdaptiveExponentialSmoothing(nn.Module):
    """
    Adaptive EMA: alpha varies per frame based on prediction confidence (kappa).

    When confidence is high → higher alpha (less smoothing, trust prediction).
    When confidence is low  → lower alpha (more smoothing, rely on history).

    alpha[t] = sigmoid(beta * kappa[t] + bias)
    smoothed[t] = alpha[t] * gaze[t] + (1 - alpha[t]) * smoothed[t-1]
    """

    def __init__(self, beta_init: float = 1.0, bias_init: float = 0.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))

    def forward(
        self, gaze_sequence: torch.Tensor, kappa: torch.Tensor
    ) -> dict:
        """
        Args:
            gaze_sequence: [B, T, D] gaze direction vectors
            kappa:         [B, T, 1] confidence (concentration) per frame

        Returns:
            dict with 'smoothed' [B,T,D], 'alphas' [B,T,1], 'raw' [B,T,D]
        """
        B, T, D = gaze_sequence.shape

        # Frame-adaptive alpha from kappa
        logits = self.beta * kappa.squeeze(-1) + self.bias  # [B, T]
        alphas = torch.sigmoid(logits)  # [B, T]

        smoothed = []
        s_prev = gaze_sequence[:, 0]

        for t in range(T):
            x_t = gaze_sequence[:, t]
            a_t = alphas[:, t:t + 1]  # [B, 1]
            s_t = a_t * x_t + (1.0 - a_t) * s_prev
            s_t = s_t / (torch.norm(s_t, dim=-1, keepdim=True) + 1e-8)
            smoothed.append(s_t)
            s_prev = s_t

        smoothed = torch.stack(smoothed, dim=1)

        return {
            'smoothed': smoothed,
            'alphas': alphas.unsqueeze(-1),
            'raw': gaze_sequence,
        }
