"""
TWIESN: Tchebichef Weighted Infinite Echo State Network.

A reservoir computing module that smooths gaze direction sequences using
a fixed random reservoir with Tchebichef polynomial expansion.

Key properties:
  - Reservoir weights (W_in, W_res) are randomly initialized and FIXED
  - Only the readout layer is trained (cheap to optimize)
  - Tchebichef polynomials provide orthogonal multi-scale temporal features
  - Echo State Property (spectral radius < 1) ensures stable dynamics
  - Well-suited for short sequences (7 frames) where learned models overfit

Tchebichef polynomials (first 4 orders):
  T_0(x) = 1
  T_1(x) = x
  T_2(x) = 2x^2 - 1
  T_3(x) = 4x^3 - 3x
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TchebichefExpansion(nn.Module):
    """
    Tchebichef polynomial expansion of reservoir states.
    Applies T_0 through T_{order-1} to each reservoir dimension.
    """

    def __init__(self, order: int = 4):
        super().__init__()
        self.order = order

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, D] reservoir state

        Returns:
            list of [B, D] tensors, one per Tchebichef order
        """
        # Clamp to [-1, 1] for numerical stability of polynomials
        x = torch.clamp(x, -0.999, 0.999)

        expansions = []
        if self.order >= 0:
            T0 = torch.ones_like(x)       # T_0(x) = 1
            expansions.append(T0)
        if self.order >= 1:
            T1 = x                          # T_1(x) = x
            expansions.append(T1)
        if self.order >= 2:
            T2 = 2.0 * x ** 2 - 1.0        # T_2(x) = 2x^2 - 1
            expansions.append(T2)
        if self.order >= 3:
            T3 = 4.0 * x ** 3 - 3.0 * x    # T_3(x) = 4x^3 - 3x
            expansions.append(T3)
        # Higher orders use recurrence: T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x)
        for n in range(4, self.order):
            Tn = 2.0 * x * expansions[-1] - expansions[-2]
            expansions.append(Tn)

        return expansions


class TWIESN(nn.Module):
    """
    Tchebichef Weighted Infinite Echo State Network.

    Processes a sequence of gaze direction vectors through a fixed reservoir
    with Tchebichef polynomial expansion, producing smoothed outputs.

    Architecture:
        1. Input projection:  x_t → W_in @ x_t
        2. Reservoir update:  r_t = tanh(W_in @ x_t + W_res @ r_{t-1})
        3. Tchebichef expansion: T_0(r_t), T_1(r_t), ..., T_{order-1}(r_t)
        4. Readout: y_t = W_out @ [r_t; T_1(r_t); ...; T_{order-1}(r_t)]
        5. Normalize: y_t = y_t / ||y_t||
    """

    def __init__(
        self,
        input_dim: int = 3,
        reservoir_dim: int = 128,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        sparsity: float = 0.9,
        tchebichef_order: int = 4,
        leaking_rate: float = 0.3,
    ):
        """
        Args:
            input_dim:        dimension of input (3 for 3D gaze direction)
            reservoir_dim:    number of reservoir neurons
            spectral_radius:  target spectral radius of reservoir (< 1 for ESP)
            input_scaling:    scaling factor for input weights
            sparsity:         fraction of zero weights in reservoir
            tchebichef_order: number of Tchebichef polynomial orders (incl. T_0)
            leaking_rate:     leaky integration rate (0→static, 1→no leakage)
        """
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.tchebichef_order = tchebichef_order
        self.leaking_rate = leaking_rate

        # --- Fixed (non-trainable) reservoir weights ---
        # Input weights: sparse random
        W_in = torch.randn(reservoir_dim, input_dim) * input_scaling
        mask_in = (torch.rand(reservoir_dim, input_dim) > sparsity * 0.5).float()
        self.register_buffer('W_in', W_in * mask_in)

        # Reservoir weights: sparse random with controlled spectral radius
        W_res = torch.randn(reservoir_dim, reservoir_dim)
        mask_res = (torch.rand(reservoir_dim, reservoir_dim) > sparsity).float()
        W_res = W_res * mask_res
        # Scale to target spectral radius
        if spectral_radius > 0:
            with torch.no_grad():
                eigenvalues = torch.linalg.eigvals(W_res)
                # torch.abs handles complex dtypes automatically
                # Use .float() to convert complex abs result to real tensor
                abs_vals = torch.abs(eigenvalues)
                if abs_vals.is_complex():
                    abs_vals = abs_vals.real
                max_eigenvalue = torch.max(abs_vals)
                if max_eigenvalue > 0:
                    W_res = W_res * (spectral_radius / max_eigenvalue)
        self.register_buffer('W_res', W_res)

        # --- Tchebichef expansion ---
        self.tchebichef = TchebichefExpansion(order=tchebichef_order)

        # --- Trainable readout layer ---
        # Expanded dimension = reservoir_dim * tchebichef_order
        expanded_dim = reservoir_dim * tchebichef_order
        self.readout = nn.Sequential(
            nn.Linear(expanded_dim, reservoir_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reservoir_dim, input_dim),
        )

        # Initial reservoir state (learnable)
        self.r0 = nn.Parameter(torch.zeros(1, reservoir_dim))

    def _update_reservoir(
        self, x_t: torch.Tensor, r_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Single reservoir update step with leaky integration.

        Args:
            x_t:    [B, input_dim] current input
            r_prev: [B, reservoir_dim] previous reservoir state

        Returns:
            r_t: [B, reservoir_dim] updated reservoir state
        """
        # Reservoir update
        pre_activation = (
            F.linear(x_t, self.W_in) +
            F.linear(r_prev, self.W_res)
        )
        r_candidate = torch.tanh(pre_activation)

        # Leaky integration
        r_t = (1.0 - self.leaking_rate) * r_prev + self.leaking_rate * r_candidate
        return r_t

    def _readout(self, r_t: torch.Tensor) -> torch.Tensor:
        """
        Compute output from reservoir state with Tchebichef expansion.

        Args:
            r_t: [B, reservoir_dim] reservoir state

        Returns:
            y_t: [B, input_dim] output (gaze direction, unnormalized)
        """
        # Tchebichef polynomial expansion
        expansions = self.tchebichef(r_t)  # list of [B, D], length = order
        expanded = torch.cat(expansions, dim=-1)  # [B, D * order]

        # Readout
        y_t = self.readout(expanded)  # [B, input_dim]
        return y_t

    def forward(self, gaze_init: torch.Tensor) -> dict:
        """
        Process gaze sequence through TWIESN.

        Args:
            gaze_init: [B, T, input_dim] initial gaze direction estimates (from LSTM)

        Returns:
            dict with:
                'direction':   [B, T, input_dim] smoothed gaze directions (normalized)
                'raw':         [B, T, input_dim] pre-normalization readout
                'reservoir':   [B, T, reservoir_dim] reservoir states
        """
        B, T, _ = gaze_init.shape

        # Initialize reservoir state
        r_t = self.r0.expand(B, -1)  # [B, reservoir_dim]

        outputs = []
        reservoirs = []

        for t in range(T):
            x_t = gaze_init[:, t]  # [B, 3]
            r_t = self._update_reservoir(x_t, r_t)
            y_t = self._readout(r_t)

            # Normalize to unit vector
            y_t_normalized = y_t / (torch.norm(y_t, dim=-1, keepdim=True) + 1e-8)

            outputs.append(y_t_normalized)
            reservoirs.append(r_t)

        direction = torch.stack(outputs, dim=1)  # [B, T, 3]
        raw = torch.stack(
            [self._readout(r) for r in reservoirs], dim=1
        )
        reservoir_states = torch.stack(reservoirs, dim=1)  # [B, T, D]

        return {
            'direction': direction,
            'raw': raw,
            'reservoir': reservoir_states,
        }

    def reset_state(self, batch_size: int = 1):
        """Reset reservoir state (useful between sequences)."""
        self._r_current = self.r0.expand(batch_size, -1)
