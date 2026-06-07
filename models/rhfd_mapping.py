"""
RHFD Gaze Probability Mapping Module.

Implements the key RHFD insight: first predict a probability distribution
over spherical anchor points, then convert the distribution to a 3D gaze
direction vector. This "probability-first" approach provides:
  - Better uncertainty quantification via distribution entropy
  - Multi-modal capability (distribution can be multi-peaked)
  - Natural kappa estimation from distribution concentration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeProbabilityHead(nn.Module):
    """
    Predicts a probability distribution over K spherical anchor points
    from LSTM hidden states, then converts to a 3D gaze direction.

    Architecture:
        hidden [B, T, hidden_dim] → Linear → ReLU → Dropout → Linear → Softmax
        → gaze_probs [B, T, K]
        → weighted sum of anchors → gaze_direction [B, T, 3]
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 128,
        n_anchors: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_anchors = n_anchors
        self.in_dim = in_dim

        self.prob_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_anchors),
        )

        # Kappa estimator from distribution entropy
        self.kappa_from_entropy = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        hard: bool = False,
    ) -> dict:
        """
        Args:
            hidden:  [B, T, in_dim] LSTM hidden states
            anchors: [K, 3] spherical anchor points (pre-computed, non-trainable)
            hard:    if True, use argmax mapping (inference only)

        Returns:
            dict with:
                'probs':      [B, T, K] probability distribution over anchors
                'direction':  [B, T, 3] predicted 3D gaze direction (normalized)
                'kappa':      [B, T, 1] concentration parameter from entropy
                'entropy':    [B, T, 1] distribution entropy (for diagnostics)
        """
        B, T, _ = hidden.shape
        K = anchors.shape[0]

        # Predict logits and apply softmax
        logits = self.prob_net(hidden)  # [B, T, K]
        probs = F.softmax(logits, dim=-1)  # [B, T, K]

        # Convert probability to direction
        if hard:
            # Argmax mapping (non-differentiable, for inference)
            indices = torch.argmax(probs, dim=-1)  # [B, T]
            direction = anchors[indices]  # [B, T, 3]
        else:
            # Soft mapping: weighted sum of anchors (differentiable)
            direction = torch.einsum('btk,kd->btd', probs, anchors)  # [B, T, 3]
            direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)

        # Estimate kappa from distribution entropy
        # High concentration (low entropy) → high kappa (confident prediction)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)  # [B, T, 1]
        # Max entropy for K anchors = log(K)
        max_entropy = torch.log(torch.tensor(K, dtype=entropy.dtype, device=entropy.device))
        normalized_entropy = entropy / max_entropy  # [0, 1]
        # Concentration = 1 - normalized_entropy (0 = uniform, 1 = delta)
        concentration = 1.0 - normalized_entropy
        kappa = self.kappa_from_entropy(concentration) + 0.5  # minimum kappa = 0.5

        return {
            'probs': probs,
            'direction': direction,
            'kappa': kappa,
            'entropy': entropy,
        }


class GazeDirectRegressionHead(nn.Module):
    """
    Fallback: direct regression head for gaze direction (original GAFA style).
    Used when use_probability_head=False.

    Architecture:
        hidden [B,T,in_dim] → Linear → ReLU → Linear → 3 → normalize
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.direction_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.kappa_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, hidden: torch.Tensor) -> dict:
        """
        Args:
            hidden: [B, T, in_dim] LSTM hidden states

        Returns:
            dict with 'direction' [B,T,3] and 'kappa' [B,T,1]
        """
        direction = self.direction_layer(hidden)
        direction = direction / (
            torch.norm(direction, dim=-1, keepdim=True) + 1e-8
        )
        kappa = self.kappa_layer(hidden)

        return {
            'probs': None,
            'direction': direction,
            'kappa': kappa,
            'entropy': None,
        }
