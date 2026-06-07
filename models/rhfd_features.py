"""
RHFD (Refined Hidden Follower Detection) Gaze Feature Extraction Module.

Extracts two key gaze-related temporal features from head direction sequences:
  - Gf (Gaze Fixation Frequency): frame-to-frame angular velocity of gaze
  - Gd (Gaze Density): spatial concentration of gaze points within a sliding window

These features are fused and fed as additional channels to the GazeModule LSTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeFixationFrequency(nn.Module):
    """
    Compute gaze fixation frequency (Gf) from head direction sequences.

    Gf measures the angular velocity of gaze direction changes.
    Higher Gf → more frequent gaze shifts (scanning behavior).
    Lower Gf → more stable fixation.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, head_dir: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head_dir: [B, T, 3] normalized head direction vectors

        Returns:
            gf_features: [B, T, 1] fixation frequency per frame
        """
        B, T, D = head_dir.shape

        # Compute cosine similarity between adjacent frames
        # head_dir[:, :-1] * head_dir[:, 1:] → [B, T-1]
        cos_angle = torch.sum(head_dir[:, :-1] * head_dir[:, 1:], dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0 + self.eps, 1.0 - self.eps)

        # Angular difference in radians
        angular_diff = torch.acos(cos_angle)  # [B, T-1]

        # Pad to T frames: first frame has 0 velocity
        padding = torch.zeros(B, 1, device=head_dir.device, dtype=angular_diff.dtype)
        angular_diff_padded = torch.cat([padding, angular_diff], dim=1)  # [B, T]

        # Normalize by max expected angular velocity (π radians)
        gf = angular_diff_padded / torch.pi  # [B, T]

        return gf.unsqueeze(-1)  # [B, T, 1]


class GazeDensity(nn.Module):
    """
    Compute gaze density (Gd) from head direction sequences.

    Gd measures spatial concentration of gaze within a temporal window.
    High density → person is focusing on a narrow region.
    """

    def __init__(self, window_size: int = 3):
        """
        Args:
            window_size: sliding window size for density estimation (should be odd)
        """
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = window_size
        self.half_window = window_size // 2

    def forward(self, head_dir: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head_dir: [B, T, 3] normalized head direction vectors

        Returns:
            gd_features: [B, T, 1] gaze density per frame
        """
        B, T, D = head_dir.shape

        # Pad the sequence for sliding window
        head_padded = F.pad(
            head_dir.transpose(1, 2),  # [B, 3, T]
            (self.half_window, self.half_window),
            mode='replicate'
        ).transpose(1, 2)  # [B, T+2*hw, 3]

        # For each frame, compute mean cosine similarity within window
        density_list = []
        for t in range(T):
            window = head_padded[:, t:t + self.window_size]  # [B, W, 3]
            # Pairwise cosine similarity within window
            center = head_dir[:, t:t + 1]  # [B, 1, 3]
            similarities = torch.sum(center * window, dim=-1)  # [B, W]
            # Mean similarity as density measure (excluding self which is 1.0)
            density = similarities.mean(dim=-1, keepdim=True)  # [B, 1]
            density_list.append(density)

        gd = torch.stack(density_list, dim=1)  # [B, T, 1]

        # Normalize: map from [-1, 1] similarity to [0, 1] density
        gd = (gd + 1.0) / 2.0

        return gd


class RHFDFeatureFusion(nn.Module):
    """
    Learnable fusion of Gf and Gd features into a compact representation
    for injection into the GazeModule LSTM.

    Architecture:
        [Gf, Gd] → Linear(2, 16) → ReLU → Linear(16, 16) → output [B, T, 16]
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 16):
        super().__init__()
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, gf: torch.Tensor, gd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gf: [B, T, 1] gaze fixation frequency
            gd: [B, T, 1] gaze density

        Returns:
            fusion: [B, T, output_dim] fused RHFD feature
        """
        combined = torch.cat([gf, gd], dim=-1)  # [B, T, 2]
        fusion = self.fusion_net(combined)       # [B, T, output_dim]
        return fusion


class RHFDFeatureExtractor(nn.Module):
    """
    Complete RHFD feature extraction pipeline.

    Extracts Gf (fixation frequency), Gd (gaze density) from head direction
    sequences, and fuses them into a compact feature representation for
    downstream gaze estimation.
    """

    def __init__(self, window_size: int = 3, fusion_hidden: int = 16, fusion_output: int = 16):
        super().__init__()
        self.gf_extractor = GazeFixationFrequency()
        self.gd_extractor = GazeDensity(window_size=window_size)
        self.fusion = RHFDFeatureFusion(
            input_dim=2,
            hidden_dim=fusion_hidden,
            output_dim=fusion_output,
        )

    def forward(self, head_dir: torch.Tensor):
        """
        Args:
            head_dir: [B, T, 3] normalized head direction vectors

        Returns:
            gf:       [B, T, 1] gaze fixation frequency
            gd:       [B, T, 1] gaze density
            fusion:   [B, T, fusion_output] fused RHFD features
        """
        gf = self.gf_extractor(head_dir)
        gd = self.gd_extractor(head_dir)
        fusion = self.fusion(gf, gd)
        return gf, gd, fusion
