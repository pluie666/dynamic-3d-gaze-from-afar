"""
GazeNet with RHFD-enhanced Hybrid LSTM-TWIESN Architecture.

Extended from: "Dynamic 3D Gaze from Afar" (Nonaka et al., CVPR 2022)
RHFD integration: Refined Hidden Follower Detection gaze features.

Architecture overview:
  1. HBNet:  extract head/body direction from image + head_mask + body_dv
  2. RHFD Features: extract Gf (fixation freq) & Gd (gaze density) from head_dir
  3. RHFDGazeModule: enhanced LSTM with probability-first gaze mapping
  4. TWIESN: Tchebichef-weighted Echo State Network for temporal smoothing
  5. Exponential Smoothing: EMA on final gaze predictions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.hbnet import HBNet
from models.utils import get_rotation, compute_mae, generate_sphere_anchors
from models.loss import (
    compute_basic_cos_loss,
    compute_kappa_vMF3_loss,
    compute_gaze_probability_loss,
    compute_temporal_smoothness_loss,
)
from models.rhfd_features import RHFDFeatureExtractor
from models.rhfd_mapping import GazeProbabilityHead, GazeDirectRegressionHead
from models.twiesn import TWIESN
from models.smoothing import ExponentialSmoothing, AdaptiveExponentialSmoothing


# ==============================================================================
# Original GazeModule (kept for backward compatibility)
# ==============================================================================

class GazeModule(pl.LightningModule):
    """Original GAFA gaze module with optional RHFD feature channels.

    Input: concatenated [body_dir(3), head_dir(3), + opt rhfd(2)] per timestep.
    Architecture: LSTM(128, bidirectional, 2 layers) → flatten → FC → direction + kappa.
    """

    def __init__(self, n_frames, n_hidden=128, use_rhfd=False):
        super().__init__()
        assert n_frames % 2 == 1
        self.n_frames = n_frames
        self.use_rhfd = use_rhfd

        # Input: body_dir(3) + head_dir(3) [+ gf(1) + gd(1)]
        lstm_input_dim = 6 + (2 if use_rhfd else 0)

        self.lstm = nn.LSTM(lstm_input_dim, n_hidden, bidirectional=True, num_layers=2)
        self.direction_layer = nn.Sequential(
            nn.Linear(2 * n_hidden * n_frames, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * n_frames),
        )
        self.kappa_layer = nn.Sequential(
            nn.Linear(2 * n_hidden * n_frames, 64),
            nn.ReLU(),
            nn.Linear(64, n_frames),
            nn.Softplus()
        )

    def forward(self, x, rhfd_features=None):
        """
        Args:
            x: [B, T, C] LSTM input (body_dir + head_dir, possibly + gf/gd)
            rhfd_features: [B, T, 2] optional gf/gd features
        """
        if self.use_rhfd and rhfd_features is not None:
            x = torch.cat([x, rhfd_features], dim=-1)

        # LSTM
        fc_out, _ = self.lstm(x)
        fc_out = F.relu(fc_out).view(fc_out.shape[0], -1)

        # estimate mean of vMF
        direction = self.direction_layer(fc_out)
        direction = direction.reshape(x.shape[0], x.shape[1], 3)
        direction /= torch.norm(direction, dim=-1, keepdim=True)
        kappa = self.kappa_layer(fc_out).reshape(x.shape[0], x.shape[1], 1)

        output = {
            'direction': direction,
            'kappa': kappa,
            'probs': None,
            'entropy': None,
        }

        return output


# ==============================================================================
# RHFD-Enhanced Gaze Module
# ==============================================================================

class RHFDGazeModule(pl.LightningModule):
    """
    Enhanced Gaze Module with RHFD features and probability-first mapping.

    Input channels (per time step):
      - body_dir * body_kappa  [3]   weighted body direction
      - head_dir * head_kappa  [3]   weighted head direction
      - gf_features            [1]   gaze fixation frequency
      - gd_features            [1]   gaze density
      - rhfd_fusion            [16]  learned fusion of Gf/Gd
      Total: 24 dimensions

    Architecture:
      LSTM(24 → 128, bidirectional, 2 layers) → hidden states [B,T,256]
        ├── GazeProbabilityHead → gaze_probs [B,T,K] → gaze_init [B,T,3]
        └── Kappa head           → kappa [B,T,1]
    """

    def __init__(
        self,
        n_frames: int = 7,
        n_hidden: int = 128,
        n_anchors: int = 64,
        use_probability_head: bool = True,
        use_rhfd_features: bool = True,
        rhfd_fusion_dim: int = 16,
        use_flat_head: bool = False,
    ):
        super().__init__()
        assert n_frames % 2 == 1
        self.n_frames = n_frames
        self.use_flat_head = use_flat_head
        self.use_probability_head = use_probability_head
        self.use_rhfd_features = use_rhfd_features
        self.n_anchors = n_anchors

        # Input dimension depends on whether RHFD features are used
        # Base: body_dir(3) + head_dir(3) = 6
        # RHFD: base(6) + gf(1) + gd(1) + fusion(16) = 24
        if use_rhfd_features:
            lstm_input_dim = 6 + 1 + 1 + rhfd_fusion_dim  # 24
        else:
            lstm_input_dim = 6

        # LSTM encoder
        self.lstm = nn.LSTM(
            lstm_input_dim,
            n_hidden,
            bidirectional=True,
            num_layers=2,
        )
        lstm_output_dim = 2 * n_hidden  # 256 (bidirectional)

        # Gaze prediction head
        if use_probability_head:
            self.gaze_head = GazeProbabilityHead(
                in_dim=lstm_output_dim,
                hidden_dim=128,
                n_anchors=n_anchors,
            )
        else:
            self.gaze_head = GazeDirectRegressionHead(
                in_dim=lstm_output_dim,
                hidden_dim=64,
            )

        # Kappa (concentration) head (used when not using probability head)
        self.kappa_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        # Flat head: matches original GazeModule architecture
        # Flatten T*256 → FC → T*3 (direction) or T (kappa)
        if use_flat_head:
            flat_input_dim = lstm_output_dim * n_frames  # 256 * 7 = 1792
            self._flat_direction_layer = nn.Sequential(
                nn.Linear(flat_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3 * n_frames),
            )
            self._flat_kappa_layer = nn.Sequential(
                nn.Linear(flat_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_frames),
                nn.Softplus(),
            )

    def forward(
        self,
        body_dir_weighted: torch.Tensor,
        head_dir_weighted: torch.Tensor,
        gf_features: torch.Tensor = None,
        gd_features: torch.Tensor = None,
        rhfd_fusion: torch.Tensor = None,
        anchors: torch.Tensor = None,
        hard_mapping: bool = False,
    ) -> dict:
        """
        Args:
            body_dir_weighted: [B, T, 3] body_dir * body_kappa
            head_dir_weighted: [B, T, 3] head_dir * head_kappa
            gf_features:       [B, T, 1] gaze fixation frequency (optional)
            gd_features:       [B, T, 1] gaze density (optional)
            rhfd_fusion:       [B, T, fusion_dim] fused RHFD features (optional)
            anchors:           [K, 3] spherical anchor points (for prob head)
            hard_mapping:      if True, use argmax (inference only)

        Returns:
            dict with 'direction', 'kappa', 'probs', 'entropy', 'hidden'
        """
        B, T, _ = body_dir_weighted.shape

        # Build LSTM input
        lstm_inputs = [body_dir_weighted, head_dir_weighted]

        if self.use_rhfd_features and gf_features is not None:
            lstm_inputs.append(gf_features)
        if self.use_rhfd_features and gd_features is not None:
            lstm_inputs.append(gd_features)
        if self.use_rhfd_features and rhfd_fusion is not None:
            lstm_inputs.append(rhfd_fusion)

        lstm_input = torch.cat(lstm_inputs, dim=-1)  # [B, T, C]

        # LSTM: expects [T, B, C] input format
        lstm_input_t = lstm_input.transpose(0, 1)  # [T, B, C]
        lstm_out, _ = self.lstm(lstm_input_t)       # [T, B, 256]
        lstm_out = lstm_out.transpose(0, 1)          # [B, T, 256]

        # Gaze prediction via probability or direct head
        if self.use_probability_head and anchors is not None:
            gaze_output = self.gaze_head(lstm_out, anchors, hard=hard_mapping)
        elif self.use_flat_head:
            # Match original GazeModule: flatten all frames → FC → reshape
            fc_out = F.relu(lstm_out).reshape(lstm_out.shape[0], -1)  # [B, T*256]
            direction = self._flat_direction_layer(fc_out)             # [B, T*3]
            direction = direction.reshape(B, T, 3)
            direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
            kappa = self._flat_kappa_layer(fc_out).reshape(B, T, 1)   # [B, T, 1]
            gaze_output = {
                'direction': direction,
                'kappa': kappa,
                'probs': None,
                'entropy': None,
            }
        else:
            gaze_output = self.gaze_head(lstm_out)
            gaze_output['kappa'] = self.kappa_layer(lstm_out)

        gaze_output['hidden'] = lstm_out

        return gaze_output


# =============================================================================
# Main GazeNet with RHFD + TWIESN + EMA
# =============================================================================

class GazeNet(pl.LightningModule):
    """
    Extended GazeNet with RHFD features, LSTM-TWIESN hybrid architecture,
    and exponential smoothing.

    Configurable components:
      - use_rhfd_features:      enable Gf/Gd feature extraction
      - use_probability_head:   enable probability-first gaze mapping
      - use_twiesn:             enable TWIESN temporal smoothing
      - use_ema:                enable exponential moving average
      - use_adaptive_ema:       use kappa-adaptive EMA (requires use_ema=True)
    """

    def __init__(
        self,
        n_frames: int = 7,
        # RHFD flags
        use_rhfd_features: bool = True,
        use_probability_head: bool = True,
        # TWIESN flags
        use_twiesn: bool = True,
        twiesn_reservoir_dim: int = 128,
        twiesn_tchebichef_order: int = 4,
        twiesn_spectral_radius: float = 0.9,
        # EMA flags
        use_ema: bool = True,
        ema_alpha: float = 0.6,
        use_adaptive_ema: bool = False,
        # Probability head
        n_anchors: int = 64,
        # Loss weights
        loss_weights: dict = None,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.use_rhfd_features = use_rhfd_features
        self.use_probability_head = use_probability_head
        self.use_twiesn = use_twiesn
        self.use_ema = use_ema
        self.use_adaptive_ema = use_adaptive_ema
        self.n_anchors = n_anchors

        # Loss weights
        self.loss_weights = loss_weights or {
            'cos': 1.0,
            'vmf': 0.5,
            'prob': 0.3,
            'smoothness': 0.01,
        }

        # ---- Submodules ----

        # HBNet: head/body direction estimation (unchanged from original)
        self.hbnet = HBNet()

        # RHFD Feature Extractor
        if use_rhfd_features:
            self.rhfd_extractor = RHFDFeatureExtractor(
                window_size=3,
                fusion_hidden=16,
                fusion_output=16,
            )

        # Enhanced Gaze Module
        self.gazemodule = RHFDGazeModule(
            n_frames=n_frames,
            n_hidden=128,
            n_anchors=n_anchors,
            use_probability_head=use_probability_head,
            use_rhfd_features=use_rhfd_features,
            rhfd_fusion_dim=16,
        )

        # TWIESN: temporal smoothing
        if use_twiesn:
            self.twiesn = TWIESN(
                input_dim=3,
                reservoir_dim=twiesn_reservoir_dim,
                spectral_radius=twiesn_spectral_radius,
                tchebichef_order=twiesn_tchebichef_order,
            )

        # Exponential smoothing
        if use_ema:
            if use_adaptive_ema:
                self.ema = AdaptiveExponentialSmoothing()
            else:
                self.ema = ExponentialSmoothing(alpha=ema_alpha, learnable=True)

        # Spherical anchors (registered as buffer so they move to device with model)
        anchors = generate_sphere_anchors(n_anchors)
        self.register_buffer('anchors', anchors)

        # Manual optimization
        self.automatic_optimization = False

    def load_pretrained_hbnet(self, checkpoint_path: str, map_location: str = 'cpu'):
        """
        Load only HBNet weights from an original GAFA checkpoint.

        The new RHFDGazeModule, TWIESN, and EMA modules have different
        architecture from the original GazeModule, so their weights
        cannot be transferred and will remain randomly initialized.

        Args:
            checkpoint_path: path to .pth checkpoint
            map_location:    device to load weights to
        """
        state = torch.load(checkpoint_path, map_location=map_location)
        if 'state_dict' in state:
            state = state['state_dict']

        # Filter to only HBNet keys, strip "hbnet." prefix
        hbnet_state = {}
        skipped = []
        for k, v in state.items():
            if k.startswith('hbnet.'):
                # Strip the "hbnet." prefix for loading into self.hbnet
                hbnet_state[k[len('hbnet.'):]] = v
            else:
                skipped.append(k)

        # Load HBNet weights (strict=True because only HBNet keys are passed)
        self.hbnet.load_state_dict(hbnet_state, strict=True)

        print(f"Loaded {len(hbnet_state)} HBNet parameters")
        print(f"Skipped {len(skipped)} GazeModule parameters (architecture changed)")
        print(f"New modules (RHFDGazeModule, TWIESN, EMA) are randomly initialized")
        return self

    def forward(
        self,
        img: torch.Tensor,
        head_mask: torch.Tensor,
        body_dv: torch.Tensor,
        hard_mapping: bool = False,
    ) -> tuple:
        """
        Full forward pass through the hybrid architecture.

        Args:
            img:       [B, T, 3, 256, 192] body images
            head_mask: [B, T, 1, 256, 192] head position masks
            body_dv:   [B, T, 2] body velocity in image plane
            hard_mapping: use argmax for probability→direction (inference)

        Returns:
            (gaze_res, head_outputs, body_outputs)
            gaze_res includes: 'direction', 'kappa', 'probs', 'entropy',
                               'direction_init', 'direction_twiesn'
        """
        # ---- Stage 1: HBNet (unchanged) ----
        head_outputs, body_outputs = self.hbnet(img, head_mask, body_dv)

        # Get raw head/body directions (before rotation) for RHFD features
        head_dir_raw = head_outputs['direction']  # [B, T, 3]

        # ---- Stage 2: RHFD Feature Extraction ----
        gf_features, gd_features, rhfd_fusion = None, None, None
        if self.use_rhfd_features:
            with torch.no_grad():
                gf_features, gd_features, rhfd_fusion = self.rhfd_extractor(head_dir_raw.detach())
            gf_features = gf_features.detach()
            gd_features = gd_features.detach()
            rhfd_fusion = rhfd_fusion.detach()

        # ---- Stage 3: Rotation Normalization (unchanged) ----
        reference_rad = head_outputs['direction'][:, self.n_frames // 2]
        dst_rad = torch.zeros_like(reference_rad)
        dst_rad[:, 2] = -1
        R = get_rotation(reference_rad, dst_rad)
        head_dir = torch.einsum('bij,bfj->bfi', R, head_outputs['direction'])
        body_dir = torch.einsum('bij,bfj->bfi', R, body_outputs['direction'])

        # Weight direction with kappa
        head_dir_weighted = head_dir * head_outputs['kappa']
        body_dir_weighted = body_dir * body_outputs['kappa']

        # ---- Stage 4: Enhanced GazeModule (LSTM + Probability Head) ----
        gaze_res = self.gazemodule(
            body_dir_weighted=body_dir_weighted,
            head_dir_weighted=head_dir_weighted,
            gf_features=gf_features,
            gd_features=gd_features,
            rhfd_fusion=rhfd_fusion,
            anchors=self.anchors,
            hard_mapping=hard_mapping,
        )

        # Save initial LSTM estimate for reference
        gaze_init = gaze_res['direction']  # [B, T, 3]

        # ---- Stage 5: TWIESN Temporal Smoothing ----
        gaze_current = gaze_init
        if self.use_twiesn:
            twiesn_res = self.twiesn(gaze_current)
            gaze_twiesn = twiesn_res['direction']  # [B, T, 3]

            # Residual connection: preserve some of the original signal
            gaze_refined = 0.5 * gaze_twiesn + 0.5 * gaze_init
            gaze_refined = gaze_refined / (
                torch.norm(gaze_refined, dim=-1, keepdim=True) + 1e-8
            )

            gaze_res['direction_twiesn'] = gaze_twiesn
            gaze_res['direction_init'] = gaze_init
            gaze_current = gaze_refined

        # ---- Stage 6: Final EMA (single pass) ----
        if self.use_ema:
            if self.use_adaptive_ema:
                ema_res = self.ema(gaze_current, gaze_res['kappa'])
            else:
                ema_res = self.ema(gaze_current)
            gaze_current = ema_res['smoothed']

        # Update final direction
        gaze_res['direction'] = gaze_current

        # ---- Stage 8: Inverse Rotation ----
        gaze_res['direction'] = torch.einsum(
            'bij,bfj->bfi', R.transpose(1, 2), gaze_res['direction']
        )

        # Also rotate intermediate outputs for diagnostics
        if 'direction_twiesn' in gaze_res:
            gaze_res['direction_twiesn'] = torch.einsum(
                'bij,bfj->bfi', R.transpose(1, 2), gaze_res['direction_twiesn']
            )
        if 'direction_init' in gaze_res:
            gaze_res['direction_init'] = torch.einsum(
                'bij,bfj->bfi', R.transpose(1, 2), gaze_res['direction_init']
            )

        return gaze_res, head_outputs, body_outputs

    # =========================================================================
    # Training
    # =========================================================================

    def configure_optimizers(self):
        opt_direction = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=5e-5
        )
        opt_kappa = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=5e-5
        )
        return opt_direction, opt_kappa

    def training_step(self, batch, batch_idx):
        image = batch['image']
        head_mask = batch['head_mask']
        body_dv = batch['body_dv']

        gaze_res, head_res, body_res = self.forward(image, head_mask, body_dv)

        # NaN guard
        if torch.isnan(gaze_res['direction']).any():
            return None

        opt_direction, opt_kappa = self.optimizers()

        if batch_idx % 10 != 0:
            # Direction training steps
            loss_cos = (
                compute_basic_cos_loss(head_res, batch['head_dir']) +
                compute_basic_cos_loss(body_res, batch['body_dir']) +
                compute_basic_cos_loss(gaze_res, batch['gaze_dir'])
            ) / 3.0

            loss_smooth = compute_temporal_smoothness_loss(gaze_res['direction'])
            loss = loss_cos + 0.01 * loss_smooth

            opt_direction.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(opt_direction, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
            opt_direction.step()

            self.log_dict({
                "direction_loss": loss_cos,
                "smoothness_loss": loss_smooth,
            }, prog_bar=True)
        else:
            # Kappa training steps (every 10th batch)
            loss = (
                compute_kappa_vMF3_loss(head_res, batch['head_dir']) +
                compute_kappa_vMF3_loss(body_res, batch['body_dir']) +
                compute_kappa_vMF3_loss(gaze_res, batch['gaze_dir'])
            ) / 3.0

            if torch.isnan(loss):
                return None

            opt_kappa.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(opt_kappa, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
            opt_kappa.step()
            self.log_dict({"kappa_loss": loss}, prog_bar=True)

        mae = compute_mae(gaze_res['direction'], batch['gaze_dir'])
        self.log('train_mae', mae)

        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        head_mask = batch['head_mask']
        body_dv = batch['body_dv']
        gaze_label = batch['gaze_dir']

        gaze_res, _, _ = self.forward(image, head_mask, body_dv)

        # Loss for gaze
        loss = compute_kappa_vMF3_loss(gaze_res, gaze_label)
        mae = compute_mae(gaze_res['direction'], gaze_label)

        self.log("val_mae", mae)
        self.log("val_loss", loss)

        return mae

    def validation_epoch_end(self, outputs):
        val_mae = torch.stack([x for x in outputs]).flatten()
        val_mae_mean = val_mae[~torch.isnan(val_mae)].mean()
        print('MAE (validation): ', val_mae_mean)
        self.log("val_mae", val_mae_mean)

    def test_step(self, batch, batch_idx):
        image = batch['image']
        head_mask = batch['head_mask']
        body_dv = batch['body_dv']
        gaze_label = batch['gaze_dir']

        gaze_res, _, _ = self.forward(
            image, head_mask, body_dv, hard_mapping=True
        )
        prediction = gaze_res['direction']

        if gaze_label.shape[-1] == 3:
            front_index = torch.arange(gaze_label.shape[0])[
                gaze_label[:, 0, -1] <= 0
            ]
            back_index = torch.arange(gaze_label.shape[0])[
                gaze_label[:, 0, -1] > 0
            ]

            # 3D MAE
            mae = compute_mae(prediction, gaze_label)
            front_mae = compute_mae(
                prediction[front_index], gaze_label[front_index]
            )
            back_mae = compute_mae(
                prediction[back_index], gaze_label[back_index]
            )

            gaze_label_2d = (
                gaze_label[..., :2] /
                torch.norm(gaze_label[..., :2], dim=-1, keepdim=True)
            )
            prediction_2d = (
                prediction[..., :2] /
                torch.norm(prediction[..., :2], dim=-1, keepdim=True)
            )

            # 2D MAE
            mae_2d = compute_mae(prediction_2d, gaze_label_2d)
            front_mae_2d = compute_mae(
                prediction_2d[front_index], gaze_label_2d[front_index]
            )
            back_mae_2d = compute_mae(
                prediction_2d[back_index], gaze_label_2d[back_index]
            )

        elif gaze_label.shape[-1] == 2:
            gaze_label_2d = (
                gaze_label[..., :2] /
                torch.norm(gaze_label[..., :2], dim=-1, keepdim=True)
            )
            prediction_2d = (
                prediction[..., :2] /
                torch.norm(prediction[..., :2], dim=-1, keepdim=True)
            )
            mae = front_mae = back_mae = 0
            front_mae_2d = back_mae_2d = 0

            mae_2d = compute_mae(prediction_2d, gaze_label_2d)
            print(mae_2d)

        return mae, mae_2d, front_mae, front_mae_2d, back_mae, back_mae_2d

    def test_epoch_end(self, outputs):
        mae = np.nanmean([x[0] for x in outputs])
        mae_2d = np.nanmean([x[1] for x in outputs])
        front_mae = np.nanmean([x[2] for x in outputs])
        front_mae_2d = np.nanmean([x[3] for x in outputs])
        back_mae = np.nanmean([x[4] for x in outputs])
        back_mae_2d = np.nanmean([x[5] for x in outputs])

        print('MAE (3D front): ', front_mae)
        print('MAE (2D front): ', front_mae_2d)
        print('MAE (3D back): ', back_mae)
        print('MAE (2D back): ', back_mae_2d)
        print('MAE (3D all): ', mae)
        print('MAE (2D all): ', mae_2d)


# =============================================================================
# Simple RHFD-Enhanced GazeNet — minimal, stable enhancement
# =============================================================================

class SimpleRHFDGazeNet(pl.LightningModule):
    """
    Minimal RHFD-enhanced gaze network using the proven original GazeModule
    architecture with 2 extra input channels (Gf + Gd).

    NO TWIESN, NO EMA, NO probability head — just the reliable GAFA pipeline
    enhanced with RHFD fixation frequency and density features.

    This is designed for stable multi-scene training.
    """

    def __init__(self, n_frames: int = 7):
        super().__init__()
        self.n_frames = n_frames

        self.hbnet = HBNet()
        self.gazemodule = GazeModule(n_frames, use_rhfd=True)
        self.rhfd_extractor = RHFDFeatureExtractor(window_size=3)

        self.automatic_optimization = False

    def load_pretrained_hbnet(self, checkpoint_path: str, map_location: str = 'cpu'):
        """Load HBNet weights from original GAFA checkpoint."""
        state = torch.load(checkpoint_path, map_location=map_location)
        if 'state_dict' in state:
            state = state['state_dict']
        hbnet_state = {}
        for k, v in state.items():
            if k.startswith('hbnet.'):
                hbnet_state[k[len('hbnet.'):]] = v
        self.hbnet.load_state_dict(hbnet_state, strict=True)
        print(f"Loaded {len(hbnet_state)} HBNet parameters")
        return self

    def forward(self, img, head_mask, body_dv):
        # HBNet
        head_outputs, body_outputs = self.hbnet(img, head_mask, body_dv)

        # RHFD features from head direction (detached — computed features, no gradient)
        with torch.no_grad():
            gf, gd, _ = self.rhfd_extractor(head_outputs['direction'].detach())
        rhfd_concat = torch.cat([gf, gd], dim=-1).detach()  # [B,T,2]

        # Rotation normalization
        reference_rad = head_outputs['direction'][:, self.n_frames // 2]
        dst_rad = torch.zeros_like(reference_rad)
        dst_rad[:, 2] = -1
        R = get_rotation(reference_rad, dst_rad)
        head_dir = torch.einsum('bij,bfj->bfi', R, head_outputs['direction'])
        body_dir = torch.einsum('bij,bfj->bfi', R, body_outputs['direction'])

        head_dir_weighted = head_dir * head_outputs['kappa']
        body_dir_weighted = body_dir * body_outputs['kappa']

        # GazeModule with RHFD features
        gaze_input = torch.cat([body_dir_weighted, head_dir_weighted], dim=2)
        gaze_res = self.gazemodule(gaze_input, rhfd_concat)

        # Inverse rotation
        gaze_res['direction'] = torch.einsum(
            'bij,bfj->bfi', R.transpose(1, 2), gaze_res['direction']
        )

        return gaze_res, head_outputs, body_outputs

    def configure_optimizers(self):
        opt_direction = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4
        )
        opt_kappa = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4
        )
        return opt_direction, opt_kappa

    def training_step(self, batch, batch_idx):
        image = batch['image']
        head_mask = batch['head_mask']
        body_dv = batch['body_dv']

        gaze_res, head_res, body_res = self.forward(image, head_mask, body_dv)

        opt_direction, opt_kappa = self.optimizers()

        if batch_idx % 10 != 0:
            loss = (
                compute_basic_cos_loss(head_res, batch['head_dir']) +
                compute_basic_cos_loss(body_res, batch['body_dir']) +
                compute_basic_cos_loss(gaze_res, batch['gaze_dir'])
            ) / 3.0
            opt_direction.zero_grad()
            self.manual_backward(loss)
            opt_direction.step()
            self.log_dict({"direction_loss": loss}, prog_bar=True)
        else:
            loss = (
                compute_kappa_vMF3_loss(head_res, batch['head_dir']) +
                compute_kappa_vMF3_loss(body_res, batch['body_dir']) +
                compute_kappa_vMF3_loss(gaze_res, batch['gaze_dir'])
            ) / 3.0
            opt_kappa.zero_grad()
            self.manual_backward(loss)
            opt_kappa.step()
            self.log_dict({"kappa_loss": loss}, prog_bar=True)

        mae = compute_mae(gaze_res['direction'], batch['gaze_dir'])
        self.log('train_mae', mae)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        head_mask = batch['head_mask']
        body_dv = batch['body_dv']
        gaze_label = batch['gaze_dir']

        gaze_res, _, _ = self.forward(image, head_mask, body_dv)

        loss = compute_kappa_vMF3_loss(gaze_res, gaze_label)
        mae = compute_mae(gaze_res['direction'], gaze_label)

        self.log("val_mae", mae)
        self.log("val_loss", loss)
        return mae

    def validation_epoch_end(self, outputs):
        val_mae = torch.stack([x for x in outputs]).flatten()
        val_mae_mean = val_mae[~torch.isnan(val_mae)].mean()
        print('MAE (validation): ', val_mae_mean)
        self.log("val_mae", val_mae_mean)

    def test_step(self, batch, batch_idx):
        image = batch['image']
        head_mask = batch['head_mask']
        body_dv = batch['body_dv']
        gaze_label = batch['gaze_dir']

        gaze_res, _, _ = self.forward(image, head_mask, body_dv)
        prediction = gaze_res['direction']

        if gaze_label.shape[-1] == 3:
            front_index = torch.arange(gaze_label.shape[0])[gaze_label[:, 0, -1] <= 0]
            back_index = torch.arange(gaze_label.shape[0])[gaze_label[:, 0, -1] > 0]

            mae = compute_mae(prediction, gaze_label)
            front_mae = compute_mae(prediction[front_index], gaze_label[front_index])
            back_mae = compute_mae(prediction[back_index], gaze_label[back_index])

            gaze_label_2d = gaze_label[..., :2] / torch.norm(gaze_label[..., :2], dim=-1, keepdim=True)
            prediction_2d = prediction[..., :2] / torch.norm(prediction[..., :2], dim=-1, keepdim=True)

            mae_2d = compute_mae(prediction_2d, gaze_label_2d)
            front_mae_2d = compute_mae(prediction_2d[front_index], gaze_label_2d[front_index])
            back_mae_2d = compute_mae(prediction_2d[back_index], gaze_label_2d[back_index])

        elif gaze_label.shape[-1] == 2:
            gaze_label_2d = gaze_label[..., :2] / torch.norm(gaze_label[..., :2], dim=-1, keepdim=True)
            prediction_2d = prediction[..., :2] / torch.norm(prediction[..., :2], dim=-1, keepdim=True)
            mae = front_mae = back_mae = 0
            front_mae_2d = back_mae_2d = 0
            mae_2d = compute_mae(prediction_2d, gaze_label_2d)

        return mae, mae_2d, front_mae, front_mae_2d, back_mae, back_mae_2d

    def test_epoch_end(self, outputs):
        mae = np.nanmean([x[0] for x in outputs])
        mae_2d = np.nanmean([x[1] for x in outputs])
        front_mae = np.nanmean([x[2] for x in outputs])
        front_mae_2d = np.nanmean([x[3] for x in outputs])
        back_mae = np.nanmean([x[4] for x in outputs])
        back_mae_2d = np.nanmean([x[5] for x in outputs])

        print('MAE (3D front): ', front_mae)
        print('MAE (2D front): ', front_mae_2d)
        print('MAE (3D back): ', back_mae)
        print('MAE (2D back): ', back_mae_2d)
        print('MAE (3D all): ', mae)
        print('MAE (2D all): ', mae_2d)
