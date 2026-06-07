import torch
import torch.nn.functional as F


def vMF3_log_likelihood(y_true, mu_pred, kappa_pred):
    cosin_dist = torch.sum(y_true * mu_pred, dim=1)
    log_likelihood = kappa_pred * cosin_dist + torch.log(kappa_pred) - torch.log(1 - torch.exp(-2*kappa_pred)) - kappa_pred

    return log_likelihood


def compute_vMF3_loss(outputs, true_dir):
    mu = outputs['direction']
    kappa = outputs['kappa']

    if true_dir.shape[-1] == 2:
        mu = mu[..., :2] / torch.norm(mu[..., :2], dim=-1, keepdim=True)

    mu = mu.reshape(-1, mu.shape[-1])
    kappa = kappa.reshape(-1, kappa.shape[-1])
    true_dir = true_dir.reshape(-1, true_dir.shape[-1])
    values = vMF3_log_likelihood(true_dir, mu, kappa)
    return -values.mean()


def compute_kappa_vMF3_loss(outputs, true_dir):
    mu = outputs['direction'].detach()
    kappa = outputs['kappa']

    if true_dir.shape[-1] == 2:
        mu = mu[..., :2] / torch.norm(mu[..., :2], dim=-1, keepdim=True)

    mu = mu.reshape(-1, mu.shape[-1])
    kappa = kappa.reshape(-1, kappa.shape[-1])
    true_dir = true_dir.reshape(-1, true_dir.shape[-1])
    values = vMF3_log_likelihood(true_dir, mu, kappa)
    return -values.mean()


def compute_basic_cos_loss(outputs, true_dir):
    """
    Compute integrated loss function of spherical regression
    """

    reg_dir = outputs['direction']

    reg_dir = reg_dir.reshape(-1, reg_dir.shape[-1])
    true_dir = true_dir.reshape(-1, true_dir.shape[-1])

    if true_dir.shape[-1] == 2:
        reg_dir = reg_dir[..., :2] / torch.norm(reg_dir[..., :2], dim=-1, keepdim=True)

    cos = torch.sum(reg_dir * true_dir, dim=-1)
    cos[cos > 1] = 1
    cos[cos < -1] = -1
    loss = 1 - cos

    return loss.mean()


def compute_gaze_probability_loss(
    gaze_probs: torch.Tensor,
    true_gaze_dir: torch.Tensor,
    anchors: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Compute KL-divergence loss between predicted gaze probability distribution
    over spherical anchors and a soft target distribution derived from the
    true gaze direction.

    The soft target is created by computing cosine similarity between the
    true gaze direction and each anchor, then applying softmax with a low
    temperature to create a peaked distribution around the true direction.

    Args:
        gaze_probs:   [B*T, K] predicted probability over K anchors
        true_gaze_dir: [B*T, 3] ground truth gaze direction (normalized)
        anchors:      [K, 3] spherical anchor points
        temperature:  softmax temperature (lower = more peaked target)

    Returns:
        scalar KL-divergence loss
    """
    K = anchors.shape[0]

    # Ensure correct shapes
    gaze_probs_flat = gaze_probs.reshape(-1, K)
    true_dir_flat = true_gaze_dir.reshape(-1, 3)

    # Compute soft target: cosine similarity with temperature
    cos_sim = torch.matmul(true_dir_flat, anchors.T)  # [N, K]
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # Softmax with temperature to create soft target
    soft_target = F.softmax(cos_sim / temperature, dim=-1)

    # KL divergence: sum(target * log(target / pred))
    log_probs = torch.log(gaze_probs_flat + 1e-8)
    kl_div = torch.sum(soft_target * (torch.log(soft_target + 1e-8) - log_probs), dim=-1)

    return kl_div.mean()


def compute_temporal_smoothness_loss(
    gaze_sequence: torch.Tensor,
    p: float = 2.0,
) -> torch.Tensor:
    """
    Penalize abrupt changes between consecutive frames in the gaze sequence.

    This encourages temporally smooth gaze predictions, which is expected
    in natural human gaze behavior.

    Args:
        gaze_sequence: [B, T, D] gaze direction vectors
        p:             norm order for the difference penalty (2 = L2)

    Returns:
        scalar smoothness loss
    """
    # Frame-to-frame differences
    diffs = gaze_sequence[:, 1:] - gaze_sequence[:, :-1]  # [B, T-1, D]

    if p == 2.0:
        smoothness = torch.mean(torch.sum(diffs ** 2, dim=-1))
    else:
        smoothness = torch.mean(torch.sum(torch.abs(diffs) ** p, dim=-1))

    return smoothness


def compute_direction_consistency_loss(
    gaze_direction: torch.Tensor,
    head_direction: torch.Tensor,
    max_angle_deg: float = 90.0,
) -> torch.Tensor:
    """
    Encourage gaze direction to be consistent with head direction.
    Human gaze is usually within ~90° of head direction.

    Penalizes predictions where gaze and head diverges beyond max_angle.

    Args:
        gaze_direction: [B, T, 3] predicted gaze direction
        head_direction: [B, T, 3] head direction (as reference)
        max_angle_deg:  maximum expected angle between gaze and head

    Returns:
        scalar consistency loss
    """
    max_cos = torch.cos(torch.tensor(max_angle_deg * 3.14159 / 180.0,
                                     device=gaze_direction.device))

    cos_sim = torch.sum(gaze_direction * head_direction, dim=-1)
    # Penalize when cosine similarity < max_cos (angle > max_angle)
    violation = torch.clamp(max_cos - cos_sim, min=0.0)

    return violation.mean()
