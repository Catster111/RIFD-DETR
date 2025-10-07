"""RT-DETRv2 keypoint head with adaptive heatmap loss."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


POLAR_RADIUS_SCALE = math.sqrt(2.0) / 2.0


def heatmap_expectation_xy(heatmap,
                           domain: str = 'cartesian',
                           radius_scale: float = POLAR_RADIUS_SCALE):
    """Convert a heatmap distribution into expected (x, y) coordinates in bbox-normalized space."""
    height, width = heatmap.shape[-2:]
    device = heatmap.device
    orig_dtype = heatmap.dtype

    if domain == 'polar':
        heatmap_f32 = heatmap.to(torch.float32)
        radius_bins = torch.linspace(0.0, 1.0, height, device=device, dtype=torch.float32)
        angle_bins = torch.linspace(0.0, 2.0 * math.pi, width + 1, device=device, dtype=torch.float32)[:-1]

        prob_radius = heatmap_f32.sum(dim=-1)
        prob_angle = heatmap_f32.sum(dim=-2)

        radius = torch.dot(prob_radius, radius_bins)
        cos_mean = torch.dot(prob_angle, torch.cos(angle_bins))
        sin_mean = torch.dot(prob_angle, torch.sin(angle_bins))
        theta = torch.atan2(sin_mean, cos_mean)

        base_x = 0.5 + radius * radius_scale * torch.cos(theta)
        base_y = 0.5 + radius * radius_scale * torch.sin(theta)
        return base_x.to(orig_dtype), base_y.to(orig_dtype)

    dtype = orig_dtype
    coord_x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    coord_y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    prob_x = heatmap.sum(dim=-2)
    prob_y = heatmap.sum(dim=-1)
    base_x = torch.dot(prob_x, coord_x)
    base_y = torch.dot(prob_y, coord_y)
    return base_x, base_y


def xy_to_polar(x,
                y,
                radius_scale: float = POLAR_RADIUS_SCALE):
    """Convert bbox-normalized (x, y) into normalized polar (radius, theta)."""
    dx = x - 0.5
    dy = y - 0.5
    radius = torch.sqrt(dx ** 2 + dy ** 2) / radius_scale
    radius = torch.clamp(radius, 0.0, 1.0)
    theta = torch.atan2(dy, dx)
    theta = torch.remainder(theta, 2.0 * math.pi)
    return radius, theta


class RTDETRKeypointHead(nn.Module):
    """Keypoint head that produces softmax heatmaps and offset refinements."""

    def __init__(self,
                 hidden_dim=256,
                 num_keypoints=5,
                 heatmap_size=64,
                 heatmap_channels=128,
                 heatmap_domain: str = 'cartesian'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.heatmap_channels = heatmap_channels
        if heatmap_domain not in {'cartesian', 'polar'}:
            raise ValueError(f"Unsupported heatmap domain: {heatmap_domain}")
        self.heatmap_domain = heatmap_domain
        self.polar_radius_scale = POLAR_RADIUS_SCALE

        # project query features to spatial feature maps
        self.pre_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, heatmap_channels * heatmap_size * heatmap_size),
            nn.ReLU(inplace=True),
        )

        # lightweight conv head preserves locality before logits
        self.conv_head = nn.Sequential(
            nn.Conv2d(heatmap_channels, heatmap_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(heatmap_channels, num_keypoints, kernel_size=1),
        )

        # offset regressor operates on query features only
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_keypoints * 2),
            nn.Tanh(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, query_features):
        batch_size, num_queries, _ = query_features.shape
        H = W = self.heatmap_size

        spatial_feat = self.pre_proj(query_features)
        spatial_feat = spatial_feat.view(batch_size * num_queries, self.heatmap_channels, H, W)
        logits = self.conv_head(spatial_feat)
        logits = logits.view(batch_size, num_queries, self.num_keypoints, H, W)

        # probability maps (per keypoint softmax over spatial locations)
        heatmaps = F.softmax(logits.view(batch_size, num_queries, self.num_keypoints, -1), dim=-1)
        heatmaps = heatmaps.view(batch_size, num_queries, self.num_keypoints, H, W)

        offsets = self.offset_head(query_features)
        offsets = offsets.view(batch_size, num_queries, self.num_keypoints, 2)
        offsets = offsets * 0.05  # ±5% of bbox size

        return heatmaps, offsets

    def decode_keypoints(self, heatmaps, offsets, bbox_coords):
        batch_size, num_queries, num_keypoints, H, W = heatmaps.shape
        device = heatmaps.device

        decoded = []
        for b in range(batch_size):
            batch_kps = []
            for q in range(num_queries):
                box = bbox_coords[b, q]
                x1, y1, x2, y2 = box
                bbox_w = x2 - x1
                bbox_h = y2 - y1

                query_kps = []
                for k in range(num_keypoints):
                    prob = heatmaps[b, q, k]
                    base_x, base_y = heatmap_expectation_xy(prob, self.heatmap_domain, self.polar_radius_scale)

                    offset = offsets[b, q, k]
                    final_x_norm = torch.clamp(base_x + offset[0], 0, 1)
                    final_y_norm = torch.clamp(base_y + offset[1], 0, 1)

                    final_x = x1 + final_x_norm * bbox_w
                    final_y = y1 + final_y_norm * bbox_h
                    confidence = prob.max()
                    query_kps.append(torch.tensor([final_x, final_y, confidence], device=device))

                batch_kps.append(torch.stack(query_kps))
            decoded.append(torch.stack(batch_kps))

        return torch.stack(decoded)


class KeypointLoss(nn.Module):
    """Adaptive Gaussian heatmap + offset loss."""

    def __init__(
        self,
        heatmap_weight=1.0,
        offset_weight=0.1,
        min_sigma=1.0,
        max_sigma=5.0,
        sigma_scale=0.5,
        side_margin=0.05,
        gaussian_sigma=1.0,
        coord_weight=1.0,
        entropy_weight=0.05,
        heatmap_domain: str = 'cartesian',
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_scale = sigma_scale
        self.side_margin = side_margin
        self.gaussian_sigma = gaussian_sigma
        self.coord_weight = coord_weight
        self.entropy_weight = entropy_weight
        if heatmap_domain not in {'cartesian', 'polar'}:
            raise ValueError(f"Unsupported heatmap domain: {heatmap_domain}")
        self.heatmap_domain = heatmap_domain
        self.polar_radius_scale = POLAR_RADIUS_SCALE

    def _generate_gaussian_heatmap(self,
                                   keypoint_coords,
                                   sigma,
                                   heatmap_size=64,
                                   device=None,
                                   side=None,
                                   width=None):
        if self.heatmap_domain == 'polar':
            return self._generate_polar_heatmap(
                keypoint_coords, sigma, heatmap_size, width, device
            )

        if isinstance(keypoint_coords, (list, tuple)):
            x, y = keypoint_coords
        else:
            x, y = keypoint_coords[0], keypoint_coords[1]

        if device is None:
            device = keypoint_coords.device if hasattr(keypoint_coords, 'device') else (
                torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device)
            y = torch.tensor(y, device=device)

        x_hm = x * (heatmap_size - 1)
        y_hm = y * (heatmap_size - 1)

        xx, yy = torch.meshgrid(
            torch.arange(heatmap_size, dtype=torch.float32, device=device),
            torch.arange(heatmap_size, dtype=torch.float32, device=device),
            indexing='xy'
        )
        gaussian = torch.exp(-((xx - x_hm) ** 2 + (yy - y_hm) ** 2) / (2 * sigma ** 2))
        sum_val = torch.sum(gaussian)
        if sum_val > 0:
            gaussian = gaussian / sum_val

        if side is not None:
            x_norm = xx / max(heatmap_size - 1, 1)
            if side == 'left':
                boundary = x + self.side_margin
                penalty = torch.exp(-((x_norm - boundary).clamp(min=0) / self.side_margin) ** 2)
                gaussian = gaussian * penalty
            elif side == 'right':
                boundary = x - self.side_margin
                penalty = torch.exp(-((boundary - x_norm).clamp(min=0) / self.side_margin) ** 2)
                gaussian = gaussian * penalty

        return gaussian

    def _generate_polar_heatmap(self, keypoint_coords, sigma, height, width, device):
        if width is None:
            width = height

        if isinstance(keypoint_coords, (list, tuple)):
            x, y = keypoint_coords
        else:
            x, y = keypoint_coords[0], keypoint_coords[1]

        if device is None:
            device = keypoint_coords.device if hasattr(keypoint_coords, 'device') else (
                torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device)
            y = torch.tensor(y, device=device)

        work_dtype = torch.float32
        radius, theta = xy_to_polar(x.to(work_dtype), y.to(work_dtype), self.polar_radius_scale)
        radius_bins = torch.linspace(0.0, 1.0, height, device=device, dtype=work_dtype)
        angle_bins = torch.linspace(0.0, 2.0 * math.pi, width + 1, device=device, dtype=work_dtype)[:-1]
        rr, tt = torch.meshgrid(radius_bins, angle_bins, indexing='ij')

        if not torch.is_tensor(sigma):
            sigma = torch.tensor(float(sigma), device=device, dtype=work_dtype)
        else:
            sigma = sigma.to(work_dtype)
        sigma_radius = sigma / max(height - 1, 1)
        sigma_angle = sigma / max(width - 1, 1) * (2.0 * math.pi)
        sigma_radius = torch.clamp(sigma_radius, min=1e-6)
        sigma_angle = torch.clamp(sigma_angle, min=1e-6)

        radius_term = (rr - radius) ** 2 / (2.0 * sigma_radius ** 2)
        angle_diff = torch.atan2(torch.sin(tt - theta), torch.cos(tt - theta))
        angle_term = (angle_diff ** 2) / (2.0 * sigma_angle ** 2)
        gaussian = torch.exp(-(radius_term + angle_term))

        sum_val = torch.sum(gaussian)
        if sum_val > 0:
            gaussian = gaussian / sum_val

        return gaussian.to(x.dtype if torch.is_tensor(x) else torch.float32)

    def compute_heatmap_loss(self, pred_heatmaps, gt_keypoints, valid_queries):
        batch_size, num_queries, num_keypoints, H, W = pred_heatmaps.shape
        device = pred_heatmaps.device
        dtype = pred_heatmaps.dtype
        gt_heatmaps = torch.zeros_like(pred_heatmaps)

        for b in range(batch_size):
            for q in range(num_queries):
                if not valid_queries[b, q]:
                    continue
                for k in range(num_keypoints):
                    x, y, visibility = gt_keypoints[b, q, k]
                    if visibility > 0:
                        sigma_val = torch.tensor(float(self.gaussian_sigma), device=device, dtype=dtype)
                        side = None
                        if k == 0:
                            side = 'left'
                        elif k == 1:
                            side = 'right'
                        gt_heatmap = self._generate_gaussian_heatmap(
                            [x, y], sigma_val, heatmap_size=H, device=device, side=side, width=W
                        )
                        gt_heatmaps[b, q, k] = gt_heatmap.to(dtype)

        total_loss = torch.tensor(0.0, device=device)
        total_valid = 0

        for b in range(batch_size):
            for q in range(num_queries):
                if not valid_queries[b, q]:
                    continue
                for k in range(num_keypoints):
                    x, y, visibility = gt_keypoints[b, q, k]
                    if visibility > 0:
                        pred = pred_heatmaps[b, q, k]
                        gt = gt_heatmaps[b, q, k]
                        pred_clamped = torch.clamp(pred, 1e-6, 1.0)

                        # Cross-entropy between GT distribution and predicted softmax heatmap
                        heatmap_loss = -(gt * torch.log(pred_clamped)).sum()

                        base_x, base_y = heatmap_expectation_xy(
                            pred, self.heatmap_domain, self.polar_radius_scale
                        )
                        pred_coord = torch.stack([base_x.to(dtype), base_y.to(dtype)])
                        gt_coord = torch.stack([x.to(dtype), y.to(dtype)])
                        coord_loss = F.smooth_l1_loss(pred_coord, gt_coord)

                        entropy = -(pred * torch.log(pred_clamped)).sum()
                        entropy_loss = entropy / (H * W)

                        loss = heatmap_loss + self.coord_weight * coord_loss + self.entropy_weight * entropy_loss
                        total_loss += loss
                        total_valid += 1

        return total_loss / max(total_valid, 1)

    def compute_offset_loss(self, pred_heatmaps, pred_offsets, gt_keypoints, valid_queries):
        batch_size, num_queries, num_keypoints, H, W = pred_heatmaps.shape
        device = pred_offsets.device
        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0

        for b in range(batch_size):
            for q in range(num_queries):
                if not valid_queries[b, q]:
                    continue
                for k in range(num_keypoints):
                    gt_x, gt_y, visibility = gt_keypoints[b, q, k]
                    if visibility > 0:
                        heatmap = pred_heatmaps[b, q, k]
                        base_x, base_y = heatmap_expectation_xy(
                            heatmap, self.heatmap_domain, self.polar_radius_scale
                        )

                        target_offset = torch.tensor(
                            [gt_x - base_x, gt_y - base_y], device=device, dtype=pred_offsets.dtype
                        )
                        target_offset = torch.clamp(target_offset, -1.0, 1.0)
                        offset_loss = F.smooth_l1_loss(pred_offsets[b, q, k], target_offset, reduction='mean')
                        if torch.isnan(offset_loss) or torch.isinf(offset_loss):
                            offset_loss = torch.tensor(0.0, device=device, dtype=pred_offsets.dtype)
                        total_loss += offset_loss
                        num_valid += 1

        return total_loss / max(num_valid, 1)

    def forward(self, pred_heatmaps, pred_offsets, gt_keypoints, valid_queries, gt_keypoint_sigmas=None):
        heatmap_loss = self.compute_heatmap_loss(
            pred_heatmaps,
            gt_keypoints,
            valid_queries,
        )
        offset_loss = self.compute_offset_loss(pred_heatmaps, pred_offsets, gt_keypoints, valid_queries)
        total_loss = self.heatmap_weight * heatmap_loss + self.offset_weight * offset_loss
        return {
            'loss_keypoint': total_loss,
            'loss_keypoint_heatmap': heatmap_loss,
            'loss_keypoint_offset': offset_loss,
        }
