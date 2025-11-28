import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from navsim.agents.hidden.hidden_config import HiddenConfig


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class LossComputer(nn.Module):
    def __init__(self, config: HiddenConfig):
        self._config = config
        super(LossComputer, self).__init__()
        # self.focal_loss = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0, activated=False)
        self.cls_loss_weight = config.trajectory_cls_weight
        self.reg_loss_weight = config.trajectory_reg_weight

    def forward(self, poses_reg, poses_cls, targets, plan_anchor):
        """
        Ego-only loss
        poses_reg: (bs, 16, 20, 8, 3)
        poses_cls: (bs, 16, 20)
        targets['trajectory']: (bs, 8, 3)
        targets['neighbour_trajectories']: (bs,15,8,3)
        plan_anchor: (bs, 16, 20, 8, 2)
        """

        bs, _, num_mode, ts, d = poses_reg.shape

        # --- Keep only ego (first agent) ---
        target_traj = targets["trajectory"].unsqueeze(1)  # (bs, 1, ts, 3)
        poses_reg = poses_reg[:, 0:1, ...]  # (bs, 1, num_mode, ts, 3)
        poses_cls = poses_cls[:, 0:1, ...]  # (bs, 1, num_mode)
        plan_anchor = plan_anchor[:, 0:1, ...]  # (bs, 1, num_mode, ts, 2)

        # --- Expand target to match modes ---
        target_traj_exp = target_traj.unsqueeze(2).expand(bs, 1, num_mode, ts, d)

        # --- Find closest mode ---
        dist = torch.linalg.norm(target_traj_exp[..., :2] - plan_anchor, dim=-1)  # (bs,1,num_mode,ts)
        dist = dist.mean(dim=-1)  # (bs,1,num_mode)
        mode_idx = torch.argmin(dist, dim=-1)  # (bs,1)
        cls_target = mode_idx.squeeze(1)  # (bs,)

        # --- Gather best regression predictions ---
        # cls_target: (bs,)
        mode_idx_exp = cls_target[:, None, None, None, None].long()  # (bs,1,1,1,1)

        # gather along dim=2 (num_mode)
        best_reg = torch.gather(
            poses_reg, 2, mode_idx_exp.expand(-1, 1, 1, ts, d)
        ).squeeze(2)  # (bs,1,ts,d)

        target_best = torch.gather(
            target_traj_exp, 2, mode_idx_exp.expand(-1, 1, 1, ts, d)
        ).squeeze(2)  # (bs,1,ts,d)

        # --- Classification loss (focal loss) ---
        target_onehot = torch.zeros_like(poses_cls)
        target_onehot.scatter_(2, cls_target[:, None, None], 1)
        loss_cls = self.cls_loss_weight * py_sigmoid_focal_loss(
            poses_cls,
            target_onehot,
            weight=None,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            avg_factor=None
        )

        # Stop line loss (FIXED)
        traj_xy = best_reg[..., :2]  # predicted XY

        # --- Line Geometry ---
        p0 = targets['stop_lines'][:, 0, :]  # stop line point 0: (B, 2)
        p1 = targets['stop_lines'][:, 1, :]  # stop line point 1: (B, 2)
        line_vec = p1 - p0  # Vector along the stop line (L_x, L_y)

        # 1. Calculate the Unit Normal Vector (Perpendicular to line_vec)
        # We calculate the 2D perpendicular vector: N = (L_y, -L_x)
        perp_vec = torch.stack([line_vec[..., 1], -line_vec[..., 0]], dim=-1)

        # Normalize to get the unit normal vector (n-hat)
        norm = torch.norm(perp_vec, dim=-1, keepdim=True) + 1e-6
        unit_normal = perp_vec / norm  # Shape (B, 2)

        # 2. Vector from p0 to waypoint
        # vec_to_wp = w - p0, broadcasted to all waypoints
        vec_to_wp = traj_xy - p0[:, None, None, :]  # Shape (B, M, T, 2)

        # 3. Signed Perpendicular Distance (d_parallel)
        # This is the distance from the infinite line defined by p0 and p1.
        # d_parallel = (w - p0) . n-hat
        signed_distance = torch.sum(vec_to_wp * unit_normal[:, None, None, :], dim=-1)  # Shape (B, M, T)

        # 4. Hinge-Style Loss and Masking
        tl_mask = targets['traffic_light_state'][:, None, None].float()  # traffic light mask 1.0 = red/stop
        # Clamped distance: Only penalize POSITIVE distances (crossings)
        distance_violation = torch.clamp(signed_distance, min=0.0)

        # Combined Mask: Penalty only when light is red (tl_mask) AND crossing occurred (distance_violation > 0)
        combined_loss_term = distance_violation * tl_mask

        stop_line_loss = combined_loss_term.mean()
        # --- End of Stop Line Loss ---

        # --- Regression loss ---
        # NOTE: The scale * 2 is a hyperparameter for weighting the stop line loss
        reg_loss = (self.reg_loss_weight * F.l1_loss(best_reg, target_best)) + (self.reg_loss_weight * stop_line_loss)

        # --- Total loss ---
        ret_loss = loss_cls + reg_loss
        return ret_loss
