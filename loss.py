"""
loss.py
-------
Combined Dice + BCE loss for binary change detection.

Design rationale:
  - BCE with pos_weight: penalises missing change pixels more heavily,
    directly countering the ~7% change pixel class imbalance.
  - Dice Loss: optimises pixel-overlap directly, independent of class
    frequency — robust to imbalance by design.
  - Combined: BCE provides per-pixel gradient signal while Dice Loss
    provides global overlap optimisation.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Computes:
        Dice = (2 * |P ∩ T| + smooth) / (|P| + |T| + smooth)
        Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits)
        p_flat = probs.view(-1)
        t_flat = targets.view(-1)

        intersection = (p_flat * t_flat).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (p_flat.sum() + t_flat.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE loss.

    Parameters
    ----------
    bce_weight  : Weight for the BCE component (default 0.5)
    dice_weight : Weight for the Dice component (default 0.5)
    pos_weight  : Positive class weight for BCE — set to
                  (num_negative / num_positive) ≈ 10 for this dataset
    device      : torch.device (needed to place pos_weight tensor)
    """

    def __init__(self, bce_weight: float = 0.5,
                 dice_weight: float = 0.5,
                 pos_weight: float = 10.0,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
