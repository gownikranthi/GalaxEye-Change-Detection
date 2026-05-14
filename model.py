"""
model.py
--------
UNet with pretrained ResNet34 encoder for EO-SAR change detection.

Architecture:
  - Early fusion: EO(3ch) + SAR(1ch) → 4-channel input
  - Encoder: ResNet34 (ImageNet pretrained)
  - Decoder: UNet-style with skip connections
  - Output: single-channel logit map (apply sigmoid for probability)

Library: segmentation-models-pytorch (smp)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(
    architecture: str = "Unet",
    backbone: str = "resnet34",
    in_channels: int = 4,
    num_classes: int = 1,
    pretrained: str = "imagenet",
) -> nn.Module:
    """
    Build and return the segmentation model.

    Parameters
    ----------
    architecture : SMP architecture name (e.g. 'Unet', 'FPN', 'DeepLabV3Plus')
    backbone     : Encoder backbone (e.g. 'resnet34', 'efficientnet-b2')
    in_channels  : Number of input channels (4 = EO 3ch + SAR 1ch)
    num_classes  : Number of output classes (1 for binary)
    pretrained   : Pretrained weights for encoder ('imagenet' or None)

    Returns
    -------
    nn.Module
    """
    arch = getattr(smp, architecture)
    model = arch(
        encoder_name    = backbone,
        encoder_weights = pretrained,
        in_channels     = in_channels,
        classes         = num_classes,
        activation      = None,         # raw logits; sigmoid applied in loss
    )
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str,
                    device: torch.device) -> dict:
    """
    Load model weights from a checkpoint file.

    Parameters
    ----------
    model           : The model to load weights into
    checkpoint_path : Path to .pth checkpoint file
    device          : torch.device

    Returns
    -------
    dict containing epoch, val_f1, val_iou, config
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"  Val F1  = {ckpt['val_f1']:.4f}")
    print(f"  Val IoU = {ckpt['val_iou']:.4f}")
    return ckpt


def count_parameters(model: nn.Module) -> None:
    """Print total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
