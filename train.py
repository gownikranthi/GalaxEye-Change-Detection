"""
train.py
--------
Training script for EO-SAR binary change detection.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume /path/to/checkpoint.pth
"""

import os
import random
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from dataset import ChangeDetectionDataset
from model import build_model, count_parameters
from loss import CombinedLoss
from utils import compute_metrics


# ── Argument parser ───────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train change detection model"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


# ── Reproducibility ───────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ── One epoch ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu())
        all_targets.append(masks.detach().cpu())

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} "
                  f" Loss: {loss.item():.4f}", end="\r")

    preds_bin = (torch.sigmoid(torch.cat(all_preds)) > 0.5)\
                .numpy().flatten()
    targets   = torch.cat(all_targets).numpy().flatten()
    metrics   = compute_metrics(preds_bin, targets)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)
            loss    = criterion(outputs, masks)
            total_loss += loss.item()
            all_preds.append(outputs.cpu())
            all_targets.append(masks.cpu())

    preds_bin = (torch.sigmoid(torch.cat(all_preds)) > 0.5)\
                .numpy().flatten()
    targets   = torch.cat(all_targets).numpy().flatten()
    metrics   = compute_metrics(preds_bin, targets)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ── Main ──────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Setup
    set_seed(cfg["training"]["seed"])
    device = torch.device(
        cfg["training"]["device"]
        if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg["paths"]["save_dir"], exist_ok=True)

    # Dataset
    print("\nLoading dataset from HuggingFace...")
    ds = load_dataset(cfg["dataset"]["name"])

    image_size = cfg["dataset"]["image_size"]
    train_ds = ChangeDetectionDataset(ds["train"],      "train", image_size)
    val_ds   = ChangeDetectionDataset(ds["validation"], "val",   image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = cfg["dataset"]["num_workers"],
        pin_memory  = True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["dataset"]["num_workers"],
        pin_memory  = True
    )
    print(f"Train triplets: {len(train_ds)} | Val triplets: {len(val_ds)}")

    # Model
    model = build_model(
        architecture = cfg["model"]["architecture"],
        backbone     = cfg["model"]["backbone"],
        in_channels  = cfg["model"]["in_channels"],
        num_classes  = cfg["model"]["num_classes"],
        pretrained   = cfg["model"]["pretrained"],
    ).to(device)
    count_parameters(model)

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(
        bce_weight  = cfg["loss"]["bce_weight"],
        dice_weight = cfg["loss"]["dice_weight"],
        pos_weight  = cfg["loss"]["pos_weight"],
        device      = device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["optimizer"]["lr"],
        weight_decay = cfg["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cfg["scheduler"]["T_max"],
        eta_min = cfg["scheduler"]["eta_min"],
    )

    # Resume from checkpoint
    start_epoch = 0
    best_f1     = 0.0
    best_iou    = 0.0

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_f1     = ckpt["val_f1"]
        best_iou    = ckpt["val_iou"]
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch} | "
              f"Best F1={best_f1:.4f}")

    # Training loop
    ckpt_path = os.path.join(
        cfg["paths"]["save_dir"],
        cfg["paths"]["checkpoint_name"]
    )

    print(f"\n{'='*60}")
    print(f"Starting training for {cfg['training']['epochs']} epochs")
    print(f"{'='*60}")

    for epoch in range(start_epoch + 1, cfg["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")
        print("-" * 40)

        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_m   = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  TRAIN → Loss: {train_m['loss']:.4f} | "
              f"IoU: {train_m['iou']:.4f} | F1: {train_m['f1']:.4f} | "
              f"Prec: {train_m['precision']:.4f} | "
              f"Rec: {train_m['recall']:.4f}")
        print(f"  VAL   → Loss: {val_m['loss']:.4f} | "
              f"IoU: {val_m['iou']:.4f} | F1: {val_m['f1']:.4f} | "
              f"Prec: {val_m['precision']:.4f} | "
              f"Rec: {val_m['recall']:.4f}")

        # Save best model
        if val_m["f1"] > best_f1:
            best_f1  = val_m["f1"]
            best_iou = val_m["iou"]
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_f1"      : best_f1,
                "val_iou"     : best_iou,
                "config"      : cfg,
            }, ckpt_path)
            print(f"  ✅ Best model saved! "
                  f"F1={best_f1:.4f}  IoU={best_iou:.4f}")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best Val F1 : {best_f1:.4f}")
    print(f"Best Val IoU: {best_iou:.4f}")
    print(f"Checkpoint  : {ckpt_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
