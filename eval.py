"""
eval.py
-------
Evaluation script for EO-SAR binary change detection.

Usage:
    # Evaluate on test set
    python eval.py --data_path test --weights /path/to/best_model.pth

    # Evaluate with custom threshold and nodata masking
    python eval.py --data_path test \
                   --weights /path/to/best_model.pth \
                   --threshold 0.85 \
                   --mask_nodata

    # Evaluate on validation set
    python eval.py --data_path val --weights /path/to/best_model.pth
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from dataset import ChangeDetectionDataset
from model import build_model
from utils import (
    compute_metrics,
    print_metrics,
    plot_confusion_matrix,
    visualize_predictions,
    sweep_thresholds,
    get_valid_mask,
    apply_center_crop,
)


# ── Argument parser ───────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate change detection model"
    )
    parser.add_argument(
        "--data_path", type=str, default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to .pth checkpoint file"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Decision threshold for binary prediction"
    )
    parser.add_argument(
        "--mask_nodata", action="store_true",
        help="Exclude satellite swath edges from metrics"
    )
    parser.add_argument(
        "--nodata_thresh", type=int, default=10,
        help="SAR pixel value below which = nodata"
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Patch size used during training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory to save confusion matrix and visualisations"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep thresholds and print results (only on val/test)"
    )
    return parser.parse_args()


# ── Core evaluation ───────────────────────────────────────────────

def evaluate(model, dataset, loader, device,
             threshold, mask_nodata, nodata_thresh, image_size):
    """
    Run evaluation and return flat prediction and target arrays.
    Optionally masks nodata (swath edge) pixels.
    """
    model.eval()

    if mask_nodata:
        # Per-sample evaluation to apply per-sample nodata mask
        all_preds, all_targets = [], []

        with torch.no_grad():
            for idx in range(len(dataset)):
                image, mask = dataset[idx]
                pred = torch.sigmoid(
                    model(image.unsqueeze(0).to(device))
                ).cpu().squeeze().numpy()

                pred_bin = (pred > threshold).astype(np.float32)
                mask_np  = mask.squeeze().numpy()

                # Build valid-pixel mask from raw SAR
                raw_sar = np.array(dataset.ds[idx]["image"])
                valid   = get_valid_mask(raw_sar, nodata_thresh)
                valid   = apply_center_crop(valid, image_size)

                valid_flat = valid.flatten()
                all_preds.append(pred_bin.flatten()[valid_flat])
                all_targets.append(mask_np.flatten()[valid_flat])

        preds_bin = np.concatenate(all_preds)
        targets   = np.concatenate(all_targets)

    else:
        # Batch evaluation (faster, no nodata masking)
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, masks in loader:
                images  = images.to(device)
                outputs = torch.sigmoid(model(images))
                preds   = (outputs > threshold).float()
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())

        preds_bin = torch.cat(all_preds).numpy().flatten()
        targets   = torch.cat(all_targets).numpy().flatten()

    return preds_bin, targets


# ── Main ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print(f"\nLoading dataset (split={args.data_path})...")
    ds    = load_dataset("doron333/change-detection-dataset")
    split = "validation" if args.data_path == "val" else args.data_path

    dataset = ChangeDetectionDataset(ds[split], args.data_path,
                                     args.image_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=2)
    print(f"Samples: {len(dataset)}")

    # Load model
    print(f"\nLoading model from: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device)
    cfg  = ckpt.get("config", {})

    model = build_model(
        architecture = cfg.get("model", {}).get("architecture", "Unet"),
        backbone     = cfg.get("model", {}).get("backbone", "resnet34"),
        in_channels  = cfg.get("model", {}).get("in_channels", 4),
        num_classes  = cfg.get("model", {}).get("num_classes", 1),
        pretrained   = None,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Optional: threshold sweep
    if args.sweep:
        print("\nRunning threshold sweep on validation set...")
        all_probs, all_targets = [], []
        with torch.no_grad():
            for images, masks in loader:
                images  = images.to(device)
                outputs = torch.sigmoid(model(images))
                all_probs.append(outputs.cpu())
                all_targets.append(masks.cpu())
        probs   = torch.cat(all_probs).numpy().flatten()
        targets = torch.cat(all_targets).numpy().flatten()
        sweep_thresholds(probs, targets)
        return

    # Evaluate
    print(f"\nEvaluating on {args.data_path} set...")
    print(f"  Threshold   : {args.threshold}")
    print(f"  Nodata mask : {args.mask_nodata}")

    preds_bin, targets = evaluate(
        model, dataset, loader, device,
        args.threshold, args.mask_nodata,
        args.nodata_thresh, args.image_size
    )

    # Metrics
    metrics = compute_metrics(preds_bin, targets)
    print_metrics(metrics, args.data_path, args.threshold)

    # Confusion matrix
    cm_path = os.path.join(
        args.output_dir,
        f"confusion_matrix_{args.data_path}.png"
    )
    plot_confusion_matrix(
        targets, preds_bin,
        split_name=args.data_path,
        threshold=args.threshold,
        save_path=cm_path
    )

    # Visualise predictions
    vis_path = os.path.join(
        args.output_dir,
        f"predictions_{args.data_path}.png"
    )
    visualize_predictions(
        model, dataset, device,
        split_name=args.data_path,
        threshold=args.threshold,
        n_samples=8,
        save_path=vis_path
    )

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
