"""
utils.py
--------
Metric computation, nodata masking, and visualisation helpers.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ── Metrics ──────────────────────────────────────────────────────

def compute_metrics(preds_bin: np.ndarray,
                    targets: np.ndarray) -> dict:
    """
    Compute IoU, Precision, Recall and F1 for the change class (label=1).

    Parameters
    ----------
    preds_bin : binary numpy array (0/1), shape (N,)
    targets   : binary numpy array (0/1), shape (N,)

    Returns
    -------
    dict with keys: iou, precision, recall, f1, TP, FP, FN, TN
    """
    TP = int(((preds_bin == 1) & (targets == 1)).sum())
    FP = int(((preds_bin == 1) & (targets == 0)).sum())
    FN = int(((preds_bin == 0) & (targets == 1)).sum())
    TN = int(((preds_bin == 0) & (targets == 0)).sum())

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = TP / (TP + FP + FN + 1e-8)

    return {
        "iou"      : float(iou),
        "precision": float(precision),
        "recall"   : float(recall),
        "f1"       : float(f1),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    }


def print_metrics(metrics: dict, split_name: str,
                  threshold: float) -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n{'='*55}")
    print(f"  RESULTS ON {split_name.upper()}  (threshold={threshold:.2f})")
    print(f"{'='*55}")
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"{'='*55}")
    print(f"  TP: {metrics['TP']:,}  FP: {metrics['FP']:,}")
    print(f"  FN: {metrics['FN']:,}  TN: {metrics['TN']:,}")


# ── Nodata masking ───────────────────────────────────────────────

def get_valid_mask(raw_sar: np.ndarray,
                   nodata_thresh: int = 10) -> np.ndarray:
    """
    Build a boolean valid-pixel mask by thresholding the SAR image.
    Pixels where SAR intensity < nodata_thresh are satellite swath
    edges (nodata) and should be excluded from metric computation.

    Parameters
    ----------
    raw_sar       : 2D uint8 SAR numpy array (H, W)
    nodata_thresh : Pixel value below which = nodata

    Returns
    -------
    Boolean numpy array, True = valid pixel
    """
    return raw_sar > nodata_thresh


def apply_center_crop(mask: np.ndarray,
                      crop_size: int) -> np.ndarray:
    """Apply a centre crop to a 2D mask (same as val/test transform)."""
    h, w = mask.shape
    top  = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return mask[top:top + crop_size, left:left + crop_size]


# ── Confusion matrix ─────────────────────────────────────────────

def plot_confusion_matrix(targets: np.ndarray,
                          preds_bin: np.ndarray,
                          split_name: str,
                          threshold: float,
                          save_path: str = None) -> None:
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(targets.astype(int), preds_bin.astype(int))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["Pred No-Change", "Pred Change"],
        yticklabels=["True No-Change", "True Change"],
        ax=ax
    )
    ax.set_title(
        f"Confusion Matrix — {split_name} (thr={threshold:.2f})",
        fontsize=12
    )
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Prediction")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix → {save_path}")
    plt.show()


# ── Visualisation ────────────────────────────────────────────────

def visualize_predictions(model, dataset, device,
                           split_name: str,
                           threshold: float = 0.85,
                           n_samples: int = 8,
                           save_path: str = None) -> None:
    """
    Visualise n_samples predictions side-by-side:
    [EO pre-event | SAR post-event | Ground Truth | Prediction]
    """
    model.eval()
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))[:n_samples]

    fig, axes = plt.subplots(n_samples, 4,
                             figsize=(20, n_samples * 5))
    for ax, title in zip(
        axes[0],
        ["EO Pre-event", "SAR Post-event",
         "Ground Truth", f"Prediction (thr={threshold:.2f})"]
    ):
        ax.set_title(title, fontsize=13, fontweight="bold")

    for row, idx in enumerate(indices):
        image, mask = dataset[int(idx)]

        with torch.no_grad():
            pred = torch.sigmoid(
                model(image.unsqueeze(0).to(device))
            ).cpu().squeeze().numpy()

        pred_bin = (pred > threshold).astype(np.uint8)
        mask_np  = mask.squeeze().numpy()

        # Per-sample F1
        tp = int(((pred_bin == 1) & (mask_np == 1)).sum())
        fp = int(((pred_bin == 1) & (mask_np == 0)).sum())
        fn = int(((pred_bin == 0) & (mask_np == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

        # Raw images for display
        pre_eo, post_sar, _ = dataset.get_raw_triplet(int(idx))
        pre_disp = pre_eo.astype(float)
        pre_disp = (pre_disp - pre_disp.min()) / \
                   (pre_disp.max() - pre_disp.min() + 1e-8)

        axes[row][0].imshow(pre_disp)
        axes[row][0].axis("off")
        axes[row][1].imshow(post_sar, cmap="gray")
        axes[row][1].axis("off")
        axes[row][2].imshow(mask_np, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[row][2].axis("off")
        axes[row][3].imshow(pred_bin, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[row][3].set_title(
            f"F1={f1:.3f}",
            color="green" if f1 > 0.6 else "orange" if f1 > 0.3 else "red",
            fontsize=11
        )
        axes[row][3].axis("off")

    plt.suptitle(
        f"Predictions — {split_name} | Red=Change | Green=No-Change",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved visualisations → {save_path}")
    plt.show()


# ── Threshold sweep ──────────────────────────────────────────────

def sweep_thresholds(preds_raw: np.ndarray,
                     targets: np.ndarray,
                     thresholds: np.ndarray = None) -> list:
    """
    Sweep decision thresholds and return metrics at each.

    Parameters
    ----------
    preds_raw  : raw sigmoid probabilities, shape (N,)
    targets    : binary ground truth, shape (N,)
    thresholds : array of threshold values to try

    Returns
    -------
    list of dicts, each with threshold + metrics
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.95, 0.05)

    results = []
    print(f"\n{'Threshold':>10} {'F1':>8} {'IoU':>8} "
          f"{'Prec':>8} {'Rec':>8}")
    print("-" * 50)

    for thresh in thresholds:
        preds_bin = (preds_raw > thresh).astype(np.float32)
        m = compute_metrics(preds_bin, targets)
        m["threshold"] = float(thresh)
        results.append(m)
        print(f"{thresh:>10.2f} {m['f1']:>8.4f} {m['iou']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f}")

    best = max(results, key=lambda x: x["f1"])
    print(f"\n Best threshold: {best['threshold']:.2f}  "
          f"F1={best['f1']:.4f}  IoU={best['iou']:.4f}")
    return results
