# Binary Change Detection on EO-SAR Image Pairs
### GalaxEye Space — Satellite AI Research Intern Assignment

A pixel-level binary change detection model that identifies damaged and destroyed structures from paired pre-event Electro-Optical (EO) and post-event Synthetic Aperture Radar (SAR) satellite imagery. Built using an Early Fusion UNet with a pretrained ResNet34 backbone.

---

## Table of Contents
1. [Project Description](#project-description)
2. [Requirements](#requirements)
3. [Environment Setup](#environment-setup)
4. [Dataset Structure](#dataset-structure)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Model Weights](#model-weights)
8. [Results](#results)
9. [Citation / References](#citation--references)

---

## Project Description

**Task:** Given a co-registered pre-event EO (RGB) and post-event SAR (grayscale) image pair, predict a binary pixel-level change mask where:
- `1 = Change` (damaged or destroyed)
- `0 = No-Change` (background or intact)

**Approach:**
- Early fusion: EO (3ch) + SAR (1ch) concatenated → 4-channel input
- UNet decoder with ResNet34 ImageNet-pretrained encoder
- Combined Dice + BCE loss with positive class weighting to handle severe class imbalance (~7% change pixels)
- Threshold calibration on validation set (optimal threshold = 0.85)
- Nodata masking to exclude satellite swath edges from metric computation

---

## Requirements

**Python version:** 3.10+

All dependencies with pinned versions:

```
torch==2.1.0
torchvision==0.16.0
segmentation-models-pytorch==0.3.3
albumentations==1.3.1
datasets==2.16.1
numpy==1.24.4
Pillow==10.0.0
scikit-learn==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
tqdm==4.66.1
PyYAML==6.0.1
huggingface-hub==0.19.4
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Environment Setup

### Option A — Google Colab (Recommended, no GPU required locally)

```bash
# Run in a Colab cell
!pip install segmentation-models-pytorch albumentations datasets -q
```

Mount Google Drive to save checkpoints:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Option B — Local with conda

```bash
# Create environment
conda create -n galaxeye python=3.10 -y
conda activate galaxeye

# Install PyTorch (adjust cuda version for your GPU)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### Option C — Local with venv

```bash
python -m venv galaxeye_env
source galaxeye_env/bin/activate        # Linux/Mac
# galaxeye_env\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Structure

The dataset is loaded directly from HuggingFace — no manual download needed.

```python
from datasets import load_dataset
ds = load_dataset("doron333/change-detection-dataset")
```

**Internal triplet structure** (automatically handled by the dataloader):

```
For each split of N triplets, HuggingFace stores 3N samples:
  index i          → POST-event SAR image (grayscale, 1024×1024)
  index i + N      → PRE-event  EO  image (RGB, 1024×1024)
  index i + 2N     → Binary change mask   (0=No-Change, 1=Change)

Train : 2,781 triplets
Val   :   334 triplets
Test  :    77 triplets
```

**If you have the data as local files**, place them as follows:

```
data/
├── train/
│   ├── pre/          ← EO RGB images (.tif)
│   ├── post/         ← SAR grayscale images (.tif)
│   └── masks/        ← binary masks (.tif)
├── val/
│   ├── pre/
│   ├── post/
│   └── masks/
└── test/
    ├── pre/
    ├── post/
    └── masks/
```

---

## Training

### Step 1 — Edit config

All hyperparameters are in `config.yaml`. Edit as needed:

```bash
nano config.yaml   # or open in any text editor
```

### Step 2 — Run training from scratch

```bash
python train.py --config config.yaml
```

**Training will:**
- Automatically download the dataset from HuggingFace on first run
- Save the best checkpoint (by validation F1) to `save_dir` specified in config
- Print train and val metrics after every epoch
- Take ~3 minutes/epoch on a T4 GPU (30 epochs ≈ 90 minutes)

### Step 3 — Resume from checkpoint (if interrupted)

```bash
python train.py --config config.yaml --resume /path/to/best_model.pth
```

### Key config options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 30 | Total training epochs |
| `batch_size` | 8 | Batch size per step |
| `lr` | 1e-4 | Initial learning rate |
| `pos_weight` | 10.0 | BCE positive class weight |
| `image_size` | 256 | Patch size (cropped from 1024×1024) |
| `backbone` | resnet34 | Encoder backbone |
| `save_dir` | ./checkpoints | Where to save best model |

---

## Evaluation

### Evaluate on test set

```bash
python eval.py --data_path test --weights /path/to/best_model.pth
```

### Evaluate on any split

```bash
# Validation set
python eval.py --data_path val --weights /path/to/best_model.pth

# Test set with custom threshold
python eval.py --data_path test --weights /path/to/best_model.pth --threshold 0.85

# With nodata masking (recommended)
python eval.py --data_path test --weights /path/to/best_model.pth \
               --threshold 0.85 --mask_nodata
```

**Evaluation outputs:**
- IoU, Precision, Recall, F1 printed to console
- Confusion matrix saved as PNG
- Prediction visualisations saved as PNG (8 samples)

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `test` | Split to evaluate: `train`, `val`, `test` |
| `--weights` | required | Path to `.pth` checkpoint file |
| `--threshold` | `0.85` | Decision threshold for binary prediction |
| `--mask_nodata` | `False` | Exclude satellite swath edges from metrics |
| `--output_dir` | `./outputs` | Where to save visualisations |

---

## Model Weights

The final trained checkpoint is publicly available for download:

**Download link:** `[ADD YOUR GOOGLE DRIVE OR HUGGINGFACE LINK HERE]`

```bash
# Download with gdown (Google Drive)
pip install gdown
gdown "YOUR_GDRIVE_FILE_ID" -O best_model.pth

# Or download with wget (HuggingFace)
wget "YOUR_HF_URL" -O best_model.pth
```

**Checkpoint contents:**
```python
{
    'epoch'       : 25,
    'model_state' : ...,   # UNet ResNet34 weights
    'optimizer'   : ...,   # AdamW state
    'val_f1'      : 0.7893,
    'val_iou'     : 0.6520,
    'config'      : {...}  # full config used for training
}
```

---

## Results

All metrics computed for the **change class (label=1)** only.

### With threshold=0.85 and nodata masking

| Metric | Validation | Test |
|--------|-----------|------|
| **IoU** | **0.7252** | **0.3897** |
| **Precision** | **0.8096** | **0.4466** |
| **Recall** | **0.8742** | **0.7533** |
| **F1 Score** | **0.8406** | **0.5608** |

### Threshold sensitivity (Validation set)

| Threshold | Val F1 | Val IoU | Val Precision | Val Recall |
|-----------|--------|---------|---------------|------------|
| 0.50 | 0.7893 | 0.6520 | 0.6674 | 0.9657 |
| 0.70 | 0.8264 | 0.7042 | 0.7408 | 0.9345 |
| **0.85** | **0.8406** | **0.7252** | **0.8096** | **0.8742** |
| 0.90 | 0.8373 | 0.7202 | 0.8389 | 0.8358 |

### Training curve summary

| Epoch | Val F1 | Val IoU | Note |
|-------|--------|---------|------|
| 1 | 0.466 | 0.304 | Start |
| 7 | 0.749 | 0.599 | First major jump |
| 18 | 0.768 | 0.624 | Resume after interrupt |
| 25 | 0.789 | 0.652 | **Best checkpoint saved** |
| 30 | 0.770 | 0.625 | End of training |

### Model details

```
Architecture : UNet
Backbone     : ResNet34 (ImageNet pretrained)
Input        : 4-channel (EO RGB 3ch + SAR grayscale 1ch)
Output       : 1-channel binary mask (sigmoid)
Parameters   : ~24.4M total, ~24.4M trainable
Loss         : 0.5 × Dice + 0.5 × BCE (pos_weight=10)
Optimizer    : AdamW (lr=1e-4, weight_decay=1e-4)
Scheduler    : CosineAnnealing (T_max=30)
Epochs       : 30
Batch size   : 8
Image size   : 256×256 patches
Seed         : 42
```

---

## Citation / References

```bibtex
@inproceedings{daudt2018,
  title     = {Fully Convolutional Siamese Networks for Change Detection},
  author    = {Daudt, Rodrigo Caye and Le Saux, Bertr and Boulch, Alexandre},
  booktitle = {ICIP},
  year      = {2018}
}

@article{chen2021bit,
  title   = {Remote Sensing Image Change Detection with Transformers},
  author  = {Chen, Hao and Qi, Zipeng and Shi, Zhenwei},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2021}
}

@inproceedings{bandara2022changeformer,
  title     = {A Transformer-Based Siamese Network for Change Detection},
  author    = {Bandara, Wele Gedara Chaminda and Patel, Vishal M.},
  booktitle = {IGARSS},
  year      = {2022}
}

@article{fang2021snunet,
  title   = {SNUNet-CD: A Densely Connected Siamese Network for 
             Change Detection of VHR Images},
  author  = {Fang, Sheng and Li, Kaiyu and Shao, Jinyuan and Li, Zhe},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2021}
}

@inproceedings{lin2017focal,
  title     = {Focal Loss for Dense Object Detection},
  author    = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross 
               and He, Kaiming and Dollár, Piotr},
  booktitle = {ICCV},
  year      = {2017}
}

@inproceedings{he2016resnet,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {CVPR},
  year      = {2016}
}
```

**Libraries:**
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [albumentations](https://github.com/albumentations-team/albumentations)
- [HuggingFace datasets](https://huggingface.co/docs/datasets)

---

## Repository Structure

```
galaxeye_change_detection/
│
├── config.yaml          ← all hyperparameters
├── dataset.py           ← HuggingFace dataloader + augmentations
├── model.py             ← UNet model definition
├── loss.py              ← Combined Dice + BCE loss
├── train.py             ← training script
├── eval.py              ← evaluation + visualisation script
├── utils.py             ← metrics, nodata masking helpers
├── requirements.txt     ← pinned dependencies
└── README.md            ← this file
```

---

*Submitted for GalaxEye Space — Satellite AI Research Intern Technical Assessment*
*Deadline: 14 May 2026*
