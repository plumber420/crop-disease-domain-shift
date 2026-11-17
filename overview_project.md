# Project Status & Technical Audit (`crop-disease-domain-shift`)

> **Last Audited:** 2026-11-17  
> **Project Title:** *Overcoming Domain Shift in Deep Learning-Based Crop Disease Classification Under Field Conditions*  
> **Institution / Cluster:** KU Leuven — Vlaams Supercomputer Centrum (VSC) Tier-2 `wICE` GPU Cluster  
> **Git HEAD:** `b1bee33e` (`results resnett and vision transform first try`, committed `2026-11-05` from `vsc39394@tier2-p-login-4.genius.hpc.kuleuven.be`)

---

## 1. Executive Summary & Current Project Stage

### Research Goal
This project investigates the **Lab-to-Field Generalization Gap** and **Shortcut Learning** in crop disease classification when vision models trained exclusively on curated laboratory imagery (**PlantVillage**: isolated leaves on uniform backgrounds under controlled studio lighting) are evaluated zero-shot on complex outdoor field imagery (**PlantDoc**: variable lighting, soil/mud, overlapping leaves, and cluttered backgrounds). It compares:
- **Architectures:** CNN baseline (**ResNet-50**, `torchvision.models.resnet50`) vs. hierarchical Vision Transformer (**Swin Transformer Tiny / Swin-T**, `torchvision.models.swin_t`), both ImageNet-1K pretrained.
- **Domain-Shift Mitigation:** Baseline augmentations vs. **CutMix + MixUp** batch-level regularization (`SoftTargetCrossEntropy`).
- **Explainability:** Grad-CAM spatial attribution (`layer4[-1]` for ResNet-50; `features[-1]` with dynamic token reshaping for Swin-T) with quantitative shortcut-learning diagnostics (Center Focus Ratio, Border Ratio, Top-10% Focal Concentration, Spatial Entropy).

### Exact Pipeline & Artifact Status
Based on Git history (`b1bee33e`) and the files present on disk:

| Pipeline Stage | SLURM Script / Entrypoint | Execution Status | Artifacts Present on Local Disk | Missing / Remote-Only Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **1. Environment Setup** | `slurm/01_setup_env.sh` | Executed on VSC (`requirements.txt` updated to comment out `pytorch-grad-cam`) | Script present (`slurm/01_setup_env.sh`) | Venv `/data/leuven/393/vsc39394/venvs/dl_plants_env/` resides on VSC |
| **2. Class Audit & Manifest** | `data/dataset_mapping.py`, `src/dataset.py` | Executed on VSC | `data/canonical_classes.json`, `data/dataset_splits.json` | — |
| **3. ResNet-50 Training** | `slurm/02_train_resnet.slurm` | Executed on VSC (produced `best_checkpoint.pt` referenced in eval JSONs) | None (excluded by `.gitignore`) | `experiments/outputs/resnet50_{baseline,cutmix_mixup}/` (`*.pt`, `config.json`, `metrics_history.csv`) & `slurm/logs/` are only on VSC |
| **4. Swin-T Training** | `slurm/03_train_swin.slurm` | Executed on VSC (produced `best_checkpoint.pt` referenced in eval JSONs) | None (excluded by `.gitignore`) | `experiments/outputs/swin_t_{baseline,cutmix_mixup}/` (`*.pt`, `config.json`, `metrics_history.csv`) & `slurm/logs/` are only on VSC |
| **5. Cross-Domain Evaluation** | `slurm/04_evaluate_all.slurm` | Executed on VSC for all 4 experiments | 4 JSON reports (`domain_shift_report_<exp>.json`) + 12 PNG plots in `experiments/evaluation_results/` | `slurm/logs/eval_domain_shift_*.{out,err}` are only on VSC |
| **6. Grad-CAM Explainability** | `slurm/05_run_gradcam.slurm` | Executed on VSC for all 4 experiments (via `NativeGradCAM`) | 96 PNG visualizations (4 experiments × 2 domains × 12 samples) in `experiments/gradcam_analysis/` | `slurm/logs/gradcam_explain_*.{out,err}` are only on VSC |

**Important Local Workspace Notes:**
1. **`experiments/outputs/` and `slurm/logs/` are NOT on local disk:** `.gitignore` ignores `experiments/outputs/`, `*.pt`, `slurm/logs/`, `*.out`, `*.err`, and `*.log`. Consequently, model checkpoints, `config.json`, `metrics_history.csv`, and SLURM `.out`/`.err` logs remain on VSC (`/vsc-hard-mounts/leuven-data/393/vsc39394/crop-disease-domain-shift/`) and cannot be inspected locally until synced back.
2. **Local rename in `experiments/evaluation_results/`:** Commit `b1bee33e` tracked `domain_shift_report.json` inside each experiment folder, whereas on local disk those 4 files were renamed to `domain_shift_report_<experiment_name>.json` (showing in `git status -uall` as 4 deleted tracked JSONs and 4 untracked JSONs).
3. **`RUNBOOK.md` contains illustrative placeholders:** The benchmark table in `RUNBOOK.md` (e.g., 32.4% → 45.8% for ResNet-50, 38.6% → 51.2% for Swin-T, `$N \approx 8.000$`) reflects target expectations rather than actual observed numbers (`test_pv = 2,897`; actual 13-class shared PD Macro F1 = 14.38% → 16.30% for ResNet-50 and 15.33% → 18.07% for Swin-T).
4. **Major experimental bottleneck discovered:** Only a **15-folder subset of PlantVillage** (Bell Pepper, Potato, Tomato; 14 mappable classes) was present during training, while **PlantDoc has 28 classes**. Thus **1,515 / 2,922 PlantDoc samples (51.85%) belong to 15 unseen crops/diseases**, causing severe open-set collapse into attractor classes (`tomato_late_blight` and `bell_pepper_healthy`).

---

## 2. Repository & Artifact Inventory

### Directory Tree (Excluding Raw Dataset Image Files)

```text
crop-disease-domain-shift/
├── .gitignore
├── RUNBOOK.md
├── requirements.txt
├── data/
│   ├── canonical_classes.json
│   ├── dataset_mapping.py
│   ├── dataset_splits.json
│   ├── PlantDoc/                           (local junction -> ../../plantdoc; gitignored)
│   └── PlantVillage/                       (local junction -> ../../plantVillage/PlantVillage; gitignored)
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── models.py
│   └── train.py
├── slurm/
│   ├── 01_setup_env.sh
│   ├── 02_train_resnet.slurm
│   ├── 03_train_swin.slurm
│   ├── 04_evaluate_all.slurm
│   └── 05_run_gradcam.slurm
│   └── [logs/ NOT PRESENT ON LOCAL DISK - gitignored on VSC]
└── experiments/
    ├── [outputs/ NOT PRESENT ON LOCAL DISK - gitignored on VSC]
    ├── evaluation_results/
    │   ├── resnet50_baseline/
    │   │   ├── confusion_matrix_source_lab.png
    │   │   ├── confusion_matrix_target_field.png
    │   │   ├── domain_shift_report_resnet50_baseline.json
    │   │   └── per_class_domain_gap.png
    │   ├── resnet50_cutmix_mixup/
    │   │   ├── confusion_matrix_source_lab.png
    │   │   ├── confusion_matrix_target_field.png
    │   │   ├── domain_shift_report_resnet50_cutmix_mixup.json
    │   │   └── per_class_domain_gap.png
    │   ├── swin_t_baseline/
    │   │   ├── confusion_matrix_source_lab.png
    │   │   ├── confusion_matrix_target_field.png
    │   │   ├── domain_shift_report_swin_t_baseline.json
    │   │   └── per_class_domain_gap.png
    │   └── swin_t_cutmix_mixup/
    │       ├── confusion_matrix_source_lab.png
    │       ├── confusion_matrix_target_field.png
    │       ├── domain_shift_report_swin_t_cutmix_mixup.json
    │       └── per_class_domain_gap.png
    └── gradcam_analysis/
        ├── resnet50_baseline/
        │   ├── resnet50_plantdoc_field/    (12 PNG files)
        │   └── resnet50_plantvillage_lab/  (12 PNG files)
        ├── resnet50_cutmix_mixup/
        │   ├── resnet50_plantdoc_field/    (12 PNG files)
        │   └── resnet50_plantvillage_lab/  (12 PNG files)
        ├── swin_t_baseline/
        │   ├── swin_t_plantdoc_field/      (12 PNG files)
        │   └── swin_t_plantvillage_lab/    (12 PNG files)
        └── swin_t_cutmix_mixup/
            ├── swin_t_plantdoc_field/      (12 PNG files)
            └── swin_t_plantvillage_lab/    (12 PNG files)
```

### File-by-File Inventory & Status

| Path | Role | Current Status |
| :--- | :--- | :--- |
| `.gitignore` | Ignores raw datasets, `experiments/outputs/`, `*.pt`, and `slurm/logs/`, while unignoring `evaluation_results/` and `gradcam_analysis/`. | Committed in `b1bee33e` (working as intended, though it blocks syncing `config.json`/`metrics_history.csv` inside `outputs/`). |
| `requirements.txt` | Python dependencies (`torch`, `torchvision`, `timm`, `numpy`, `scipy`, `scikit-learn`, `pillow`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `rich`, `tensorboard`). | `pytorch-grad-cam>=0.2.1` is commented out (`b1bee33e`). |
| `RUNBOOK.md` | Dutch step-by-step VSC cluster execution & scientific guide. | Up to date on cluster workflow, but benchmark table & split counts (~8,000) are illustrative targets. |
| `data/dataset_mapping.py` | Defines 29 canonical classes (`CANONICAL_MAPPING`), regex normalization, nested split scanning, and alignment audit CLI. | Functional; tested locally (detects 13 matched classes, 19,082 matched PV / 1,407 matched PD). |
| `data/canonical_classes.json` | Exported JSON specification of all 29 canonical classes (`id` 0–28, description, `pv_candidates`, `pd_candidates`). | Generated on VSC and committed in `b1bee33e`. |
| `data/dataset_splits.json` | Persistent deterministic 70/15/15 stratified manifest of PlantVillage relative paths (`train`, `val`, `test_pv`, seed 42). | Generated on VSC (`13,458` / `2,879` / `2,897` = `19,234` total across 14 classes) and committed in `b1bee33e`. |
| `src/__init__.py` | Package initialization (`__version__ = "1.0.0"`). | Complete. |
| `src/dataset.py` | `MappedCropDataset`, persistent split generator, transforms, `CutMixMixupCollator`, `SoftTargetCrossEntropy`, `build_dataloaders`. | Functional; inspects nested `train`/`test`/`val` folders and expands `$VSC_DATA`. |
| `src/models.py` | ResNet-50 and Swin-T builders with ImageNet-1K pretrained weights, custom dropout+linear head, and differential LR parameter groups. | Functional; minor dead variable `is_head_param = False` at line 140. |
| `src/train.py` | AMP training loop with gradient clipping, differential AdamW LR, cosine schedule with linear warmup, and CSV/checkpoint saving. | Functional; executed on VSC for all 4 experiments. |
| `src/evaluate.py` | Evaluates source (`test_pv`) and target (`PlantDoc`) splits, computes metrics, plots confusion matrices & per-class gap bars, exports JSON. | Executed on VSC; `macro_f1` omits `labels=` causing 28-vs-29 denominator inconsistency. |
| `src/explainability.py` | `NativeGradCAM` + `PyTorchGradCAMEngine` wrapper, `swin_reshape_transform`, shortcut metrics, and 3-panel PNG generator. | Executed on VSC (`NativeGradCAM`); samples indiscriminately across train/val/test and unseen classes. |
| `slurm/01_setup_env.sh` | Creates `$VSC_DATA/venvs/dl_plants_env` with CUDA 12.1 PyTorch and project requirements (Unix LF). | Ready / previously used on VSC. |
| `slurm/02_train_resnet.slurm` | Runs `resnet50_baseline` and `resnet50_cutmix_mixup` sequentially on 1× wICE GPU (4h walltime, Unix LF). | Executed on VSC; logs/checkpoints remote-only. |
| `slurm/03_train_swin.slurm` | Runs `swin_t_baseline` and `swin_t_cutmix_mixup` sequentially on 1× wICE GPU (4.5h walltime, Unix LF). | Executed on VSC; logs/checkpoints remote-only. |
| `slurm/04_evaluate_all.slurm` | Evaluates all 4 `best_checkpoint.pt` models and prints comparative leaderboard (1.5h walltime, Unix LF). | Executed on VSC. |
| `slurm/05_run_gradcam.slurm` | Runs Grad-CAM (`--num_samples 12` per domain) on all 4 `best_checkpoint.pt` models (1h walltime, Unix LF). | Executed on VSC. |
| `experiments/evaluation_results/<exp>/domain_shift_report_<exp>.json` | Full aggregate & 29-class evaluation JSONs for the 4 runs (`resnet50_baseline`, `resnet50_cutmix_mixup`, `swin_t_baseline`, `swin_t_cutmix_mixup`). | Present on disk; renamed locally from tracked `domain_shift_report.json`. |
| `experiments/evaluation_results/<exp>/*.png` | 3 plots per run (12 total): `confusion_matrix_source_lab.png`, `confusion_matrix_target_field.png`, `per_class_domain_gap.png`. | Present on disk and tracked in Git. |
| `experiments/gradcam_analysis/<exp>/<model>_{plantvillage_lab,plantdoc_field}/*.png` | 24 three-panel Grad-CAM figures per experiment (12 PlantVillage + 12 PlantDoc = **96 PNGs total**). | Present on disk and tracked in Git. |

---

## 3. Dataset State & Class Mapping Audit

### High-Level Dataset Inventory & Split Counts
Inspected directly from `data/dataset_splits.json`, `data/canonical_classes.json`, `data/dataset_mapping.py`, and the local dataset directories (`data/PlantVillage` and `data/PlantDoc`):

- **Source Dataset (`PlantVillage`):**
  - **15 folders on disk** (`20,638` files total — corresponds to the Kaggle `emmarex/plantdisease` Solanaceae subset: Bell Pepper, Potato, Tomato):
    - **14 mapped classes** included in `data/dataset_splits.json`: **19,234 images**
      - `train`: **13,458** images (70%)
      - `val`: **2,879** images (15%)
      - `test_pv`: **2,897** images (15%)
    - **1 unmapped folder skipped:** `Tomato__Target_Spot` (**1,404** images) is not in `CANONICAL_MAPPING` (PlantDoc has no Target Spot class).
    - **1 non-image file skipped:** `Tomato__Tomato_YellowLeaf__Curl_Virus` contains `3,209` files on disk (`3,208` `.jpg` images + `1` extensionless non-image file).
- **Target Dataset (`PlantDoc`):**
  - Nested under `train/` (**28** class folders, **2,670** images) and `test/` (**27** class folders, **252** images — missing `Tomato_two_spotted_spider_mites_leaf`).
  - Both `verify_dataset_alignment` and `MappedCropDataset` merge `train/` + `test/` into **2,922 images** across **28 canonical classes** (`potato_healthy` has **0** images in PlantDoc).
- **Cross-Domain Overlap Breakdown:**
  - **13 `MATCH` classes** (`support > 0` in both PV and PD): **19,082** PV images (`13,352` train / `2,857` val / `2,873` test_pv) and **1,407** PD images (`1,284` train / `123` test).
  - **1 `PV ONLY` class** (`potato_healthy`): **152** PV images (`106` train / `22` val / `24` test_pv) and **0** PD images.
  - **15 `PD ONLY` classes** (`support == 0` in PV): **0** PV images and **1,515** PD images (`1,386` train / `129` test) — representing **51.85%** of all evaluated PlantDoc samples (`1,515 / 2,922`).

### Complete 29-Class Canonical Support Matrix

| ID | Canonical Class | PV Train | PV Val | PV Test | PV Total | PD Train | PD Test | PD Total | Alignment Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | `apple_healthy` | 0 | 0 | 0 | 0 | 79 | 9 | 88 | `PD ONLY` (unseen) |
| 1 | `apple_rust` | 0 | 0 | 0 | 0 | 96 | 10 | 106 | `PD ONLY` (unseen) |
| 2 | `apple_scab` | 0 | 0 | 0 | 0 | 83 | 10 | 93 | `PD ONLY` (unseen) |
| 3 | `bell_pepper_bacterial_spot` | 697 | 149 | 151 | 997 | 74 | 9 | 83 | **`MATCH`** |
| 4 | `bell_pepper_healthy` | 1,034 | 221 | 223 | 1,478 | 34 | 8 | 42 | **`MATCH`** |
| 5 | `blueberry_healthy` | 0 | 0 | 0 | 0 | 106 | 11 | 117 | `PD ONLY` (unseen) |
| 6 | `cherry_healthy` | 0 | 0 | 0 | 0 | 47 | 10 | 57 | `PD ONLY` (unseen) |
| 7 | `corn_gray_leaf_spot` | 0 | 0 | 0 | 0 | 63 | 4 | 67 | `PD ONLY` (unseen) |
| 8 | `corn_leaf_blight` | 0 | 0 | 0 | 0 | 182 | 12 | 194 | `PD ONLY` (unseen) |
| 9 | `corn_rust` | 0 | 0 | 0 | 0 | 107 | 10 | 117 | `PD ONLY` (unseen) |
| 10 | `grape_black_rot` | 0 | 0 | 0 | 0 | 71 | 8 | 79 | `PD ONLY` (unseen) |
| 11 | `grape_healthy` | 0 | 0 | 0 | 0 | 63 | 12 | 75 | `PD ONLY` (unseen) |
| 12 | `peach_healthy` | 0 | 0 | 0 | 0 | 103 | 9 | 112 | `PD ONLY` (unseen) |
| 13 | `potato_early_blight` | 700 | 150 | 150 | 1,000 | 157 | 14 | 171 | **`MATCH`** |
| 14 | `potato_healthy` | 106 | 22 | 24 | 152 | 0 | 0 | 0 | `PV ONLY` |
| 15 | `potato_late_blight` | 700 | 150 | 150 | 1,000 | 200 | 8 | 208 | **`MATCH`** |
| 16 | `raspberry_healthy` | 0 | 0 | 0 | 0 | 112 | 7 | 119 | `PD ONLY` (unseen) |
| 17 | `soybean_healthy` | 0 | 0 | 0 | 0 | 57 | 8 | 65 | `PD ONLY` (unseen) |
| 18 | `squash_powdery_mildew` | 0 | 0 | 0 | 0 | 124 | 6 | 130 | `PD ONLY` (unseen) |
| 19 | `strawberry_healthy` | 0 | 0 | 0 | 0 | 88 | 8 | 96 | `PD ONLY` (unseen) |
| 20 | `tomato_bacterial_spot` | 1,488 | 319 | 320 | 2,127 | 101 | 9 | 110 | **`MATCH`** |
| 21 | `tomato_early_blight` | 700 | 150 | 150 | 1,000 | 79 | 9 | 88 | **`MATCH`** |
| 22 | `tomato_healthy` | 1,113 | 238 | 240 | 1,591 | 44 | 8 | 52 | **`MATCH`** |
| 23 | `tomato_late_blight` | 1,336 | 286 | 287 | 1,909 | 101 | 10 | 111 | **`MATCH`** |
| 24 | `tomato_leaf_mold` | 666 | 142 | 144 | 952 | 85 | 6 | 91 | **`MATCH`** |
| 25 | `tomato_mosaic_virus` | 261 | 55 | 57 | 373 | 44 | 10 | 54 | **`MATCH`** |
| 26 | `tomato_septoria_leaf_spot` | 1,239 | 265 | 267 | 1,771 | 145 | 12 | 157 | **`MATCH`** |
| 27 | `tomato_spider_mites` | 1,173 | 251 | 252 | 1,676 | 2 | 0 | 2 | **`MATCH`** |
| 28 | `tomato_yellow_leaf_curl_virus` | 2,245 | 481 | 482 | 3,208 | 223 | 15 | 238 | **`MATCH`** |
| **Sum** | **29 Canonical Classes** | **13,458** | **2,879** | **2,897** | **19,234** | **2,670** | **252** | **2,922** | **13 MATCH / 1 PV / 15 PD** |

### How `MappedCropDataset` Handles Missing / Unseen Classes
1. **Static 29-class output space:** `build_dataloaders()` and `build_model()` always fix `num_classes = len(get_canonical_class_list()) = 29` regardless of how many classes are actually present on disk.
2. **Optional filtering left unused:** `MappedCropDataset.__init__` accepts `allowed_classes: Optional[List[str]]`, but neither `train.py`, `evaluate.py`, nor `build_dataloaders()` passes `active_common_classes` to it. Every folder matching any of the 29 canonical keys is loaded.
3. **Unseen PlantDoc classes during evaluation:** Because `PlantDoc` contains 15 `PD ONLY` folders that match `CANONICAL_MAPPING`, `MappedCropDataset(pd_dir, domain="pd")` loads all 1,515 images with ground-truth IDs `{0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19}`. Meanwhile, the model's 15 corresponding output logits never received positive supervision during training (only negative softmax push), guaranteeing 0% recall on those 15 classes.
4. **Unmapped folders & corrupt images:** Folders absent from `CANONICAL_MAPPING` (such as `Tomato__Target_Spot`) are silently dropped (`resolve_folder_to_canonical` returns `None`). Corrupt image files in `__getitem__` fall back to a black `224x224` RGB image rather than crashing.

---

## 4. Codebase Architecture & Exact Training Configurations

### Module Architecture Summary

- **`data/dataset_mapping.py` (633 lines):**
  - Holds `CANONICAL_MAPPING` (29 crop-disease pairs) sorted alphabetically to assign class IDs `0..28`.
  - `normalize_string()` lowercases, collapses `[_\s\-]+` to `_`, removes `[(),]`, and strips trailing `_leaf` / `_leaves`.
  - `_get_target_folders()` and `verify_dataset_alignment()` flatten direct split folders (`train`, `test`, `val`, `validation`) one level down so nested PlantDoc splits work transparently.
  - `get_default_dataset_paths()` resolves `$VSC_DATA/datasets/...`, `/data/leuven/393/vsc39394/datasets/...`, then local `data/...`.
- **`src/dataset.py` (604 lines):**
  - **Transforms (`get_transforms`):**
    - *Train (standard):* `Resize(256)` → `RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15))` → `RandomHorizontalFlip(0.5)` → `RandomVerticalFlip(0.3)` → `RandomRotation(20)` → `ColorJitter(0.25, 0.25, 0.25, 0.1)` → `ToTensor` → ImageNet normalize → `RandomErasing(p=0.2, scale=(0.02, 0.2), value="random")`.
    - *Train (`--heavy_aug`, unused in SLURM scripts):* inserts `RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1))` and `GaussianBlur(3, sigma=(0.1,2.0))`.
    - *Eval:* `Resize(256)` → `CenterCrop(224)` → `ToTensor` → ImageNet normalize.
  - **`get_or_create_dataset_splits()`:** Reads `data/dataset_splits.json` if present; otherwise performs deterministic 70/15/15 stratified shuffle per class (`seed=42`) and writes POSIX relative paths.
  - **`CutMixMixupCollator` & `SoftTargetCrossEntropy`:** Applied at batch level (`prob=1.0`, `switch_prob=0.5` between CutMix `Beta(1.0, 1.0)` and MixUp `Beta(0.8, 0.8)`). `DataLoader` uses `drop_last=True` on train and `pin_memory=torch.cuda.is_available()`.
- **`src/models.py` (182 lines):**
  - `create_resnet50`: `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`, replaces `fc` (2048 → `Dropout(0.2)` + `Linear(2048, 29)`), sets `gradcam_target_layer = model.layer4[-1]`.
  - `create_swin_t`: `torchvision.models.swin_t(weights=Swin_T_Weights.DEFAULT)`, replaces `head` (768 → `Dropout(0.2)` + `Linear(768, 29)`), sets `gradcam_target_layer = model.features[-1]`.
  - `get_parameter_groups`: splits parameters into 4 AdamW groups (backbone/head × decay/no_decay where `ndim <= 1` or `.bias` get `weight_decay=0.0`, and `"fc" in name or "head" in name` receive `base_lr * head_lr_multiplier`).
- **`src/train.py` (402 lines):**
  - Deterministic seeding (`cudnn.deterministic=True`, `cudnn.benchmark=False`), AMP (`torch.cuda.amp.autocast` + `GradScaler`), gradient norm clipping (`max_norm=5.0`).
  - Scheduler: linear warmup over `warmup_epochs` (`(epoch + 1) / warmup_epochs`) followed by cosine decay (`0.5 * (1 + cos(pi * progress))`).
  - Evaluates `val_loader` every epoch and `test_pd_loader` every 5 epochs (plus final epoch).
  - Saves `last_checkpoint.pt` every epoch and `best_checkpoint.pt` when `val_f1 > best_val_f1`. Writes `config.json` and `metrics_history.csv`.
- **`src/evaluate.py` (365 lines):**
  - Loads `best_checkpoint.pt` (`pretrained=False`), evaluates `splits["test_pv"]` and full `PlantDoc` (`train/` + `test/`), computes aggregate & per-class metrics via `sklearn.metrics`, saves 3 PNG plots (`confusion_matrix_source_lab.png`, `confusion_matrix_target_field.png`, `per_class_domain_gap.png`) at 300 DPI and `domain_shift_report.json`.
- **`src/explainability.py` (563 lines):**
  - Provides `NativeGradCAM` (forward + full backward hook) and `PyTorchGradCAMEngine` wrapper, plus dynamic `swin_reshape_transform` (`side = isqrt(N)` converting `[B, H*W, C]` or `[B, H, W, C]` → `[B, C, H, W]`).
  - Computes 4 shortcut metrics per CAM (`center_focus_ratio` on inner 70%×70%, `border_shortcut_ratio` on outer 15%, `focal_concentration` on top 10% pixels, normalized `spatial_entropy`) and renders 3-panel PNGs at 250 DPI.

### Exact Hyperparameters & SLURM Configurations
*(Extracted directly from `slurm/*.slurm` and `src/train.py` defaults; `experiments/outputs/*/config.json` is not on local disk.)*

| Parameter | `resnet50_baseline` | `resnet50_cutmix_mixup` | `swin_t_baseline` | `swin_t_cutmix_mixup` |
| :--- | :--- | :--- | :--- | :--- |
| **SLURM Script** | `02_train_resnet.slurm` | `02_train_resnet.slurm` | `03_train_swin.slurm` | `03_train_swin.slurm` |
| **SLURM Resources** | 1× GPU, 8 CPU, 36 GB RAM, 4h | 1× GPU, 8 CPU, 36 GB RAM, 4h | 1× GPU, 8 CPU, 36 GB RAM, 4.5h | 1× GPU, 8 CPU, 36 GB RAM, 4.5h |
| **Model (`--model_name`)** | `resnet50` | `resnet50` | `swin_t` | `swin_t` |
| **Pretrained / Dropout** | `True` / `0.2` | `True` / `0.2` | `True` / `0.2` | `True` / `0.2` |
| **Epochs / Warmup** | `30` / `3` (default) | `30` / `3` (default) | `30` / `4` | `30` / `4` |
| **Batch Size / Workers** | `64` / `8` | `64` / `8` | `32` / `8` | `32` / `8` |
| **Backbone LR / Head Mult** | `1e-4` / `5.0` (`5e-4`) | `1e-4` / `5.0` (`5e-4`) | `5e-5` / `10.0` (`5e-4`) | `5e-5` / `10.0` (`5e-4`) |
| **Weight Decay** | `1e-2` | `1e-2` | `5e-2` | `5e-2` |
| **CutMix / MixUp** | Disabled (`CrossEntropyLoss`) | Enabled (`mixup=0.8`, `cutmix=1.0`, `p=1.0`, `switch=0.5`) | Disabled (`CrossEntropyLoss`) | Enabled (`mixup=0.8`, `cutmix=1.0`, `p=1.0`, `switch=0.5`) |
| **Heavy Aug / Label Smooth** | `False` / `0.0` | `False` / `0.0` | `False` / `0.0` | `False` / `0.0` |
| **AMP / Seed** | Enabled / `42` | Enabled / `42` | Enabled / `42` | Enabled / `42` |

- **Evaluation (`slurm/04_evaluate_all.slurm`):** 1× GPU, 8 CPU, 32 GB RAM, 1.5h; `--batch_size 32 --num_workers 4 --seed 42` on `checkpoints/best_checkpoint.pt`.
- **Explainability (`slurm/05_run_gradcam.slurm`):** 1× GPU, 4 CPU, 24 GB RAM, 1h; `--num_samples 12` per domain on `checkpoints/best_checkpoint.pt`.

---

## 5. Experimental Results & Logs

### Training Dynamics (`metrics_history.csv`)
**Unavailable on local disk:** `experiments/outputs/*/metrics_history.csv` and `experiments/outputs/*/config.json` are excluded by `.gitignore` and remain on VSC. Therefore, exact epoch-by-epoch training/validation loss curves, best checkpoint epoch numbers, and intermediate (every-5-epoch) PlantDoc F1 trajectories cannot be reported from local files. What is confirmed from the evaluation reports is that all 4 runs completed training and saved `/vsc-hard-mounts/leuven-data/393/vsc39394/crop-disease-domain-shift/experiments/outputs/<exp>/checkpoints/best_checkpoint.pt`.

### Cross-Domain Evaluation Summary
Compiled directly from `experiments/evaluation_results/*/domain_shift_report_<exp>.json`:

#### Table A: As-Reported Aggregate Metrics (All Samples: PV `N = 2,897`, PD `N = 2,922`)

| Experiment | Arch | PV Acc (%) | PV Macro F1 (%) | PV Weighted F1 (%) | PD Acc (%) | PD Macro F1 (%) | PD Weighted F1 (%) | $\Delta$ Acc (pp) | $\Delta$ Macro F1 (pp) | F1 Retention (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`resnet50_baseline`** | `resnet50` | **99.97** | **99.96** | **99.97** | 13.35 | 6.45 | 10.05 | 86.62 | 93.52 | 6.45 |
| **`resnet50_cutmix_mixup`** | `resnet50` | 99.79 | 99.80 | 99.79 | 14.54 | 7.57 *(28-cls)* | 11.39 | 85.25 | 92.23 | 7.58 |
| **`swin_t_baseline`** | `swin_t` | 99.83 | 99.76 | 99.83 | 14.13 | 6.87 | 10.06 | 85.69 | 92.89 | 6.89 |
| **`swin_t_cutmix_mixup`** | `swin_t` | 99.86 | 99.89 | 99.86 | **16.15** | **8.10** | **12.31** | **83.71** | **91.79** | **8.11** |

#### Table B: Denominator-Corrected & Shared-Class Recalculation
Because `sklearn.metrics.precision_recall_fscore_support(..., average="macro")` omits `labels=`, PV Macro F1 averages over **14** classes while PD Macro F1 averages over **29** classes for 3 models (which predicted `potato_healthy` at least once) vs. **28** classes for `resnet50_cutmix_mixup` (0 `potato_healthy` predictions). Filtering strictly to the **13 `MATCH` classes** (`N_PV = 2,873`, `N_PD = 1,407`) removes the 1,515 unseen-class PlantDoc samples:

| Experiment | PV Macro F1 (13 shared, %) | PD Macro F1 (all 29 cls, %) | PD Macro F1 (13 shared, %) | PD Shared Acc (13 cls, %) | Shared Correct / 1,407 | Shared $\Delta$ F1 (pp) | Shared F1 Retention (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`resnet50_baseline`** | **99.96** | 6.45 | 14.38 | 27.72 | 390 / 1,407 | 85.58 | 14.39 |
| **`resnet50_cutmix_mixup`** | 99.78 | 7.31 | 16.30 | 30.21 | 425 / 1,407 | 83.48 | 16.34 |
| **`swin_t_baseline`** | 99.74 | 6.87 | 15.33 | 29.35 | 413 / 1,407 | 84.41 | 15.37 |
| **`swin_t_cutmix_mixup`** | 99.88 | **8.10** | **18.07** | **33.55** | **472 / 1,407** | **81.81** | **18.09** |

**Key Comparative Takeaways (Even Under Fair 13-Class Comparison):**
1. **Swin-T > ResNet-50:** `swin_t_baseline` beats `resnet50_baseline` on shared PD Macro F1 (`15.33%` vs `14.38%`, `+0.95 pp`) and Accuracy (`29.35%` vs `27.72%`, `+1.63 pp`).
2. **CutMix/MixUp improves both backbones:** ResNet-50 shared Macro F1 rises `+1.92 pp` (`14.38%` → `16.30%`) and Swin-T rises `+2.74 pp` (`15.33%` → `18.07%`). Combined (`swin_t_cutmix_mixup`), shared PD Accuracy reaches `33.55%` (`+5.83 pp` over `resnet50_baseline`).
3. **In-domain ceiling:** All 4 models achieve `99.79%–99.97%` accuracy on PlantVillage `test_pv`.

#### Table C: Per-Class PlantDoc F1 (%) & False-Positive Attractor Analysis (13 Shared Classes + Attractor Counts)

| Canonical Class | PD Sup | ResNet-50 Base F1 (EstPred) | ResNet-50 CM+MU F1 (EstPred) | Swin-T Base F1 (EstPred) | Swin-T CM+MU F1 (EstPred) | Behavior / Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `tomato_yellow_leaf_curl_virus` | 238 | 44.00% *(112)* | **50.27%** *(136)* | 26.67% *(47)* | **54.50%** *(151)* | **Top performer** (High precision: 68.8–80.9%) |
| `tomato_septoria_leaf_spot` | 157 | 32.34% *(245)* | 33.99% *(196)* | **44.18%** *(178)* | 40.35% *(299)* | **Strong performer** (Consistent F1 32–44%) |
| `potato_early_blight` | 171 | 19.73% *(194)* | 23.21% *(165)* | 25.11% *(299)* | **26.15%** *(89)* | Moderate transfer |
| `potato_late_blight` | 208 | 20.63% *(44)* | 25.37% *(60)* | **25.52%** *(82)* | 22.38% *(78)* | Moderate precision (41–59%), low recall (12–18%) |
| `tomato_early_blight` | 88 | 24.88% *(314)* | 24.07% *(394)* | **25.58%** *(303)* | 24.15% *(293)* | Over-predicted (~3–4.5× support) |
| `bell_pepper_bacterial_spot` | 83 | 17.45% *(192)* | 19.12% *(304)* | 19.91% *(379)* | **24.73%** *(192)* | Moderate attractor (192–379 preds) |
| `tomato_late_blight` | 111 | 14.11% *(1,009)* | 14.97% *(784)* | 13.99% *(918)* | **15.80%** *(965)* | **#1 False-Positive Attractor (26.8–34.5% of all 2,922 PD preds; P = 7.8–8.8%)** |
| `bell_pepper_healthy` | 42 | 7.54% *(727)* | 7.22% *(761)* | **10.43%** *(591)* | 9.17% *(743)* | **#2 False-Positive Attractor (20.2–26.0% of all 2,922 PD preds; P = 3.8–5.6%)** |
| `tomato_leaf_mold` | 91 | 3.64% *(19)* | 6.67% *(29)* | 1.87% *(16)* | **7.62%** *(14)* | Near-collapse (R ≤ 4.4%) |
| `tomato_mosaic_virus` | 54 | 0.00% *(0)* | 3.51% *(3)* | 3.64% *(1)* | **6.90%** *(4)* | Near-collapse (≤ 2 TP) |
| `tomato_healthy` | 52 | 0.00% *(0)* | 2.27% *(36)* | 0.00% *(0)* | **3.12%** *(12)* | Collapsed in baselines (0–1 TP) |
| `tomato_bacterial_spot` | 110 | **2.65%** *(41)* | 1.23% *(53)* | 2.40% *(57)* | 0.00% *(0)* | Severe collapse (0–2 TP) |
| `tomato_spider_mites` | 2 | 0.00% *(0)* | 0.00% *(≤1)* | 0.00% *(0)* | 0.00% *(0)* | Collapsed (only 2 samples in PD `train/`) |

*Note:* `EstPred = round((recall * support) / precision)`. The remaining predictions (`25` for `resnet50_baseline`, `51` for `swin_t_baseline`, `82` for `swin_t_cutmix_mixup`, and `1` for `resnet50_cutmix_mixup`) went to zero-TP classes (`potato_healthy` in the first three; `tomato_spider_mites` in `resnet50_cutmix_mixup`). In the target confusion matrices (`confusion_matrix_target_field.png`), all 15 unseen `PD ONLY` rows drain almost entirely into the two vertical columns `bell_pepper_healthy` and `tomato_late_blight` (plus `bell_pepper_bacterial_spot`, `tomato_early_blight`, and `tomato_septoria_leaf_spot`).

### Grad-CAM Visualizations & SLURM Logs
- **SLURM Logs (`slurm/logs/`):** **Not present on local disk** (gitignored; remain on VSC). No warnings, errors, or stdout diagnostics from the cluster runs are available locally.
- **Grad-CAM Outputs (`experiments/gradcam_analysis/`):** **96 PNG files** exist on disk (**24 per experiment**: 12 PlantVillage Lab + 12 PlantDoc Field), generated using `NativeGradCAM` (`pytorch-grad-cam` is commented out in `requirements.txt`).
  - **12 PlantVillage Lab filenames per model** (`pv_<arch>_sample_{01..12}_...`): sampled from `bell_pepper_healthy` (×3), `tomato_yellow_leaf_curl_virus` (×4), `tomato_healthy` (×1), `tomato_spider_mites` (×2), `tomato_early_blight` (×1), `tomato_septoria_leaf_spot` (×1). All 12/12 are correctly predicted across all 4 models.
  - **12 PlantDoc Field filenames per model** (`pd_<arch>_sample_{01..12}_...`): sampled from `corn_leaf_blight` (unseen → predicted `bell_pepper_bacterial_spot`), `blueberry_healthy` (unseen → `bell_pepper_healthy` or `potato_healthy`), `tomato_bacterial_spot`, `apple_rust` (unseen → `bell_pepper_bacterial_spot`), `tomato_healthy`, `tomato_early_blight` (×2), `tomato_yellow_leaf_curl_virus`, `apple_scab` (unseen → `bell_pepper_healthy` or `tomato_late_blight`), `potato_early_blight`, `tomato_septoria_leaf_spot`, `strawberry_healthy` (unseen → `tomato_late_blight`). Only 2–3 out of 12 sampled field images are correctly predicted (`tomato_early_blight` sample 06 and `tomato_yellow_leaf_curl_virus` sample 07 across all 4 runs).
  - **Directly inspected shortcut metrics on `sample_06` (`tomato_early_blight` → `tomato_early_blight`, Field):**
    - `resnet50_baseline`: Center Focus `67.4%` | Border `32.6%` | Focal Conc (Top 10%) `30.0%` | Spatial Entropy `0.956`
    - `resnet50_cutmix_mixup`: Center Focus `82.3%` | Border `17.7%` | Focal Conc (Top 10%) `38.7%` | Spatial Entropy `0.938`
    - `swin_t_baseline`: Center Focus `62.4%` | Border `37.6%` | Focal Conc (Top 10%) `15.9%` | Spatial Entropy `0.986`
    - `swin_t_cutmix_mixup`: Center Focus `84.0%` | Border `16.0%` | Focal Conc (Top 10%) `33.0%` | Spatial Entropy `0.941`
  - **Visual defect observed in PNGs:** In `visualize_explanation()`, the top line (`Pred: ...`) of the 3-line panel-3 title is partially clipped at the upper image border by `plt.tight_layout()`.

---

## 6. Known Bugs, Methodological Bottlenecks & Next Steps

### Identified Bugs & Methodological Bottlenecks

1. **[CRITICAL — Methodological] 14-Class PlantVillage Subset vs. 28-Class PlantDoc Open-Set Mismatch:**
   - Training used only the 15-class Solanaceae subset of PlantVillage (14 mapped classes) while evaluating against all 28 PlantDoc classes, causing **1,515 / 2,922 (51.85%)** PlantDoc samples to come from completely unseen crops (`apple`, `blueberry`, `cherry`, `corn`, `grape`, `peach`, `raspberry`, `soybean`, `squash`, `strawberry`) and creating severe false-positive attractor collapse (`tomato_late_blight`, `bell_pepper_healthy`).
   - Additionally, `potato_healthy` (152 PV images) is trained even though PlantDoc contains 0 `potato_healthy` images.
2. **[HIGH — Metric Bug] `sklearn` Macro-F1 Denominator Mismatch (`src/evaluate.py`, `src/train.py`):**
   - `precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)` in `src/evaluate.py` (line 100) and `f1_score(..., average="macro", zero_division=0)` in `src/train.py` (lines 110, 159) omit `labels=`.
   - This makes Target Macro-F1 depend on whether `y_pred` happened to contain `potato_healthy` (`28` vs `29` classes: `resnet50_cutmix_mixup` got `7.57%` over 28 classes instead of `7.31%` over 29), and mixes unseen-class zeros into the benchmark score.
3. **[MEDIUM — Explainability Audit Bias & Bugs] `src/explainability.py`:**
   - `process_domain_samples()` scans all images via `rglob("*")` instead of restricting PlantVillage to `dataset_splits.json["test_pv"]` and restricting both domains to the **13 shared classes** (`active_common_classes`). Currently 5/12 PlantDoc Grad-CAM images are unseen classes.
   - Line 148 calls `score.backward(retain_graph=True)` unnecessarily (retains the autograd graph per sample).
   - No quantitative aggregation (mean/std of `center_focus_ratio`, `border_shortcut_ratio`, `focal_concentration`, `spatial_entropy` over all test samples) is exported to JSON/CSV.
   - Panel-3 title clips at the top of saved PNGs under `plt.tight_layout()`.
4. **[MEDIUM — Data/Split Governance] PlantDoc `train/` + `test/` Merging & Collator Determinism:**
   - `MappedCropDataset._load_samples_from_disk()` merges `PlantDoc/train/` (2,670) and `PlantDoc/test/` (252) together. While acceptable for pure zero-shot evaluation, it prevents clean few-shot fine-tuning without risking test leakage.
   - `CutMixMixupCollator` (`src/dataset.py` lines 435, 444) calls `np.random.beta` inside DataLoader workers without a `worker_init_fn`.
5. **[LOW — Repo & Artifact Sync] Missing Remote Training Logs & Renamed JSONs:**
   - `experiments/outputs/*/metrics_history.csv`, `config.json`, and `slurm/logs/*.{out,err}` are not synced from VSC due to `.gitignore`.
   - Local `domain_shift_report_<exp>.json` filenames diverge from `slurm/04_evaluate_all.slurm` (which globs `*/domain_shift_report.json`) and Git `HEAD` (`b1bee33e`).
   - Minor hygiene: unused `is_head_param = False` in `src/models.py` (line 140), substring `"fc" in name or "head" in name` parameter matching (`src/models.py` line 146), and illustrative numbers in `RUNBOOK.md`.

### Prioritized Next Steps (For A1 Poster & 5-Page Final Report)

1. **[P0 — Sync VSC Artifacts]** Pull `experiments/outputs/*/{config.json,metrics_history.csv}` and `slurm/logs/*.{out,err}` from VSC (`rsync` / `scp`) into a backup or unignore `config.json` / `metrics_history.csv` / `slurm/logs/` (keeping `*.pt` ignored) so training curves, best epochs, runtime, and SLURM diagnostics can be plotted in the report.
2. **[P0 — Resolve Closed-Set Dataset Scope]** Choose one clean, scientifically defensible protocol before re-running:
   - **Option A (Recommended & fastest on current VSC data):** Filter both `PlantVillage` and `PlantDoc` strictly to the **13 `active_common_classes`** (`num_classes = 13`, passing `allowed_classes=sorted(active_common_classes)` in `build_dataloaders()`, `evaluate.py`, and `explainability.py`), regenerate `dataset_splits.json`, and retrain all 4 models (~4–5 GPU hours total).
   - **Option B (Full 28/29-class benchmark):** Download the complete **38-class PlantVillage** dataset (adding Apple, Blueberry, Cherry, Corn, Grape, Peach, Raspberry, Soybean, Squash, Strawberry) into `$VSC_DATA/datasets/PlantVillage`, re-run `dataset_mapping.py` to confirm all 28+ overlapping classes have `support > 0`, regenerate `dataset_splits.json`, and retrain.
3. **[P1 — Fix Metric Calculation & `evaluate.py`]** Update `calculate_domain_metrics()` and `train.py` to compute Macro-F1 explicitly over `labels=active_class_ids` (classes with `support > 0` in the evaluated domain, or the aligned shared class list), eliminating the 28-vs-29 denominator bug and reporting both `overall` and `shared_classes_only` metrics.
4. **[P1 — Fix & Quantify Grad-CAM (`src/explainability.py`)]**
   - Sample PlantVillage strictly from `dataset_splits.json["test_pv"]` and only from shared classes (`active_common_classes`).
   - Remove `retain_graph=True`, add `plt.subplots_adjust(top=0.86)` / larger `figsize` to fix clipped panel titles, and add an aggregate CSV/JSON export computing mean ± std of all 4 shortcut metrics across 100–200 test samples per model × domain for statistical significance testing (ResNet-50 vs Swin-T; Baseline vs CutMix/MixUp) on the poster.
5. **[P2 — Report & Poster Preparation]** Reconcile `domain_shift_report.json` naming with `slurm/04_evaluate_all.slurm`, generate publication figures (learning curves from `metrics_history.csv`, clean 13×13 or 28×28 confusion matrices, per-class F1 gain waterfall/bar charts, and side-by-side 4-model Grad-CAM grids on identical PlantDoc field leaves), and update `RUNBOOK.md` numbers to match final observed results.
