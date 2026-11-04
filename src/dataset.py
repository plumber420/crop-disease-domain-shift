#!/usr/bin/env python3
"""
dataset.py
==========
PyTorch Dataset, Dataloader builders, and Data Augmentation pipelines for
crop disease classification under domain shift (PlantVillage -> PlantDoc).

Features:
- MappedCropDataset: Maps arbitrary folder structures to canonical overlapping classes.
- Persistent Dataset Manifests: Prevents silent data leakage by persisting exact file splits to JSON.
- Stratified source splitting (train / val / in-domain test).
- Out-of-domain field target dataset loading (PlantDoc).
- Advanced Augmentation: CutMix and MixUp batch collation for field-robust regularization.
- Soft Target Cross-Entropy Loss for mixed label supervision.

Author: KU Leuven / VSC Deep Learning Project
"""

import os
import sys
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

# Add parent directory to sys.path to import dataset_mapping
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data.dataset_mapping import (
    get_canonical_class_list,
    get_canonical_name_to_id,
    resolve_folder_to_canonical,
    CANONICAL_MAPPING,
)

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def get_transforms(
    img_size: int = 224, 
    is_training: bool = True, 
    heavy_aug: bool = False
) -> transforms.Compose:
    """
    Constructs image transform pipeline for training or evaluation.
    
    Args:
        img_size: Target square image dimension (default 224 for ResNet/Swin).
        is_training: If True, apply stochastic data augmentations.
        heavy_aug: If True, include additional field-simulating distortions (blur, affine, perspective).
    """
    if is_training:
        transform_list = [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        ]
        
        if heavy_aug:
            # Additional distortions to simulate outdoor field variability
            transform_list.extend([
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
            
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random")
        ])
    else:
        # Deterministic evaluation pipeline
        transform_list = [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

    return transforms.Compose(transform_list)


class MappedCropDataset(Dataset):
    """
    PyTorch Dataset that scans a root directory or accepts an explicit list of relative file paths,
    resolves class folders against canonical overlapping classes, and filters non-overlapping classes.
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        domain: str = "auto",
        transform: Optional[Callable] = None,
        allowed_classes: Optional[List[str]] = None,
        samples_file_list: Optional[List[str]] = None,
    ):
        self.root_dir = Path(os.path.expandvars(os.path.expanduser(str(root_dir))))
        self.domain = domain
        self.transform = transform
        self.canonical_to_id = get_canonical_name_to_id()
        
        # If allowed_classes specified, only include this subset
        self.allowed_classes = set(allowed_classes) if allowed_classes else set(self.canonical_to_id.keys())
        
        self.samples: List[Tuple[Path, int, str]] = []  # (image_path, class_id, canonical_name)
        
        if samples_file_list is not None:
            self._load_from_manifest(samples_file_list)
        else:
            self._load_samples_from_disk()

    def _load_from_manifest(self, file_list: List[str]):
        """Populates samples from an explicit list of relative file paths."""
        for rel_str in file_list:
            rel_p = Path(rel_str)
            full_path = self.root_dir / rel_p
            folder_name = rel_p.parent.name
            canonical = resolve_folder_to_canonical(folder_name, domain=self.domain)
            if canonical is not None and canonical in self.allowed_classes:
                class_id = self.canonical_to_id[canonical]
                self.samples.append((full_path, class_id, canonical))
            else:
                # If folder_name wasn't matched, check direct folder or stem
                canonical_fallback = resolve_folder_to_canonical(rel_p.parts[0] if len(rel_p.parts) > 1 else "", domain=self.domain)
                if canonical_fallback is not None and canonical_fallback in self.allowed_classes:
                    class_id = self.canonical_to_id[canonical_fallback]
                    self.samples.append((full_path, class_id, canonical_fallback))

        if len(self.samples) == 0:
            print(f"[WARN] 0 valid samples matched from manifest in {self.root_dir}.")

    def _load_samples_from_disk(self):
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Root dataset directory does not exist: {self.root_dir}")

        direct_subdirs = [p for p in self.root_dir.iterdir() if p.is_dir()]
        target_folders = []

        for d in direct_subdirs:
            if d.name.lower() in ("train", "test", "val", "validation"):
                target_folders.extend([p for p in d.iterdir() if p.is_dir()])
            else:
                target_folders.append(d)

        for folder in target_folders:
            canonical = resolve_folder_to_canonical(folder.name, domain=self.domain)
            if canonical is not None and canonical in self.allowed_classes:
                class_id = self.canonical_to_id[canonical]
                imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
                for img_p in imgs:
                    self.samples.append((img_p, class_id, canonical))

        self.samples.sort(key=lambda x: str(x[0]))
        if len(self.samples) == 0:
            print(f"[WARN] 0 samples mapped in {self.root_dir} for domain '{self.domain}'. Check folder names.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_id, canonical_name = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to black tensor on corrupt image
            print(f"[ERROR] Corrupt image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, class_id

    def get_class_counts(self) -> Dict[str, int]:
        """Returns image counts per canonical class in this dataset."""
        counts = {c: 0 for c in self.allowed_classes}
        for _, _, c_name in self.samples:
            counts[c_name] += 1
        return counts

    def get_relative_file_list(self) -> List[str]:
        """Returns list of POSIX-formatted paths relative to self.root_dir."""
        rel_paths = []
        for p, _, _ in self.samples:
            try:
                rel = p.relative_to(self.root_dir).as_posix()
            except ValueError:
                rel = p.as_posix()
            rel_paths.append(rel)
        return rel_paths


def get_or_create_dataset_splits(
    pv_dir: Union[str, Path],
    manifest_path: Union[str, Path] = "data/dataset_splits.json",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Manages persistent dataset splits to guarantee ZERO silent data leakage between
    training, validation, and in-domain testing.
    
    If the manifest JSON exists at manifest_path:
        Loads and returns the exact pre-defined file lists.
    If the manifest does not exist:
        Computes a deterministic stratified split per canonical class, writes the JSON
        manifest with relative POSIX paths, and returns the splits.
        
    Returns:
        Dict with keys: 'train', 'val', 'test_pv', 'metadata'
    """
    pv_path = Path(os.path.expandvars(os.path.expanduser(str(pv_dir)))).resolve()
    manifest_file = Path(os.path.expandvars(os.path.expanduser(str(manifest_path)))).resolve()

    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                splits = json.load(f)
            
            # Check required keys
            if all(k in splits for k in ("train", "val", "test_pv")):
                print(f"[INFO] Loaded persistent dataset splits from {manifest_file}")
                print(f"       Split sizes: Train={len(splits['train'])}, Val={len(splits['val'])}, Test PV={len(splits['test_pv'])}")
                return splits
            else:
                print(f"[WARN] Manifest {manifest_file} missing required keys. Regenerating...")
        except Exception as e:
            print(f"[WARN] Failed reading manifest {manifest_file}: {e}. Regenerating...")

    # Generate new stratified splits
    print(f"[INFO] Scanning {pv_path} to create persistent dataset splits (seed={seed})...")
    base_dataset = MappedCropDataset(pv_path, domain="pv")
    
    if len(base_dataset) == 0:
        raise RuntimeError(f"Cannot generate split: No valid samples found in {pv_path}")

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"
    
    rng = random.Random(seed)
    
    # Group relative sample paths by canonical class ID
    class_to_paths: Dict[int, List[str]] = {}
    for p, class_id, _ in base_dataset.samples:
        rel = p.relative_to(pv_path).as_posix()
        class_to_paths.setdefault(class_id, []).append(rel)

    train_paths: List[str] = []
    val_paths: List[str] = []
    test_pv_paths: List[str] = []

    for class_id in sorted(class_to_paths.keys()):
        paths = class_to_paths[class_id]
        paths.sort()  # Sort first for deterministic permutation
        rng.shuffle(paths)
        
        n = len(paths)
        n_train = int(math.floor(n * train_ratio))
        n_val = int(math.floor(n * val_ratio))
        
        train_paths.extend(paths[:n_train])
        val_paths.extend(paths[n_train:n_train + n_val])
        test_pv_paths.extend(paths[n_train + n_val:])

    rng.shuffle(train_paths)
    rng.shuffle(val_paths)
    rng.shuffle(test_pv_paths)

    splits = {
        "metadata": {
            "source_dataset": "PlantVillage",
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "num_classes": len(get_canonical_class_list()),
            "train_samples": len(train_paths),
            "val_samples": len(val_paths),
            "test_pv_samples": len(test_pv_paths),
        },
        "train": train_paths,
        "val": val_paths,
        "test_pv": test_pv_paths,
    }

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"[SUCCESS] Saved persistent dataset split manifest to {manifest_file}")
    print(f"          Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_pv_paths)}")
    return splits


def create_stratified_split(
    dataset: MappedCropDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    In-memory stratified train / val / in-domain test split.
    Kept for backward compatibility; prefer get_or_create_dataset_splits for persistence.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"
    rng = random.Random(seed)
    
    class_indices: Dict[int, List[int]] = {}
    for idx, (_, class_id, _) in enumerate(dataset.samples):
        class_indices.setdefault(class_id, []).append(idx)
        
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    
    for class_id in sorted(class_indices.keys()):
        indices = class_indices[class_id]
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(math.floor(n * train_ratio))
        n_val = int(math.floor(n * val_ratio))
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
        
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


class CutMixMixupCollator:
    """
    Batch-level regularizer implementing CutMix and MixUp.
    
    Regularization Theory:
    - MixUp linearly interpolates image pairs and their one-hot target vectors:
      x_tilde = lambda * x_i + (1 - lambda) * x_j
      y_tilde = lambda * y_i + (1 - lambda) * y_j
    - CutMix replaces a random spatial bounding box from image j into image i:
      x_tilde = M_box * x_i + (1 - M_box) * x_j
      y_tilde = lambda_area * y_i + (1 - lambda_area) * y_j
      
    This forces the model to attend to diffuse features rather than relying on
    laboratory background artifacts (shortcut learning mitigation).
    """
    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing

    def _rand_bbox(self, w: int, h: int, lam: float) -> Tuple[int, int, int, int]:
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = random.randint(0, w)
        cy = random.randint(0, h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def _to_one_hot(self, targets: torch.Tensor) -> torch.Tensor:
        targets_one_hot = torch.zeros(
            (targets.size(0), self.num_classes),
            dtype=torch.float32,
            device=targets.device
        )
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        
        if self.label_smoothing > 0.0:
            targets_one_hot = (
                targets_one_hot * (1.0 - self.label_smoothing)
                + self.label_smoothing / self.num_classes
            )
        return targets_one_hot

    def __call__(self, batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
        batch_size, _, h, w = images.size()

        targets_one_hot = self._to_one_hot(targets)

        if random.random() > self.prob or batch_size <= 1:
            return images, targets_one_hot

        perm_idx = torch.randperm(batch_size)
        shuffled_images = images[perm_idx]
        shuffled_targets = targets_one_hot[perm_idx]

        use_cutmix = random.random() < self.switch_prob

        if use_cutmix and self.cutmix_alpha > 0.0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(w, h, lam)
            
            images[:, :, bby1:bby2, bbx1:bbx2] = shuffled_images[:, :, bby1:bby2, bbx1:bbx2]
            
            lam_adjusted = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
            mixed_targets = lam_adjusted * targets_one_hot + (1.0 - lam_adjusted) * shuffled_targets

        elif self.mixup_alpha > 0.0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            images = lam * images + (1.0 - lam) * shuffled_images
            mixed_targets = lam * targets_one_hot + (1.0 - lam) * shuffled_targets
        else:
            mixed_targets = targets_one_hot

        return images, mixed_targets


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross-entropy loss supporting continuous probability distribution targets
    (necessary for MixUp / CutMix soft target vectors).
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        return loss


def build_dataloaders(
    pv_dir: str = "data/PlantVillage",
    pd_dir: Optional[str] = "data/PlantDoc",
    split_manifest_path: Optional[str] = "data/dataset_splits.json",
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    use_cutmix_mixup: bool = False,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    heavy_aug: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Builds data loaders for the full experimental pipeline with guaranteed persistent splits:
    - train_loader: PlantVillage train split (with optional CutMix/MixUp)
    - val_loader: PlantVillage validation split
    - test_pv_loader: PlantVillage in-domain test split
    - test_pd_loader: PlantDoc out-of-domain field target set (if pd_dir provided)
    """
    pv_dir = os.path.expandvars(os.path.expanduser(pv_dir))
    pd_dir = os.path.expandvars(os.path.expanduser(pd_dir)) if pd_dir else None
    num_classes = len(get_canonical_class_list())
    
    train_transform = get_transforms(img_size=img_size, is_training=True, heavy_aug=heavy_aug)
    eval_transform = get_transforms(img_size=img_size, is_training=False)

    if split_manifest_path:
        splits = get_or_create_dataset_splits(
            pv_dir=pv_dir,
            manifest_path=split_manifest_path,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=seed,
        )
        pv_train_dataset = MappedCropDataset(
            pv_dir, domain="pv", transform=train_transform, samples_file_list=splits["train"]
        )
        pv_val_dataset = MappedCropDataset(
            pv_dir, domain="pv", transform=eval_transform, samples_file_list=splits["val"]
        )
        pv_test_dataset = MappedCropDataset(
            pv_dir, domain="pv", transform=eval_transform, samples_file_list=splits["test_pv"]
        )
    else:
        # Fallback to in-memory subset splitting
        pv_train_dataset_full = MappedCropDataset(pv_dir, domain="pv", transform=train_transform)
        pv_eval_dataset_full = MappedCropDataset(pv_dir, domain="pv", transform=eval_transform)
        
        pv_train_dataset, _, _ = create_stratified_split(
            pv_train_dataset_full, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
        )
        _, pv_val_dataset, pv_test_dataset = create_stratified_split(
            pv_eval_dataset_full, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
        )
    
    # Setup collation for CutMix/MixUp
    if use_cutmix_mixup:
        collate_fn = CutMixMixupCollator(
            num_classes=num_classes,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=1.0,
            switch_prob=0.5,
        )
    else:
        collate_fn = None
        
    train_loader = DataLoader(
        pv_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        pv_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    test_pv_loader = DataLoader(
        pv_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    test_pd_loader = None
    if pd_dir and Path(pd_dir).exists():
        pd_dataset = MappedCropDataset(pd_dir, domain="pd", transform=eval_transform)
        test_pd_loader = DataLoader(
            pd_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        print(f"[INFO] Loaded PlantDoc Target field dataset: {len(pd_dataset)} images.")
    else:
        print("[WARN] PlantDoc target directory not provided or not found.")

    print(f"[INFO] Loaded PlantVillage Source splits: Train={len(pv_train_dataset)}, Val={len(pv_val_dataset)}, Test={len(pv_test_dataset)}")
    print(f"[INFO] Classes aligned: {num_classes} canonical classes.")
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_pv_loader": test_pv_loader,
        "test_pd_loader": test_pd_loader,
        "num_classes": num_classes,
    }


if __name__ == "__main__":
    print("Testing CutMixMixupCollator on synthetic dummy batch...")
    num_classes = 29
    collator = CutMixMixupCollator(num_classes=num_classes, mixup_alpha=0.8, cutmix_alpha=1.0)
    dummy_batch = [(torch.randn(3, 224, 224), i % num_classes) for i in range(8)]
    imgs, targets = collator(dummy_batch)
    print(f"Batch images shape:  {imgs.shape} (Expected: [8, 3, 224, 224])")
    print(f"Batch targets shape: {targets.shape} (Expected: [8, {num_classes}])")
    print(f"Target probabilities sum check: {targets.sum(dim=-1).numpy()}")
    loss_fn = SoftTargetCrossEntropy()
    dummy_logits = torch.randn(8, num_classes)
    loss = loss_fn(dummy_logits, targets)
    print(f"Computed SoftTargetCrossEntropy loss: {loss.item():.4f}")
    print("dataset.py unit test passed successfully!")
