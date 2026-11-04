#!/usr/bin/env python3
"""
explainability.py
=================
Grad-CAM (Gradient-weighted Class Activation Mapping) explainability pipeline
to detect and visualize Shortcut Learning in deep learning crop pathology models.

Supported Architectures:
- ResNet-50: Standard 4D Convolutional network (Target: layer4[-1])
- Swin Transformer (Swin-T): Hierarchical Vision Transformer with shifted window
  self-attention (Target: features[-1] via spatial reshape transform)

Research Objectives:
- Determine whether models trained on laboratory images (PlantVillage) attend to actual
  pathological lesion symptoms or rely on background shortcuts (studio lighting,
  uniform background cards, leaf silhouettes).
- Quantify attention dispersion and off-center border activation under outdoor field
  conditions (PlantDoc).
- Compare inductive bias robustness between CNNs and Vision Transformers.

Author: KU Leuven / VSC Deep Learning Project
"""

import os
import sys
import math
import glob
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Optional import of pytorch-grad-cam library
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    PYTORCH_GRAD_CAM_AVAILABLE = True
except ImportError:
    PYTORCH_GRAD_CAM_AVAILABLE = False

# Add repo root to sys.path
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.dataset import (
    MappedCropDataset,
    get_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VALID_EXTENSIONS,
)
from src.models import build_model
from data.dataset_mapping import (
    get_canonical_class_list,
    get_id_to_canonical_name,
    resolve_folder_to_canonical,
)


def swin_reshape_transform(tensor: torch.Tensor, height: Optional[int] = None, width: Optional[int] = None) -> torch.Tensor:
    """
    Transforms Swin Transformer activations/gradients from sequence [B, H*W, C] 
    or channels-last format [B, H, W, C] to standard spatial tensor format [B, C, H, W]
    for 2D Grad-CAM backpropagation and pooling. Dynamically computes height and width from sequence length.
    """
    if tensor.ndim == 3:
        B, N, C = tensor.shape
        if height is None or width is None:
            side = int(math.isqrt(N))
            height, width = side, side
        result = tensor.reshape(B, height, width, C)
        return result.permute(0, 3, 1, 2).contiguous()
    elif tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2).contiguous()
    return tensor


class NativeGradCAM:
    """
    Self-contained, robust native PyTorch implementation of Grad-CAM.
    Supports both 4D Convolutional networks (ResNet-50) and Vision Transformers
    (Swin-T) via custom spatial reshape transforms without external library dependencies.
    """
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: nn.Module,
        reshape_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates Grad-CAM activation heatmap for a single image tensor [1, 3, H, W].
        
        Returns:
            cam_map: 2D numpy array [H, W] normalized in [0, 1]
            predicted_class: integer index
            confidence: softmax probability of predicted class
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=-1)
        
        pred_class = int(torch.argmax(logits, dim=-1).item())
        confidence = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        # Target score for backpropagation
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        # Apply reshape transform for Vision Transformers if configured
        if self.reshape_transform is not None:
            activations = self.reshape_transform(activations)
            gradients = self.reshape_transform(gradients)

        # Global average pooling of gradients -> importance weights alpha_k [1, C, 1, 1]
        alpha = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of forward activation maps
        cam = torch.sum(alpha * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Discard negative influences

        # Resize to original input spatial dimensions
        _, _, H, W = input_tensor.shape
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, confidence

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


class PyTorchGradCAMEngine:
    """
    Wrapper around the official pytorch_grad_cam library for comparative verification.
    """
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: nn.Module, 
        reshape_transform: Optional[Callable] = None
    ):
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform
        self.cam = GradCAM(
            model=model, 
            target_layers=[target_layer], 
            reshape_transform=reshape_transform
        )

    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=-1)
            pred_class = int(torch.argmax(logits, dim=-1).item())
            confidence = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        cam = grayscale_cam[0, :]
        return cam, pred_class, confidence

    def remove_hooks(self):
        pass


def build_gradcam_engine(
    model: nn.Module,
    target_layer: nn.Module,
    reshape_transform: Optional[Callable] = None,
    prefer_library: bool = True
) -> Union[PyTorchGradCAMEngine, NativeGradCAM]:
    """Factory creating Grad-CAM engine with automatic fallback."""
    if prefer_library and PYTORCH_GRAD_CAM_AVAILABLE:
        try:
            return PyTorchGradCAMEngine(model, target_layer, reshape_transform=reshape_transform)
        except Exception as e:
            print(f"[WARN] pytorch_grad_cam initialization note ({e}). Using robust NativeGradCAM.")
    return NativeGradCAM(model, target_layer, reshape_transform=reshape_transform)


def overlay_cam_on_image(
    raw_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Blends Grad-CAM heatmap over the original PIL image."""
    img_np = np.array(raw_image.resize((224, 224))).astype(np.float32) / 255.0
    
    cmap_fn = plt.get_cmap(colormap)
    heatmap = cmap_fn(cam)[:, :, :3]

    blended = alpha * heatmap + (1.0 - alpha) * img_np
    blended = np.clip(blended, 0.0, 1.0)
    return (blended * 255).astype(np.uint8)


def compute_shortcut_metric(cam: np.ndarray) -> Dict[str, float]:
    """
    Computes diagnostic metrics for shortcut learning and feature concentration:
    1. center_focus_ratio: Mass in center 70%x70% bounding area.
    2. border_shortcut_ratio: Mass in outer 15% margins.
    3. focal_concentration: Fraction of total activation energy captured by top 10% pixels.
       - High (> 0.50): Concentrated focal attention on distinct pathological lesions.
       - Low (< 0.25): Diffuse, non-specific activation across wide background regions.
    4. spatial_entropy: Normalized Shannon spatial entropy [0, 1].
       - Closer to 0: highly focal activation.
       - Closer to 1: diffuse, uniform activation across the entire frame.
    """
    H, W = cam.shape
    total_energy = float(cam.sum() + 1e-8)

    # 1. Geometric Center vs Border
    margin_y, margin_x = int(H * 0.15), int(W * 0.15)
    center_mask = np.zeros_like(cam, dtype=bool)
    center_mask[margin_y:H - margin_y, margin_x:W - margin_x] = True

    center_energy = float(cam[center_mask].sum())
    border_energy = float(cam[~center_mask].sum())

    # 2. Focal Concentration Ratio (Top 10% pixels)
    flattened = np.sort(cam.ravel())[::-1]
    top_10_count = max(1, int(0.10 * len(flattened)))
    top_10_energy = float(flattened[:top_10_count].sum())
    focal_ratio = float(top_10_energy / total_energy)

    # 3. Spatial Entropy (Normalized Shannon Entropy)
    prob_dist = cam / total_energy
    prob_nonzero = prob_dist[prob_dist > 1e-12]
    entropy = -float(np.sum(prob_nonzero * np.log2(prob_nonzero)))
    max_entropy = float(np.log2(H * W))
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    return {
        "center_focus_ratio": center_energy / total_energy,
        "border_shortcut_ratio": border_energy / total_energy,
        "focal_concentration": focal_ratio,
        "spatial_entropy": norm_entropy,
    }


def visualize_explanation(
    raw_image: Image.Image,
    cam: np.ndarray,
    pred_name: str,
    true_name: str,
    confidence: float,
    output_path: Path,
    model_name: str = "resnet50",
    target_layer_name: str = "layer4[-1]",
    domain_tag: str = "PlantVillage (Lab)",
    is_field_domain: bool = False,
):
    """
    Generates a 3-panel explainability figure:
    1. Raw Input Image
    2. Grad-CAM Heatmap
    3. Heatmap Overlay + Quantitative Shortcut Diagnostics
    """
    stats = compute_shortcut_metric(cam)
    overlay = overlay_cam_on_image(raw_image, cam, alpha=0.55, colormap="jet")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    # 1. Raw image
    axes[0].imshow(raw_image.resize((224, 224)))
    axes[0].set_title(f"Input ({domain_tag})\nTrue: {true_name}", fontsize=11)
    axes[0].axis("off")

    # 2. Heatmap
    im2 = axes[1].imshow(cam, cmap="jet")
    axes[1].set_title(f"Grad-CAM [{model_name.upper()}]\nTarget: {target_layer_name}", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Blended Overlay & Diagnostics
    axes[2].imshow(overlay)
    
    # Border diagnostic label with domain-appropriate context warning
    if is_field_domain:
        border_text = f"Border: {stats['border_shortcut_ratio']*100:.1f}% (Note: Field lesions may be off-center)"
    else:
        border_text = f"Border Shortcut: {stats['border_shortcut_ratio']*100:.1f}%"

    status_color = "darkgreen" if pred_name == true_name else "darkred"
    diag_summary = (
        f"Pred: {pred_name} ({confidence*100:.1f}%)\n"
        f"Center Focus: {stats['center_focus_ratio']*100:.1f}% | {border_text}\n"
        f"Focal Conc (Top 10%): {stats['focal_concentration']*100:.1f}% | Spatial Entropy: {stats['spatial_entropy']:.3f}"
    )
    axes[2].set_title(diag_summary, fontsize=9.5, color=status_color)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()
    print(f"[INFO] Saved Grad-CAM figure to {output_path}")


def process_domain_samples(
    dataset_dir: str,
    domain: str,
    model: nn.Module,
    cam_engine: Any,
    transform: Any,
    class_names: List[str],
    id_to_canonical: Dict[int, str],
    output_subdir: Path,
    model_name: str = "resnet50",
    target_layer_name: str = "layer4[-1]",
    num_samples: int = 6,
    device: torch.device = torch.device("cpu"),
):
    """Samples images from a domain, generates Grad-CAM visualizations and shortcut reports."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"[WARN] Directory not found for explainability: {dataset_dir}")
        return

    all_img_paths = [p for p in dataset_path.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS]
    if not all_img_paths:
        print(f"[WARN] No valid images found in {dataset_dir}")
        return

    valid_samples = []
    for p in all_img_paths:
        canon = resolve_folder_to_canonical(p.parent.name, domain=domain)
        if canon:
            valid_samples.append((p, canon))

    random.seed(42)
    random.shuffle(valid_samples)
    chosen_samples = valid_samples[:num_samples]

    domain_label = "PlantVillage (Lab)" if domain == "pv" else "PlantDoc (Field)"
    is_field = (domain == "pd")
    print(f"\n[INFO] Generating Grad-CAM for {len(chosen_samples)} samples in {domain_label} [{model_name.upper()}]...")

    for i, (img_path, true_canon) in enumerate(chosen_samples):
        try:
            raw_img = Image.open(img_path).convert("RGB")
            input_tensor = transform(raw_img).unsqueeze(0).to(device)
            
            cam, pred_id, conf = cam_engine.generate_cam(input_tensor)
            pred_canon = id_to_canonical.get(pred_id, f"Class_{pred_id}")

            out_filename = output_subdir / f"{domain}_{model_name}_sample_{i+1:02d}_{true_canon}_pred_{pred_canon}.png"
            visualize_explanation(
                raw_image=raw_img,
                cam=cam,
                pred_name=pred_canon,
                true_name=true_canon,
                confidence=conf,
                output_path=out_filename,
                model_name=model_name,
                target_layer_name=target_layer_name,
                domain_tag=domain_label,
                is_field_domain=is_field,
            )
        except Exception as e:
            print(f"[ERROR] Failed processing {img_path}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Grad-CAM Explainability & Shortcut Learning Analysis.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet50", "swin_t"], help="Architecture ('resnet50' or 'swin_t')")
    parser.add_argument("--pv_dir", type=str, default="data/PlantVillage", help="Path to PlantVillage directory (Lab domain)")
    parser.add_argument("--pd_dir", type=str, default="data/plantdoc", help="Path to PlantDoc directory (Field domain)")
    parser.add_argument("--image_path", type=str, default="", help="Optional single image path for targeted attribution")
    parser.add_argument("--output_dir", type=str, default="experiments/gradcam_analysis", help="Directory to save visual explanations")
    parser.add_argument("--num_samples", type=int, default=6, help="Number of random samples to visualize per domain")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = get_canonical_class_list()
    id_to_canonical = get_id_to_canonical_name()
    num_classes = len(class_names)

    # 1. Determine model architecture from checkpoint if not specified or override
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_model = checkpoint.get("model_name") or checkpoint.get("config", {}).get("model_name")
    model_name = args.model_name or ckpt_model or "resnet50"
    
    print("=" * 80)
    print(" GRAD-CAM EXPLAINABILITY & SHORTCUT LEARNING AUDIT")
    print(f" Architecture: {model_name.upper()}")
    print(f" Checkpoint:   {args.checkpoint}")
    print(f" Device:       {device}")
    print(f" Output Dir:   {out_dir}")
    print("=" * 80)

    # 2. Reconstruct Model
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Configure Target Layer and Reshape Transform
    if model_name in ("resnet50", "resnet_50"):
        target_layer = model.layer4[-1]
        target_layer_name = "layer4[-1] (Bottleneck Stage 4)"
        reshape_transform = None
    elif model_name in ("swin_t", "swin_tiny", "swin"):
        target_layer = model.features[-1]
        target_layer_name = "features[-1] (Stage 4 Self-Attention)"
        reshape_transform = swin_reshape_transform
    else:
        target_layer = getattr(model, "gradcam_target_layer", list(model.children())[-2])
        target_layer_name = str(target_layer.__class__.__name__)
        reshape_transform = None

    print(f"[INFO] Target Layer: {target_layer_name}")

    # 4. Initialize Grad-CAM Engine
    cam_engine = build_gradcam_engine(
        model=model,
        target_layer=target_layer,
        reshape_transform=reshape_transform,
        prefer_library=True,
    )

    eval_transform = get_transforms(img_size=224, is_training=False)

    # 5. Single image mode
    if args.image_path:
        p = Path(args.image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {args.image_path}")
        raw_img = Image.open(p).convert("RGB")
        input_tensor = eval_transform(raw_img).unsqueeze(0).to(device)
        cam, pred_id, conf = cam_engine.generate_cam(input_tensor)
        pred_canon = id_to_canonical.get(pred_id, f"Class_{pred_id}")
        true_canon = resolve_folder_to_canonical(p.parent.name) or "Unknown"

        out_file = out_dir / f"single_{model_name}_{p.stem}_pred_{pred_canon}.png"
        visualize_explanation(
            raw_image=raw_img,
            cam=cam,
            pred_name=pred_canon,
            true_name=true_canon,
            confidence=conf,
            output_path=out_file,
            model_name=model_name,
            target_layer_name=target_layer_name,
            domain_tag="Target Image",
            is_field_domain=True,
        )
        cam_engine.remove_hooks()
        return

    # 6. Domain Comparative Mode: Process PlantVillage (Lab) vs PlantDoc (Field)
    if args.pv_dir:
        pv_out = out_dir / f"{model_name}_plantvillage_lab"
        pv_out.mkdir(parents=True, exist_ok=True)
        process_domain_samples(
            dataset_dir=args.pv_dir,
            domain="pv",
            model=model,
            cam_engine=cam_engine,
            transform=eval_transform,
            class_names=class_names,
            id_to_canonical=id_to_canonical,
            output_subdir=pv_out,
            model_name=model_name,
            target_layer_name=target_layer_name,
            num_samples=args.num_samples,
            device=device,
        )

    if args.pd_dir:
        pd_out = out_dir / f"{model_name}_plantdoc_field"
        pd_out.mkdir(parents=True, exist_ok=True)
        process_domain_samples(
            dataset_dir=args.pd_dir,
            domain="pd",
            model=model,
            cam_engine=cam_engine,
            transform=eval_transform,
            class_names=class_names,
            id_to_canonical=id_to_canonical,
            output_subdir=pd_out,
            model_name=model_name,
            target_layer_name=target_layer_name,
            num_samples=args.num_samples,
            device=device,
        )

    cam_engine.remove_hooks()
    print(f"\n[SUCCESS] Grad-CAM explainability batch generation for {model_name.upper()} completed successfully.")


if __name__ == "__main__":
    main()
