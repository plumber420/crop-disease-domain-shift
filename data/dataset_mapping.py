#!/usr/bin/env python3
"""
dataset_mapping.py
==================
Module for aligning and mapping classes between the laboratory source dataset (PlantVillage)
and the field target dataset (PlantDoc).

Key Challenge:
    PlantVillage contains 38 classes (single leaves on uniform/lab backgrounds).
    PlantDoc contains ~27 classes (field conditions: complex background, varying lighting,
    overlapping leaves, dirt/mud).
    To rigorously study domain shift without label mismatch, both datasets must be projected
    onto a unified set of overlapping canonical classes.

Author: KU Leuven / VSC Deep Learning Project
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path


# Canonical definition of overlapping classes between PlantVillage and PlantDoc
# Structure: canonical_name -> {"pv_candidates": [...], "pd_candidates": [...], "description": str}
CANONICAL_MAPPING: Dict[str, Dict[str, List[str]]] = {
    "apple_scab": {
        "pv_candidates": [
            "Apple___Apple_scab", "apple_scab", "Apple_scab", "Apple_Apple_scab"
        ],
        "pd_candidates": [
            "Apple Scab Leaf", "Apple_Scab_Leaf", "apple scab leaf", "Apple scab", "Apple_scab"
        ],
        "description": "Apple - Scab (Venturia inaequalis)",
    },
    "apple_rust": {
        "pv_candidates": [
            "Apple___Cedar_apple_rust", "Apple_cedar_apple_rust", "apple_rust", "Apple_Cedar_apple_rust"
        ],
        "pd_candidates": [
            "Apple rust leaf", "Apple_rust_leaf", "apple rust leaf", "Apple_rust"
        ],
        "description": "Apple - Cedar Apple Rust (Gymnosporangium juniperi-virginianae)",
    },
    "apple_healthy": {
        "pv_candidates": [
            "Apple___healthy", "Apple_healthy", "apple_healthy"
        ],
        "pd_candidates": [
            "Apple leaf", "Apple_leaf", "apple leaf", "Apple Leaf", "Apple_healthy"
        ],
        "description": "Apple - Healthy Leaf",
    },
    "bell_pepper_bacterial_spot": {
        "pv_candidates": [
            "Pepper,_bell___Bacterial_spot",
            "Pepper__bell___Bacterial_spot",
            "Pepper_bell_bacterial_spot",
            "Pepper_bacterial_spot",
            "Pepper_bell_Bacterial_spot"
        ],
        "pd_candidates": [
            "Bell_pepper leaf spot",
            "Bell_pepper_leaf_spot",
            "Bell pepper leaf spot",
            "bell pepper leaf spot"
        ],
        "description": "Bell Pepper - Bacterial Spot (Xanthomonas)",
    },
    "bell_pepper_healthy": {
        "pv_candidates": [
            "Pepper,_bell___healthy",
            "Pepper__bell___healthy",
            "Pepper_bell_healthy",
            "Pepper_healthy"
        ],
        "pd_candidates": [
            "Bell_pepper leaf",
            "Bell_pepper_leaf",
            "Bell pepper leaf",
            "bell pepper leaf"
        ],
        "description": "Bell Pepper - Healthy Leaf",
    },
    "blueberry_healthy": {
        "pv_candidates": ["Blueberry___healthy", "Blueberry_healthy", "blueberry_healthy"],
        "pd_candidates": ["Blueberry leaf", "Blueberry_leaf", "blueberry leaf"],
        "description": "Blueberry - Healthy Leaf",
    },
    "cherry_healthy": {
        "pv_candidates": [
            "Cherry_(including_sour)___healthy",
            "Cherry___healthy",
            "Cherry_healthy"
        ],
        "pd_candidates": ["Cherry leaf", "Cherry_leaf", "cherry leaf"],
        "description": "Cherry - Healthy Leaf",
    },
    "corn_gray_leaf_spot": {
        "pv_candidates": [
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_Cercospora_leaf_spot",
            "Corn_gray_leaf_spot",
            "Corn_Gray_leaf_spot"
        ],
        "pd_candidates": [
            "Corn Gray leaf spot",
            "Corn_Gray_leaf_spot",
            "corn gray leaf spot"
        ],
        "description": "Corn - Gray Leaf Spot (Cercospora zeae-maydis)",
    },
    "corn_leaf_blight": {
        "pv_candidates": [
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn___Northern_Leaf_Blight",
            "Corn_Northern_Leaf_Blight",
            "Corn_leaf_blight"
        ],
        "pd_candidates": [
            "Corn leaf blight",
            "Corn_leaf_blight",
            "corn leaf blight",
            "Corn Northern leaf blight"
        ],
        "description": "Corn - Northern Leaf Blight (Exserohilum turcicum)",
    },
    "corn_rust": {
        "pv_candidates": [
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Common_rust",
            "Corn___Common_rust_",
            "Corn_Common_rust",
            "Corn_rust"
        ],
        "pd_candidates": [
            "Corn rust leaf",
            "Corn_rust_leaf",
            "corn rust leaf",
            "Corn Common rust"
        ],
        "description": "Corn - Common Rust (Puccinia sorghi)",
    },
    "grape_black_rot": {
        "pv_candidates": ["Grape___Black_rot", "Grape_black_rot", "grape_black_rot"],
        "pd_candidates": [
            "grape leaf black rot",
            "Grape leaf black rot",
            "Grape_leaf_black_rot",
            "grape_leaf_black_rot",
            "grape leaf rot"
        ],
        "description": "Grape - Black Rot (Guignardia bidwellii)",
    },
    "grape_healthy": {
        "pv_candidates": ["Grape___healthy", "Grape_healthy", "grape_healthy"],
        "pd_candidates": ["grape leaf", "Grape leaf", "Grape_leaf", "grape_leaf"],
        "description": "Grape - Healthy Leaf",
    },
    "peach_healthy": {
        "pv_candidates": ["Peach___healthy", "Peach_healthy", "peach_healthy"],
        "pd_candidates": ["Peach leaf", "Peach_leaf", "peach leaf", "Peach_leaf"],
        "description": "Peach - Healthy Leaf",
    },
    "potato_early_blight": {
        "pv_candidates": [
            "Potato___Early_blight", "Potato_early_blight", "potato_early_blight", "Potato_Early_blight"
        ],
        "pd_candidates": [
            "Potato leaf early blight",
            "Potato_leaf_early_blight",
            "potato leaf early blight"
        ],
        "description": "Potato - Early Blight (Alternaria solani)",
    },
    "potato_late_blight": {
        "pv_candidates": [
            "Potato___Late_blight", "Potato_late_blight", "potato_late_blight", "Potato_Late_blight"
        ],
        "pd_candidates": [
            "Potato leaf late blight",
            "Potato_leaf_late_blight",
            "potato leaf late blight"
        ],
        "description": "Potato - Late Blight (Phytophthora infestans)",
    },
    "potato_healthy": {
        "pv_candidates": ["Potato___healthy", "Potato_healthy", "potato_healthy"],
        "pd_candidates": ["Potato leaf", "Potato_leaf", "potato leaf"],
        "description": "Potato - Healthy Leaf",
    },
    "raspberry_healthy": {
        "pv_candidates": ["Raspberry___healthy", "Raspberry_healthy", "raspberry_healthy"],
        "pd_candidates": ["Raspberry leaf", "Raspberry_leaf", "raspberry leaf"],
        "description": "Raspberry - Healthy Leaf",
    },
    "soybean_healthy": {
        "pv_candidates": ["Soybean___healthy", "Soybean_healthy", "soybean_healthy"],
        "pd_candidates": [
            "Soyabean leaf",
            "Soybean leaf",
            "Soyabean_leaf",
            "soyabean leaf"
        ],
        "description": "Soybean - Healthy Leaf",
    },
    "squash_powdery_mildew": {
        "pv_candidates": [
            "Squash___Powdery_mildew",
            "Squash_powdery_mildew",
            "squash_powdery_mildew"
        ],
        "pd_candidates": [
            "Squash Powdery mildew leaf",
            "Squash_Powdery_mildew_leaf",
            "squash powdery mildew leaf"
        ],
        "description": "Squash - Powdery Mildew (Podosphaera xanthii)",
    },
    "strawberry_healthy": {
        "pv_candidates": ["Strawberry___healthy", "Strawberry_healthy", "strawberry_healthy"],
        "pd_candidates": ["Strawberry leaf", "Strawberry_leaf", "strawberry leaf"],
        "description": "Strawberry - Healthy Leaf",
    },
    "tomato_bacterial_spot": {
        "pv_candidates": [
            "Tomato___Bacterial_spot",
            "Tomato_bacterial_spot",
            "tomato_bacterial_spot",
            "Tomato_Bacterial_spot"
        ],
        "pd_candidates": [
            "Tomato leaf bacterial spot",
            "Tomato_leaf_bacterial_spot",
            "Tomato leaf spot",
            "tomato leaf bacterial spot"
        ],
        "description": "Tomato - Bacterial Spot (Xanthomonas)",
    },
    "tomato_early_blight": {
        "pv_candidates": [
            "Tomato___Early_blight",
            "Tomato_early_blight",
            "tomato_early_blight",
            "Tomato_Early_blight"
        ],
        "pd_candidates": [
            "Tomato Early blight leaf",
            "Tomato_Early_blight_leaf",
            "Tomato leaf early blight",
            "Tomato_leaf_early_blight",
            "tomato early blight leaf"
        ],
        "description": "Tomato - Early Blight (Alternaria solani)",
    },
    "tomato_late_blight": {
        "pv_candidates": [
            "Tomato___Late_blight",
            "Tomato_late_blight",
            "tomato_late_blight",
            "Tomato_Late_blight"
        ],
        "pd_candidates": [
            "Tomato leaf late blight",
            "Tomato_leaf_late_blight",
            "tomato leaf late blight"
        ],
        "description": "Tomato - Late Blight (Phytophthora infestans)",
    },
    "tomato_leaf_mold": {
        "pv_candidates": [
            "Tomato___Leaf_Mold",
            "Tomato_leaf_mold",
            "tomato_leaf_mold",
            "Tomato_Leaf_Mold"
        ],
        "pd_candidates": [
            "Tomato mold leaf",
            "Tomato_mold_leaf",
            "Tomato_leaf_mold",
            "tomato mold leaf"
        ],
        "description": "Tomato - Leaf Mold (Passalora fulva)",
    },
    "tomato_septoria_leaf_spot": {
        "pv_candidates": [
            "Tomato___Septoria_leaf_spot",
            "Tomato_septoria_leaf_spot",
            "tomato_septoria_leaf_spot",
            "Tomato_Septoria_leaf_spot"
        ],
        "pd_candidates": [
            "Tomato Septoria leaf spot",
            "Tomato_Septoria_leaf_spot",
            "tomato septoria leaf spot"
        ],
        "description": "Tomato - Septoria Leaf Spot (Septoria lycopersici)",
    },
    "tomato_spider_mites": {
        "pv_candidates": [
            "Tomato_Spider_mites_Two_spotted_spider_mite",
            "Tomato_Spider_mites_Two_spotted_spider_mites",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato_spider_mites",
            "Tomato___Spider_mites"
        ],
        "pd_candidates": [
            "Tomato two spotted spider mites leaf",
            "Tomato_two_spotted_spider_mites_leaf",
            "tomato two spotted spider mites leaf"
        ],
        "description": "Tomato - Two-Spotted Spider Mites (Tetranychus urticae)",
    },
    "tomato_mosaic_virus": {
        "pv_candidates": [
            "Tomato__Tomato_mosaic_virus",
            "Tomato_Tomato_mosaic_virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato_mosaic_virus",
            "tomato_mosaic_virus"
        ],
        "pd_candidates": [
            "Tomato leaf mosaic virus",
            "Tomato_leaf_mosaic_virus",
            "tomato leaf mosaic virus"
        ],
        "description": "Tomato - Mosaic Virus",
    },
    "tomato_yellow_leaf_curl_virus": {
        "pv_candidates": [
            "Tomato__Tomato_YellowLeaf__Curl_Virus",
            "Tomato_Tomato_YellowLeaf__Curl_Virus",
            "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato_yellow_leaf_curl_virus",
            "tomato_yellow_leaf_curl_virus"
        ],
        "pd_candidates": [
            "Tomato leaf yellow virus",
            "Tomato_leaf_yellow_virus",
            "Tomato yellow leaf curl virus",
            "Tomato_yellow_leaf_curl_virus"
        ],
        "description": "Tomato - Yellow Leaf Curl Virus",
    },
    "tomato_healthy": {
        "pv_candidates": [
            "Tomato___healthy",
            "Tomato_healthy",
            "tomato_healthy"
        ],
        "pd_candidates": [
            "Tomato leaf",
            "Tomato_leaf",
            "tomato leaf"
        ],
        "description": "Tomato - Healthy Leaf",
    },
}


def normalize_string(name: str) -> str:
    """
    Normalize string for fuzzy/robust directory name matching.
    Collapses multiple consecutive underscores, hyphens, and whitespace into a single underscore.
    """
    s = name.lower()
    s = re.sub(r"[_\s\-]+", "_", s)
    s = re.sub(r"[(),]", "", s)
    s = re.sub(r"_leaf$", "", s)
    s = re.sub(r"_leaves$", "", s)
    return s.strip("_")


def get_canonical_class_list() -> List[str]:
    """Returns sorted list of canonical class names."""
    return sorted(list(CANONICAL_MAPPING.keys()))


def get_canonical_name_to_id() -> Dict[str, int]:
    """Returns dictionary mapping canonical class name to class index [0, C-1]."""
    classes = get_canonical_class_list()
    return {name: idx for idx, name in enumerate(classes)}


def get_id_to_canonical_name() -> Dict[int, str]:
    """Returns dictionary mapping class index to canonical class name."""
    classes = get_canonical_class_list()
    return {idx: name for idx, name in enumerate(classes)}


def resolve_folder_to_canonical(folder_name: str, domain: str = "auto") -> Optional[str]:
    """
    Given a folder name on disk, resolve it to its canonical class identifier.
    
    Args:
        folder_name: Directory name (e.g., 'Tomato___Late_blight' or 'Tomato leaf late blight')
        domain: 'pv' for PlantVillage, 'pd' for PlantDoc, or 'auto'
    
    Returns:
        canonical_name (str) or None if not an overlapping class.
    """
    norm_folder = normalize_string(folder_name)
    
    # Direct candidate matching
    for canonical_name, data in CANONICAL_MAPPING.items():
        candidates = []
        if domain in ("pv", "auto"):
            candidates.extend(data["pv_candidates"])
        if domain in ("pd", "auto"):
            candidates.extend(data["pd_candidates"])
            
        for cand in candidates:
            if folder_name == cand or norm_folder == normalize_string(cand):
                return canonical_name

    # Heuristic fallback matching for common dataset variants
    for canonical_name in CANONICAL_MAPPING.keys():
        norm_canonical = normalize_string(canonical_name)
        if norm_folder == norm_canonical:
            return canonical_name
            
    return None


def _get_target_folders(root_path: Path) -> List[Path]:
    """
    Discovers class folders under root_path, inspecting nested subdirectories
    if standard split folders ('train', 'test', 'val', 'validation') are present.
    """
    if not root_path.is_dir():
        return []
    direct_subdirs = [p for p in root_path.iterdir() if p.is_dir()]
    target_folders = []
    for d in direct_subdirs:
        if d.name.lower() in ("train", "test", "val", "validation"):
            target_folders.extend([p for p in d.iterdir() if p.is_dir()])
        else:
            target_folders.append(d)
    return target_folders


def build_dataset_mapping_dict(dataset_dir: str, domain: str = "auto") -> Dict[str, int]:
    """
    Scans a dataset directory and returns a dictionary mapping existing folder paths
    to their canonical integer class ID [0, C-1].
    Any folders not matching an overlapping canonical class are excluded.
    """
    expanded_path = Path(os.path.expandvars(os.path.expanduser(dataset_dir)))
    if not expanded_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dataset_dir} (resolved to {expanded_path})")
        
    canonical_to_id = get_canonical_name_to_id()
    folder_to_id: Dict[str, int] = {}
    
    subdirs = _get_target_folders(expanded_path)
    for subdir in subdirs:
        canonical = resolve_folder_to_canonical(subdir.name, domain=domain)
        if canonical is not None:
            folder_to_id[subdir.name] = canonical_to_id[canonical]
            
    return folder_to_id


def verify_dataset_alignment(
    pv_dir: str, 
    pd_dir: str, 
    verbose: bool = True
) -> Tuple[Dict[str, int], Dict[str, int], Set[str]]:
    """
    Inspects directories of both datasets, verifies overlapping classes,
    counts images per class, and reports domain alignment statistics.
    Directly supports nested split subfolders ('train', 'test', 'val') in PlantDoc.
    """
    pv_path = Path(os.path.expandvars(os.path.expanduser(pv_dir)))
    pd_path = Path(os.path.expandvars(os.path.expanduser(pd_dir)))
    
    canonical_to_id = get_canonical_name_to_id()
    id_to_canonical = get_id_to_canonical_name()
    
    pv_counts: Dict[str, int] = {k: 0 for k in canonical_to_id.keys()}
    pd_counts: Dict[str, int] = {k: 0 for k in canonical_to_id.keys()}
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Scan PlantVillage
    if pv_path.exists():
        direct_subdirs = [p for p in pv_path.iterdir() if p.is_dir()]
        target_folders = []
        for d in direct_subdirs:
            if d.name.lower() in ("train", "test", "val", "validation"):
                target_folders.extend([p for p in d.iterdir() if p.is_dir()])
            else:
                target_folders.append(d)

        for folder in target_folders:
            canon = resolve_folder_to_canonical(folder.name, domain="pv")
            if canon:
                num_imgs = len([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
                pv_counts[canon] += num_imgs
                    
    # Scan PlantDoc (directly inspecting nested train/test/val subfolders)
    if pd_path.exists():
        direct_subdirs = [p for p in pd_path.iterdir() if p.is_dir()]
        target_folders = []
        for d in direct_subdirs:
            if d.name.lower() in ("train", "test", "val", "validation"):
                target_folders.extend([p for p in d.iterdir() if p.is_dir()])
            else:
                target_folders.append(d)

        for folder in target_folders:
            canon = resolve_folder_to_canonical(folder.name, domain="pd")
            if canon:
                num_imgs = len([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
                pd_counts[canon] += num_imgs
                    
    # Classes with samples in both domains
    active_common_classes = {
        k for k in canonical_to_id.keys() 
        if pv_counts[k] > 0 and pd_counts[k] > 0
    }
    
    if verbose:
        print("\n" + "=" * 80)
        print(" DATASET ALIGNMENT AUDIT: PlantVillage (Lab) <--> PlantDoc (Field)")
        print(f" Source (PV): {pv_path.resolve()}")
        print(f" Target (PD): {pd_path.resolve()}")
        print("=" * 80)
        print(f"{'Class ID':<8} {'Canonical Class Name':<32} {'PV (Lab)':<12} {'PD (Field)':<12} {'Aligned?'}")
        print("-" * 80)
        for canon, idx in canonical_to_id.items():
            pv_n = pv_counts[canon]
            pd_n = pd_counts[canon]
            status = "MATCH" if (pv_n > 0 and pd_n > 0) else ("PV ONLY" if pv_n > 0 else ("PD ONLY" if pd_n > 0 else "MISSING"))
            print(f"{idx:<8} {canon:<32} {pv_n:<12} {pd_n:<12} {status}")
        print("-" * 80)
        print(f"Total Canonical Overlapping Classes Defined: {len(canonical_to_id)}")
        print(f"Classes with Images in Both Datasets:        {len(active_common_classes)}")
        print(f"Total PlantVillage Images in Matched Set:   {sum(pv_counts[c] for c in active_common_classes)}")
        print(f"Total PlantDoc Images in Matched Set:       {sum(pd_counts[c] for c in active_common_classes)}")
        print("=" * 80 + "\n")
        
    return pv_counts, pd_counts, active_common_classes


def export_mapping_json(output_path: str):
    """Exports canonical mapping definition as JSON for reference and documentation."""
    canonical_to_id = get_canonical_name_to_id()
    data = {
        "classes": [
            {
                "id": canonical_to_id[c],
                "canonical_name": c,
                "description": CANONICAL_MAPPING[c]["description"],
                "pv_candidates": CANONICAL_MAPPING[c]["pv_candidates"],
                "pd_candidates": CANONICAL_MAPPING[c]["pd_candidates"]
            }
            for c in get_canonical_class_list()
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] Exported mapping metadata to {output_path}")


def get_default_dataset_paths() -> Tuple[str, str]:
    """
    Resolves default paths for PlantVillage and PlantDoc tailored for VSC wICE cluster
    and local development environments.
    """
    vsc_data = os.environ.get("VSC_DATA", "")
    
    # 1. Check VSC_DATA environment variable
    if vsc_data:
        vsc_pv = os.path.join(vsc_data, "datasets", "PlantVillage")
        vsc_pd = os.path.join(vsc_data, "datasets", "PlantDoc")
        if Path(vsc_pv).exists() and Path(vsc_pd).exists():
            return vsc_pv, vsc_pd

    # 2. Check standard VSC user path (/data/leuven/393/vsc39394/datasets)
    vsc_std_pv = "/data/leuven/393/vsc39394/datasets/PlantVillage"
    vsc_std_pd = "/data/leuven/393/vsc39394/datasets/PlantDoc"
    if Path(vsc_std_pv).exists() and Path(vsc_std_pd).exists():
        return vsc_std_pv, vsc_std_pd

    # 3. Check local relative paths (data/PlantVillage, data/PlantDoc)
    for pv_cand in ("data/PlantVillage", "../datasets/PlantVillage"):
        if Path(pv_cand).exists():
            for pd_cand in ("data/PlantDoc", "data/plantdoc", "../datasets/PlantDoc"):
                if Path(pd_cand).exists():
                    return pv_cand, pd_cand

    # Default fallback matching VSC standard structure
    return (
        os.path.join(vsc_data, "datasets", "PlantVillage") if vsc_data else "data/PlantVillage",
        os.path.join(vsc_data, "datasets", "PlantDoc") if vsc_data else "data/PlantDoc"
    )


if __name__ == "__main__":
    default_pv, default_pd = get_default_dataset_paths()

    parser = argparse.ArgumentParser(description="Audit and export class mappings between PlantVillage and PlantDoc.")
    parser.add_argument("--pv_dir", type=str, default=default_pv, help="Path to PlantVillage root directory")
    parser.add_argument("--pd_dir", type=str, default=default_pd, help="Path to PlantDoc root directory")
    parser.add_argument("--export_json", type=str, default="", help="Path to export JSON mapping specification")
    args = parser.parse_args()

    # Expand any environment variables such as $VSC_DATA
    args.pv_dir = os.path.expandvars(os.path.expanduser(args.pv_dir))
    args.pd_dir = os.path.expandvars(os.path.expanduser(args.pd_dir))

    if args.export_json:
        export_mapping_json(args.export_json)
        
    pv_exists = Path(args.pv_dir).exists()
    pd_exists = Path(args.pd_dir).exists()

    if pv_exists and pd_exists:
        verify_dataset_alignment(args.pv_dir, args.pd_dir, verbose=True)
    elif pv_exists or pd_exists:
        print(f"[INFO] Running alignment audit on available dataset(s)...")
        verify_dataset_alignment(args.pv_dir, args.pd_dir, verbose=True)
    else:
        print(f"[INFO] Datasets not found at paths ('{args.pv_dir}', '{args.pd_dir}').")
        print(f"Loaded {len(CANONICAL_MAPPING)} canonical shared classes:")
        for idx, name in enumerate(get_canonical_class_list()):
            print(f"  [{idx:02d}] {name} ({CANONICAL_MAPPING[name]['description']})")
