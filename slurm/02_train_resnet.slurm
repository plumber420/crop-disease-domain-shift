#!/usr/bin/env bash
# ==============================================================================
# 02_train_resnet.slurm
# SLURM Batch Job: ResNet-50 Training (Baseline vs. CutMix/MixUp Regularization)
# Target Architecture: KU Leuven Tier-2 wICE GPU Cluster (NVIDIA A100 / V100)
#
# Submit:
#   sbatch slurm/02_train_resnet.slurm
# ==============================================================================

#SBATCH --job-name=train_resnet50
#SBATCH --cluster=wICE
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=36G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -eo pipefail

echo "======================================================================"
echo " JOB STARTED: ${SLURM_JOB_NAME} (Job ID: ${SLURM_JOB_ID})"
echo " Node:        ${SLURM_NODELIST}"
echo " Submit Dir:  ${SLURM_SUBMIT_DIR}"
echo " Start Time:  $(date)"
echo "======================================================================"

# 1. Load Modules & Activate Environment
module purge
if module avail cluster/wICE 2>&1 | grep -q "cluster/wICE"; then
    module load cluster/wICE
fi
module load Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.1 2>/dev/null || module load Python CUDA 2>/dev/null || true

VENV_DIR="${VSC_DATA}/venvs/dl_plants_env"
if [ ! -d "${VENV_DIR}" ]; then
    echo "[ERROR] Virtualenv not found at ${VENV_DIR}. Run slurm/01_setup_env.sh first!"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# 2. Path Configurations
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
mkdir -p slurm/logs experiments/outputs

# Dataset Locations: checks local repo data/ folder first, then $VSC_DATA
PV_DIR="${PV_DIR:-${REPO_ROOT}/data/PlantVillage}"
if [ ! -d "${PV_DIR}" ] && [ -d "${VSC_DATA}/datasets/PlantVillage" ]; then
    PV_DIR="${VSC_DATA}/datasets/PlantVillage"
fi

PD_DIR="${PD_DIR:-${REPO_ROOT}/data/plantdoc}"
if [ ! -d "${PD_DIR}" ] && [ -d "${VSC_DATA}/datasets/PlantDoc" ]; then
    PD_DIR="${VSC_DATA}/datasets/PlantDoc"
fi

# Verify Datasets Exist
if [ ! -d "${PV_DIR}" ]; then
    echo "[ERROR] PlantVillage dataset directory not found at: ${PV_DIR}"
    echo "Please download and stage the datasets according to RUNBOOK.md"
    exit 1
fi

echo "[INFO] Using PlantVillage Source Dataset: ${PV_DIR}"
echo "[INFO] Using PlantDoc Target Dataset:     ${PD_DIR}"

# 3. GPU Sanity Check
nvidia-smi

# ------------------------------------------------------------------------------
# EXPERIMENT 1: ResNet-50 Baseline (Standard Data Augmentation)
# ------------------------------------------------------------------------------
echo "======================================================================"
echo " [1/2] RUNNING EXPERIMENT: ResNet-50 Baseline"
echo "======================================================================"

python3 src/train.py \
    --model_name resnet50 \
    --pretrained \
    --pv_dir "${PV_DIR}" \
    --pd_dir "${PD_DIR}" \
    --output_dir "${REPO_ROOT}/experiments/outputs" \
    --experiment_name "resnet50_baseline" \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-4 \
    --head_lr_mult 5.0 \
    --weight_decay 1e-2 \
    --num_workers 8 \
    --seed 42

# ------------------------------------------------------------------------------
# EXPERIMENT 2: ResNet-50 + CutMix / MixUp Regularization (Domain Shift Mitigation)
# ------------------------------------------------------------------------------
echo "======================================================================"
echo " [2/2] RUNNING EXPERIMENT: ResNet-50 with CutMix & MixUp Regularization"
echo "======================================================================"

python3 src/train.py \
    --model_name resnet50 \
    --pretrained \
    --pv_dir "${PV_DIR}" \
    --pd_dir "${PD_DIR}" \
    --output_dir "${REPO_ROOT}/experiments/outputs" \
    --experiment_name "resnet50_cutmix_mixup" \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-4 \
    --head_lr_mult 5.0 \
    --weight_decay 1e-2 \
    --use_cutmix_mixup \
    --mixup_alpha 0.8 \
    --cutmix_alpha 1.0 \
    --num_workers 8 \
    --seed 42

echo "======================================================================"
echo " ALL RESNET-50 EXPERIMENTS COMPLETED!"
echo " Checkpoints stored in: ${REPO_ROOT}/experiments/outputs/"
echo " End Time: $(date)"
echo "======================================================================"
