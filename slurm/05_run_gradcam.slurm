#!/usr/bin/env bash
# ==============================================================================
# 05_run_gradcam.slurm
# SLURM Batch Job: Grad-CAM Explainability & Shortcut Learning Visualization
# Supports:
#   - ResNet-50 (CNN: layer4[-1])
#   - Swin-T (Vision Transformer: features[-1] via spatial reshape transform)
#
# Generates:
#   - Side-by-side explainability heatmaps for PlantVillage (Lab) vs PlantDoc (Field)
#   - Center vs Border activation energy diagnostic (Shortcut quantification)
#   - Focal concentration ratio & Activation Spatial Entropy
#
# Submit:
#   sbatch slurm/05_run_gradcam.slurm
# ==============================================================================

#SBATCH --job-name=gradcam_explain
#SBATCH --cluster=wICE
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -eo pipefail

echo "======================================================================"
echo " GRAD-CAM EXPLAINABILITY JOB: ID ${SLURM_JOB_ID}"
echo " Start Time: $(date)"
echo "======================================================================"

module purge
if module avail cluster/wICE 2>&1 | grep -q "cluster/wICE"; then
    module load cluster/wICE
fi
module load Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.1 2>/dev/null || module load Python CUDA 2>/dev/null || true

VENV_DIR="${VSC_DATA}/venvs/dl_plants_env"
source "${VENV_DIR}/bin/activate"

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
mkdir -p slurm/logs experiments/gradcam_analysis

PV_DIR="${VSC_DATA}/datasets/PlantVillage"
PD_DIR="${VSC_DATA}/datasets/PlantDoc"

run_cam_for_model () {
    local MODEL_NAME=$1
    local EXP_NAME=$2
    local CKPT="${REPO_ROOT}/experiments/outputs/${EXP_NAME}/checkpoints/best_checkpoint.pt"
    local OUT_DIR="${REPO_ROOT}/experiments/gradcam_analysis/${EXP_NAME}"

    if [ -f "${CKPT}" ]; then
        echo "----------------------------------------------------------------------"
        echo "[INFO] Generating Grad-CAM for ${EXP_NAME} (${MODEL_NAME})..."
        echo "----------------------------------------------------------------------"
        python3 src/explainability.py \
            --checkpoint "${CKPT}" \
            --model_name "${MODEL_NAME}" \
            --pv_dir "${PV_DIR}" \
            --pd_dir "${PD_DIR}" \
            --output_dir "${OUT_DIR}" \
            --num_samples 12
    else
        echo "[WARN] Checkpoint not found: ${CKPT}. Skipping ${EXP_NAME}."
    fi
}

# 1. ResNet-50 Baseline
run_cam_for_model "resnet50" "resnet50_baseline"

# 2. ResNet-50 with CutMix/MixUp Regularization
run_cam_for_model "resnet50" "resnet50_cutmix_mixup"

# 3. Swin-T Baseline
run_cam_for_model "swin_t" "swin_t_baseline"

# 4. Swin-T with CutMix/MixUp Regularization
run_cam_for_model "swin_t" "swin_t_cutmix_mixup"

echo "======================================================================"
echo " GRAD-CAM VISUALIZATION COMPLETED FOR ALL ARCHITECTURES!"
echo " PNG heatmaps and overlays saved in: ${REPO_ROOT}/experiments/gradcam_analysis/"
echo " End Time: $(date)"
echo "======================================================================"
