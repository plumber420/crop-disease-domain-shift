#!/usr/bin/env bash
# ==============================================================================
# 04_evaluate_all.slurm
# SLURM Batch Job: Multi-Model Cross-Domain Evaluation & Generalization Gap Audit
#
# Evaluates:
#   1. ResNet-50 Baseline
#   2. ResNet-50 + CutMix/MixUp
#   3. Swin-T Baseline
#   4. Swin-T + CutMix/MixUp
# On:
#   - PlantVillage (In-Domain Laboratory Test Set)
#   - PlantDoc (Out-of-Domain Field Conditions)
#
# Submit:
#   sbatch slurm/04_evaluate_all.slurm
# ==============================================================================

#SBATCH --job-name=eval_domain_shift
#SBATCH --cluster=wICE
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -eo pipefail

echo "======================================================================"
echo " CROSS-DOMAIN EVALUATION STARTED: Job ID ${SLURM_JOB_ID}"
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
mkdir -p slurm/logs experiments/evaluation_results

PV_DIR="${VSC_DATA}/datasets/PlantVillage"
PD_DIR="${VSC_DATA}/datasets/PlantDoc"

check_and_eval () {
    local MODEL_NAME=$1
    local EXP_NAME=$2
    local CKPT="${REPO_ROOT}/experiments/outputs/${EXP_NAME}/checkpoints/best_checkpoint.pt"
    local OUT_DIR="${REPO_ROOT}/experiments/evaluation_results/${EXP_NAME}"

    if [ -f "${CKPT}" ]; then
        echo "----------------------------------------------------------------------"
        echo "[EVALUATING] ${EXP_NAME} (${MODEL_NAME})"
        echo "Checkpoint:  ${CKPT}"
        echo "----------------------------------------------------------------------"
        python3 src/evaluate.py \
            --checkpoint "${CKPT}" \
            --model_name "${MODEL_NAME}" \
            --pv_dir "${PV_DIR}" \
            --pd_dir "${PD_DIR}" \
            --output_dir "${OUT_DIR}" \
            --batch_size 32 \
            --num_workers 4 \
            --seed 42
    else
        echo "[WARN] Checkpoint not found: ${CKPT}. Skipping ${EXP_NAME}."
    fi
}

# 1. ResNet-50 Baseline
check_and_eval "resnet50" "resnet50_baseline"

# 2. ResNet-50 CutMix/MixUp
check_and_eval "resnet50" "resnet50_cutmix_mixup"

# 3. Swin-T Baseline
check_and_eval "swin_t" "swin_t_baseline"

# 4. Swin-T CutMix/MixUp
check_and_eval "swin_t" "swin_t_cutmix_mixup"

# 5. Aggregate and print comparative leaderboard across all evaluated models
python3 -c "
import json
from pathlib import Path

eval_dir = Path('${REPO_ROOT}/experiments/evaluation_results')
reports = list(eval_dir.glob('*/domain_shift_report.json'))

if not reports:
    print('[INFO] No evaluation reports found yet.')
    exit(0)

print('\n' + '=' * 105)
print(' OVERALL DOMAIN SHIFT BENCHMARK LEADERBOARD (KU Leuven HPC wICE)')
print('=' * 105)
print(f'{\"Experiment\":<28} {\"Architecture\":<14} {\"PV Acc (%)\":<12} {\"PD Acc (%)\":<12} {\"PV F1 (%)\":<12} {\"PD F1 (%)\":<12} {\"F1 Drop (%)\":<12} {\"Retention (%)\":<12}')
print('-' * 105)

for rep_p in sorted(reports):
    with open(rep_p) as f:
        data = json.load(f)
    exp = rep_p.parent.name
    arch = data['model_architecture']
    pv_acc = data['source_domain']['accuracy'] * 100
    pd_acc = data['target_domain']['accuracy'] * 100
    pv_f1 = data['source_domain']['macro_f1'] * 100
    pd_f1 = data['target_domain']['macro_f1'] * 100
    f1_drop = (data['domain_shift_analysis']['delta_macro_f1']) * 100
    retention = data['domain_shift_analysis']['f1_retention_percentage']
    
    print(f'{exp:<28} {arch:<14} {pv_acc:<12.2f} {pd_acc:<12.2f} {pv_f1:<12.2f} {pd_f1:<12.2f} {-f1_drop:<12.2f} {retention:<12.2f}')

print('=' * 105 + '\n')
"

echo "======================================================================"
echo " EVALUATION PIPELINE COMPLETED!"
echo " Reports and figures available in: ${REPO_ROOT}/experiments/evaluation_results/"
echo " End Time: $(date)"
echo "======================================================================"
