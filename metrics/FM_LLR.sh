#!/bin/bash
#SBATCH --job-name=LR_L_metrics_multi
#SBATCH --account=project_2012243
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:15:00
#SBATCH --output=/scratch/project_2012243/image-restoration/Final_FM4x/latent/metrics_latent_LR%j.out

set -euo pipefail

REPO=/projappl/project_2012243/image-restoration/flow_matching_NAFNet/metrics

TAR=/scratch/project_2012243/image-restoration/datasets/DIV2K.tar.gz

echo "Unpacking dataset to NVME..."
mkdir -p "$LOCAL_SCRATCH/DIV2K"
tar -xzf "$TAR" -C "$LOCAL_SCRATCH/DIV2K"

FOUND_DIR=$(find "$LOCAL_SCRATCH/DIV2K" -type d -name "DIV2K_train_HR" -print -quit)

if [ -z "$FOUND_DIR" ]; then
  echo "ERROR: DIV2K_train_HR not found"
  ls -R "$LOCAL_SCRATCH" | sed -n '1,200p'
  exit 1
fi

DATA_ROOT=$(dirname "$FOUND_DIR")
echo "DATA_ROOT = $DATA_ROOT"

# Ground truth HR images
GT_DIR="$DATA_ROOT/DIV2K_valid_HR"

BASE_DIR="/scratch/project_2012243/image-restoration/Final_FM4x/latent/latent_LR_20260525_0420"

# Output directory (optional, same as base_dir if not needed)
OUT_DIR="$BASE_DIR/results"
mkdir -p "$OUT_DIR"

echo "BASE_DIR = $BASE_DIR"
echo "GT_DIR   = $GT_DIR"
echo "OUT_DIR  = $OUT_DIR"

# =========================
# ENV SETUP
# =========================
module purge
module load tykky

export PATH="/scratch/project_2012243/image-restoration/m_env/bin:$PATH"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# =========================
# RUN
# =========================
cd "$REPO"

echo "Running metrics_multi_updated1.py..."

srun python -u metrics_multi_updated.py \
  --base_dir "$BASE_DIR" \
  --gt_dir "$GT_DIR" \
  --out_dir "$OUT_DIR"

echo "Done. Results saved to $OUT_DIR"