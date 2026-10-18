#!/usr/bin/env bash
#SBATCH --job-name=sweep_loss_test
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"   # submit from tests/usability

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
set -u

echo "=== Node: $(hostname)"
echo "=== GPU:"
nvidia-smi -L
echo "=== Python: $(which python) ($(python --version 2>&1))"
echo "=== Started: $(date)"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 1. Sanity (light, ~minutes)
echo
echo "########## 01_sanity_each_loss.py ##########"
python -u 01_sanity_each_loss.py

# 2. Cycle-skipping scan (medium, ~10 min)
echo
echo "########## 02_cycle_skipping_scan.py ##########"
python -u 02_cycle_skipping_scan.py

# 3. Mini-FWI with multiple losses on a 2-layer model (heaviest)
echo
echo "########## 03_fwi_layer.py ##########"
python -u 03_fwi_layer.py

# 4. Regression check for the SoftDTW divergence-mode fix
echo
echo "########## 04_verify_softdtw_fix.py ##########"
python -u 04_verify_softdtw_fix.py

echo "=== Finished: $(date)"
