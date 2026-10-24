#!/bin/bash
# Viking 2-D frequency-selection FWI, Steps 1-5 of
# docs/tasks/datasets/viking/README.md in one job: build the trace index and
# the CSG plan, extract the six coefficient shards, run the inversion.
# On a resubmit, Steps 1-4 are skipped when their output exists and a finished
# inversion is not repeated; an unfinished one restarts from the first band.
#
# Before submitting:
#   * Step 0 of the doc done: $VIKING_HOME/raw holds seismic.segy and
#     viking_initial_vp_12.5m.npy;
#   * the environment with sweepx active in this shell (sbatch passes your
#     environment on to the job);
#   * VIKING_HOME and SWEEPX (a checkout of this repo) exported.
#
#   export VIKING_HOME=$HOME/viking SWEEPX=$HOME/sweepx
#   sbatch $SWEEPX/examples/tasks/sbatch/viking_freqsel.sbatch
#
# The log goes to viking_freqsel-<jobid>.out in the directory you submit
# from. Add your cluster's partition / account / GPU-type options on the
# sbatch command line or below.
#SBATCH --job-name=viking_freqsel
#SBATCH --output=viking_freqsel-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail
: "${VIKING_HOME:?export VIKING_HOME before sbatch}"
: "${SWEEPX:?export SWEEPX (a checkout of the sweepx repo) before sbatch}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

CFG="$SWEEPX/examples/tasks/field/viking/pipeline_config.yaml"
YAML="$SWEEPX/examples/tasks/field/viking/viking_freqsel_allteeth.yaml"
RUN="$VIKING_HOME/run"
FWI="$RUN/freqsel/viking_freqsel_allteeth"
mkdir -p "$RUN/shards"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import sweepx; print('sweepx', sweepx.__version__, sweepx.__file__)"

[ -f "$RUN/viking_index.npz" ]    || sweep-tasks build-index --config "$CFG"
[ -f "$RUN/viking_csg_plan.npz" ] || sweep-tasks build-plan  --config "$CFG"

# A shard is skipped when its file exists: after changing a band's row below,
# delete that shard so it is cut again.
#  lo hi  dh     dt      n_p    k_lo k_hi   (must match the stages in $YAML)
while read -r lo hi dh dt n_p klo khi; do
  out="$RUN/shards/coeff_${lo}_${hi}hz.npz"
  [ -f "$out" ] && continue
  # write under a temporary name, so a job killed mid-write leaves no
  # complete-looking shard behind
  sweep-tasks extract-coeff --plan "$RUN/viking_csg_plan.npz" \
      --dh-m "$dh" --dt "$dt" --min-fold 1 --node-stride 1 \
      --n-p "$n_p" --k-lo "$klo" --k-hi "$khi" -o "${out%.npz}.part.npz"
  mv "${out%.npz}.part.npz" "$out"
done <<'BANDS'
3 7  25    0.002  125000 750 1750
3 8  25    0.002  100000 600 1600
3 10 25    0.002  71450  429 1429
3 12 25    0.002  55584  334 1334
3 14 18.75 0.0015 60620  273 1273
3 16 18.75 0.0015 51292  231 1231
BANDS

if grep -q '"state": "success"' "$FWI/status.json" 2>/dev/null; then
  echo "inversion already finished: $FWI"
  exit 0
fi
# an unfinished run restarts from the first band; keep the old attempt apart
# instead of appending to its logs
if [ -d "$FWI" ]; then
  mv "$FWI" "$FWI.unfinished-$(date +%Y%m%d-%H%M%S)"
fi
sweep-tasks run "$YAML"
