#!/usr/bin/env bash
# Submit all EAV fold0 unimodal baselines (one SLURM job per model).
# Run from repo root on the cluster:  bash slurm/submit_all_eav.sh
# Optional: pass a subset, e.g.  bash slurm/submit_all_eav.sh eav_video_swin eav_eeg_cbramod
set -eu
cd "$(dirname "$0")/.."
mkdir -p NJZ_logs/eav   # #SBATCH --output dir must exist before sbatch

if [ "$#" -gt 0 ]; then
  FILES=()
  for n in "$@"; do FILES+=("slurm/${n%.slurm}.slurm"); done
else
  FILES=(slurm/eav_*.slurm)
fi

for f in "${FILES[@]}"; do
  echo ">>> sbatch $f"
  sbatch "$f"
done
echo "submitted ${#FILES[@]} job(s)."
