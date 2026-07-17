#!/bin/bash
cd "$(dirname "$0")"
for f in video_*.slurm; do echo "sbatch $f"; sbatch "$f"; done
