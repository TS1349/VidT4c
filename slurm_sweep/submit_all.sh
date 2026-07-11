#!/bin/bash
cd "$(dirname "$0")"
for f in sweep_*.slurm; do echo "sbatch $f"; sbatch "$f"; done
