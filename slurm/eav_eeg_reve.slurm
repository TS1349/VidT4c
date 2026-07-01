#!/bin/bash

#SBATCH --job-name="eav_reve"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=01:00:00
#SBATCH --output=./NJZ_logs/eav/JZ_OUTPUT_eav_reve.out
#SBATCH --error=./NJZ_logs/eav/JZ_ERROR_eav_reve.out

module purge
module load gcc/11.3.1
module load miniforge/24.9.0

conda activate tsf-low

set -x

srun python -u runner.py \
    --model vemt --eeg_backbone reve --set_eeg_only --fusion naive --eeg_full_unfreeze --eeg_full_signal \
    --dataset eav --csv_file ./datasets/updated_fold_csv_files/EAV_fold_csv/EAV_dataset_updated_fold0.csv \
    --learning_rate 1e-4 --weight_decay 0.05 \
    --epochs 100 --patience 25 \
    --pretrained --fps_normalize --checkpoint_dir ./checkpoints_clip \
    --num_gpus 1 --batch_size 12  \
    --experiment_name eav_eeg_reve_fold0_1e-4_0.05 \
    --port 20502
