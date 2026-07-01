#!/bin/bash

#SBATCH --job-name="emognition_tsf"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=01:00:00
#SBATCH --output=./NJZ_logs/emognition/JZ_OUTPUT_emognition_tsf.out
#SBATCH --error=./NJZ_logs/emognition/JZ_ERROR_emognition_tsf.out

module purge
module load gcc/11.3.1
module load miniforge/24.9.0

conda activate tsf-low

set -x

srun python -u runner.py \
    --model vemt --vemt_video TSF --set_video_only --fusion naive --dense_video_clips --dense_cache_features --video_unfreeze_last_n_blocks 1 --clip_pool attn --frame_interval 4 --clip_overlap_ratio 0.5 --dense_chunk_size 6 \
    --dataset emognition --csv_file ./datasets/updated_fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv \
    --learning_rate 2e-5 --weight_decay 0.05 \
    --epochs 100 --patience 25 \
    --pretrained --fps_normalize --checkpoint_dir ./checkpoints_clip \
    --num_gpus 3 --batch_size 4  \
    --experiment_name emognition_video_tsf_fold0_2e-5_0.05 \
    --port 20510
