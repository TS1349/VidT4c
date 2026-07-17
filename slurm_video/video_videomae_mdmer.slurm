#!/bin/bash
#SBATCH --job-name="vid_videomae_mdmer"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=10:00:00
#SBATCH --output=./NJZ_logs/video/video_videomae_mdmer.out
#SBATCH --error=./NJZ_logs/video/video_videomae_mdmer.err

module purge
module load gcc/11.3.1
module load miniforge/24.9.0
conda activate tsf-low

set -x
srun python -u runner.py \
    --model vemt --vemt_video VideoMAE --set_video_only --fusion naive \
    --dense_video_clips --frame_interval 4 --clip_overlap_ratio 0.5 --clip_pool mean \
    --dense_chunk_size 6 --dense_cache_features --video_unfreeze_last_n_blocks 1 --fps_normalize \
    --learning_rate 1e-4 --weight_decay 0.05 \
    --epochs 100 --patience 15 --pretrained --checkpoint_dir ./checkpoints_clip \
    --num_gpus 3 --batch_size 4 --dataset mdmer --csv_file ./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold1.csv \
    --experiment_name dense_fi4-overlap_1block-mean_videomae_mdmer_fold1_1e-4_0.05
