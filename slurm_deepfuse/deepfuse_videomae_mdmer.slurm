#!/bin/bash
#SBATCH --job-name="df_videomae_mdmer"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=./NJZ_logs/deepfuse/deepfuse_videomae_mdmer.out
#SBATCH --error=./NJZ_logs/deepfuse/deepfuse_videomae_mdmer.err

module purge
module load gcc/11.3.1
module load miniforge/24.9.0
conda activate tsf-low

set -x
srun python -u runner.py \
    --model vemt --eeg_backbone cbramod --fusion naive --eeg_signal --eeg_full_unfreeze --eeg_full_signal --dense_video_clips --frame_interval 4 --clip_overlap_ratio 0.5 --fps_normalize --dense_chunk_size 6 --dense_cache_features --video_unfreeze_last_n_blocks 1 --clip_pool attn --gcn --gcn_video_per_clip --gcn_temporal_adj --gcn_clip_pe --deep_fuse --learning_rate 1e-4 --gcn_learning_rate 1e-4 --weight_decay 0.05 --epochs 100 --patience 25 --pretrained --checkpoint_dir ./checkpoints_clip --num_gpus 3 --batch_size 4 --grad_clip 0.5 \
    --vemt_video VideoMAE --dataset mdmer --csv_file ./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold1.csv \
    --experiment_name deepfuse_videomae-cbramod_mdmer_fold1
