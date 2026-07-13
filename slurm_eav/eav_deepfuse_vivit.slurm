#!/bin/bash
#SBATCH --job-name="eav_deepfuse_vivit"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=./NJZ_logs/eav/eav_deepfuse_vivit.out
#SBATCH --error=./NJZ_logs/eav/eav_deepfuse_vivit.err

module purge
module load gcc/11.3.1
module load miniforge/24.9.0
conda activate tsf-low

set -x
srun python -u runner.py \
    --model vemt --vemt_video AdaMAE --eeg_backbone cbramod --fusion naive --eeg_signal --eeg_full_unfreeze --eeg_full_signal --dense_video_clips --dense_cache_features --video_unfreeze_last_n_blocks 1 --clip_pool attn --frame_interval 4 --clip_overlap_ratio 0.5 --dense_chunk_size 6 --fps_normalize --gcn --gcn_video_per_clip --gcn_temporal_adj --gcn_clip_pe --learning_rate 1e-4 --gcn_learning_rate 1e-4 --weight_decay 0.05 --epochs 100 --patience 25 --pretrained --checkpoint_dir ./checkpoints_clip --num_gpus 3 --batch_size 4 --dataset eav --csv_file ./datasets/updated_fold_csv_files/EAV_fold_csv/EAV_dataset_updated_fold0.csv \
    --vemt_video ViViT --deep_fuse \
    --experiment_name eav_deepfuse_vivit_adamae-cbramod_fold0
