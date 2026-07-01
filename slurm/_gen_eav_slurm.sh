#!/usr/bin/env bash
# Generate SLURM sbatch files for unimodal baselines.
#   EAV fold0: video (AdaMAE/VideoMAE/ViViT/TSF/Swin) + eeg (CBraMod/REVE/LaBraM/ST-Transformer)
#   Emognition fold0 / MDMER fold1: TSF video-only.
# eeg-only runs on 1 GPU (batch 12); video-only on 3 GPUs x batch 4 = effective 12 (DDP).
set -eu
cd "$(dirname "$0")/.."
OUTDIR=slurm
EAV_CSV=./datasets/updated_fold_csv_files/EAV_fold_csv/EAV_dataset_updated_fold0.csv
EMO_CSV=./datasets/updated_fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv
MD_CSV=./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold1.csv
VID_COMMON="--set_video_only --fusion naive --dense_video_clips --dense_cache_features --video_unfreeze_last_n_blocks 1 --clip_pool attn"

port=20500
gen() {
  local DS="$1" CSV="$2" FOLD="$3" EXTRA="$4" NAME="$5" KIND="$6" BATCH="$7" LR="$8" ARGS="$9"
  local NGPU CPUS
  if [ "$KIND" = "video" ]; then NGPU=3; CPUS=24; else NGPU=1; CPUS=8; fi
  port=$((port+1))
  local f="${OUTDIR}/${DS}_${KIND}_${NAME}.slurm"
  cat > "$f" <<EOF
#!/bin/bash

#SBATCH --job-name="${DS}_${NAME}"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:${NGPU}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=01:00:00
#SBATCH --output=./NJZ_logs/${DS}/JZ_OUTPUT_${DS}_${NAME}.out
#SBATCH --error=./NJZ_logs/${DS}/JZ_ERROR_${DS}_${NAME}.out

module purge
module load gcc/11.3.1
module load miniforge/24.9.0

conda activate tsf-low

set -x

srun python -u runner.py \\
    ${ARGS} \\
    --dataset ${DS} --csv_file ${CSV} \\
    --learning_rate ${LR} --weight_decay 0.05 \\
    --epochs 100 --patience 25 \\
    --pretrained --fps_normalize --checkpoint_dir ./checkpoints_clip \\
    --num_gpus ${NGPU} --batch_size ${BATCH} ${EXTRA} \\
    --experiment_name ${DS}_${KIND}_${NAME}_${FOLD}_${LR}_0.05 \\
    --port ${port}
EOF
  echo "wrote $f"
}

# EAV fold0 — all backbones (video 3-GPU / eeg 1-GPU)
gen eav "$EAV_CSV" fold0 "" cbramod       eeg   12 1e-4 "--model vemt --eeg_backbone cbramod --set_eeg_only --fusion naive --eeg_full_unfreeze --eeg_full_signal"
gen eav "$EAV_CSV" fold0 "" reve          eeg   12 1e-4 "--model vemt --eeg_backbone reve --set_eeg_only --fusion naive --eeg_full_unfreeze --eeg_full_signal"
gen eav "$EAV_CSV" fold0 "" labram        eeg   12 1e-4 "--model labram --set_eeg_only --fusion naive --eeg_full_signal"
gen eav "$EAV_CSV" fold0 "" sttransformer eeg   12 1e-4 "--model sttransformer --set_eeg_only --fusion naive --eeg_full_signal"
gen eav "$EAV_CSV" fold0 "" adamae        video 4  1e-4 "--model vemt --vemt_video AdaMAE ${VID_COMMON} --frame_interval 4 --clip_overlap_ratio 0.5 --dense_chunk_size 6"
gen eav "$EAV_CSV" fold0 "" videomae      video 4  1e-4 "--model vemt --vemt_video VideoMAE ${VID_COMMON} --frame_interval 4 --clip_overlap_ratio 0.5 --dense_chunk_size 6"
gen eav "$EAV_CSV" fold0 "" vivit         video 4  2e-5 "--model vemt --vemt_video ViViT ${VID_COMMON} --frame_interval 4 --clip_overlap_ratio 0.5 --dense_chunk_size 6"
gen eav "$EAV_CSV" fold0 "" tsf           video 4  2e-5 "--model vemt --vemt_video TSF ${VID_COMMON} --frame_interval 4 --clip_overlap_ratio 0.5 --dense_chunk_size 6"
gen eav "$EAV_CSV" fold0 "" swin          video 4  2e-5 "--model vemt --vemt_video Swin ${VID_COMMON} --frame_interval 4 --clip_overlap_ratio 0.5 --dense_chunk_size 3"

# Emognition fold0 / MDMER fold1 — TSF video-only (MDMER adds grad_clip 0.5)
gen emognition "$EMO_CSV" fold0 ""               tsf video 4 2e-5 "--model vemt --vemt_video TSF ${VID_COMMON} --frame_interval 4 --clip_overlap_ratio 0.5 --dense_chunk_size 6"
gen mdmer      "$MD_CSV"  fold1 "--grad_clip 0.5" tsf video 4 2e-5 "--model vemt --vemt_video TSF ${VID_COMMON} --frame_interval 4 --clip_overlap_ratio 0.5 --dense_chunk_size 6"

echo "done — $(ls ${OUTDIR}/*.slurm 2>/dev/null | wc -l) sbatch files"
