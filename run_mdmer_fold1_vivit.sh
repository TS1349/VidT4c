#!/usr/bin/env bash
# MDMER fold1 — ViViT (fi=2, mean → attn) on GPU 0,1,2,3
# Same lr/wd as the AdaMAE fold1 runs (1e-4 / 0.05, patience 15).
# mean and attn share the dense_cache_features cache (cache is per
# backbone+fi+unfreeze+fpsnorm+ov, NOT per pool), so cache is cleared
# AFTER both runs finish.
#
# Run from /SSD4/jh/INRIA. Run in parallel with run_mdmer_fold1_swin.sh
# on GPUs 4,5,6,7.

set -euo pipefail

cd /SSD4/jh/INRIA

GPUS=0,1,2,3
DATASET=mdmer
CSV=./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold1.csv
CSV_BASE=MDMER_dataset_updated_fold1
LR=1e-4
WD=0.05
FI=2
UNFREEZE=1
BACKBONE=ViViT

ts() { date '+%F %T'; }

run() {
  local NAME="$1"; shift
  echo "[$(ts)] >>> START $NAME"
  CUDA_VISIBLE_DEVICES=$GPUS python runner.py "$@"
  echo "[$(ts)] <<< END   $NAME"
}

clear_cache() {
  local PATH_TO_CLEAR=".feature_cache/${DATASET}/${CSV_BASE}/${BACKBONE}/fi${FI}_fpsnorm_unfreeze${UNFREEZE}"
  if [ -d "$PATH_TO_CLEAR" ]; then
    local SIZE=$(du -sh "$PATH_TO_CLEAR" 2>/dev/null | awk '{print $1}')
    echo "[$(ts)] --- removing cache: $PATH_TO_CLEAR (size: $SIZE)"
    rm -rf "$PATH_TO_CLEAR"
  else
    echo "[$(ts)] --- no cache to remove at $PATH_TO_CLEAR"
  fi
}


run "vivit_fi2_mean" \
  --model vemt --vemt_video $BACKBONE --set_video_only --fusion naive \
  --dataset $DATASET --csv_file $CSV \
  --learning_rate $LR --weight_decay $WD --epochs 100 --num_gpus 4 --batch_size 3 \
  --pretrained --patience 15 --frame_interval $FI --fps_normalize \
  --dense_video_clips --dense_chunk_size 6 --dense_cache_features \
  --video_unfreeze_last_n_blocks $UNFREEZE --clip_pool mean \
  --checkpoint_dir ./checkpoints_clip \
  --experiment_name dense_fi2_1block-mean_vivit_mdmer_fold1_${LR}_${WD} \
  --port 20120

run "vivit_fi2_attn" \
  --model vemt --vemt_video $BACKBONE --set_video_only --fusion naive \
  --dataset $DATASET --csv_file $CSV \
  --learning_rate $LR --weight_decay $WD --epochs 100 --num_gpus 4 --batch_size 3 \
  --pretrained --patience 15 --frame_interval $FI --fps_normalize \
  --dense_video_clips --dense_chunk_size 6 --dense_cache_features \
  --video_unfreeze_last_n_blocks $UNFREEZE --clip_pool attn \
  --checkpoint_dir ./checkpoints_clip \
  --experiment_name dense_fi2_1block-attn_vivit_mdmer_fold1_${LR}_${WD} \
  --port 20121

clear_cache

echo "[$(ts)] >>> ViViT mdmer fold1 chain DONE"
