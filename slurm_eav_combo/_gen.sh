#!/bin/bash
# EAV fusion repair for the MAE backbones (videomae/adamae + cbramod), where base
# fusion collapses (~0.53) far below the video-only accuracy (~0.90).
# Two recipes: combo (gcn_dropout + weight decay + video-favor gate) and warmup
# (direct video-branch loss decayed over early epochs). 2 backbones x 2 = 4 jobs.
# Run: bash slurm_eav_combo/_gen.sh ; bash slurm_eav_combo/submit_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p ../NJZ_logs/eav_combo
CSV=./datasets/updated_fold_csv_files/EAV_fold_csv/EAV_dataset_updated_fold0.csv

BASE_FLAGS="--model vemt --eeg_backbone cbramod --fusion naive --eeg_signal \
--eeg_full_unfreeze --eeg_full_signal \
--dense_video_clips --frame_interval 4 --clip_overlap_ratio 0.5 --fps_normalize \
--dense_chunk_size 6 --dense_cache_features --video_unfreeze_last_n_blocks 1 --clip_pool attn \
--gcn --gcn_video_per_clip --gcn_temporal_adj --gcn_clip_pe \
--learning_rate 1e-4 --gcn_learning_rate 1e-4 \
--epochs 100 --patience 25 --pretrained --checkpoint_dir ./checkpoints_clip \
--num_gpus 3 --batch_size 4 --dataset eav --csv_file ${CSV}"

gen() {  # $1=video $2=recipe $3=recipe_flags
  local vid=$1 recipe=$2 flags=$3 vtag
  vtag=$(echo "$vid" | tr 'A-Z' 'a-z')
  cat > "eav_${recipe}_${vtag}.slurm" <<SL
#!/bin/bash
#SBATCH --job-name="eav_${recipe}_${vtag}"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=./NJZ_logs/eav_combo/eav_${recipe}_${vtag}.out
#SBATCH --error=./NJZ_logs/eav_combo/eav_${recipe}_${vtag}.err

module purge
module load gcc/11.3.1
module load miniforge/24.9.0
conda activate tsf-low

set -x
srun python -u runner.py \\
    ${BASE_FLAGS} ${flags} \\
    --vemt_video ${vid} \\
    --experiment_name eav_${recipe}_${vtag}-cbramod_fold0
SL
  echo "eav_${recipe}_${vtag}.slurm"
}

COMBO="--gcn_dropout 0.3 --weight_decay 0.1 --gcn_weight_decay 0.1 --head_weight_decay 0.1 --fusion_gate_fixed 0.6"
WARMUP="--weight_decay 0.05 --video_aux_warmup 15 --video_aux_w 1.0"

for vid in VideoMAE AdaMAE; do
  gen "$vid" combo  "$COMBO"
  gen "$vid" warmup "$WARMUP"
done

cat > submit_all.sh <<'SUB'
#!/bin/bash
cd "$(dirname "$0")"
for f in eav_*.slurm; do echo "sbatch $f"; sbatch "$f"; done
SUB
chmod +x submit_all.sh
echo "generated $(ls eav_*.slurm | wc -l) slurm files + submit_all.sh"
