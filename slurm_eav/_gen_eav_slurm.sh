#!/bin/bash
# EAV fusion (VEMT AdaMAE+CBraMod, gcn-clip-temp-pe) slurm.
# EAV: video~90% >> eeg~40%, but eeg is full-trained + gate inits 0.269 (eeg-favor)
# so plain fusion collapses to ~55%. Two remedies:
#   1) fixed gate strongly toward video (fusion_gate_fixed = video share 0.6..0.95)
#   2) deep_fuse: stage-1 video/eeg branch logits + joint, 0.25/0.25/0.5 deep-sup
#      (lets the strong video branch contribute directly, bypassing the gate skew)
# Jean Zay H100, 3 GPU x batch 4 = 12.  Run: bash slurm_eav/_gen_eav_slurm.sh ; bash slurm_eav/submit_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p ../NJZ_logs/eav
EAV_CSV=./datasets/updated_fold_csv_files/EAV_fold_csv/EAV_dataset_updated_fold0.csv

BASE_FLAGS="--model vemt --vemt_video AdaMAE --eeg_backbone cbramod --fusion naive --eeg_signal \
--eeg_full_unfreeze --eeg_full_signal \
--dense_video_clips --dense_cache_features --video_unfreeze_last_n_blocks 1 --clip_pool attn \
--frame_interval 4 --clip_overlap_ratio 0.5 --dense_chunk_size 6 --fps_normalize \
--gcn --gcn_video_per_clip --gcn_temporal_adj --gcn_clip_pe \
--learning_rate 1e-4 --gcn_learning_rate 1e-4 --weight_decay 0.05 \
--epochs 100 --patience 25 --pretrained --checkpoint_dir ./checkpoints_clip \
--num_gpus 3 --batch_size 4 --dataset eav --csv_file ${EAV_CSV}"

gen() {  # $1=name $2=method_flags
  local name=$1 mflags=$2
  cat > "eav_${name}.slurm" <<SL
#!/bin/bash
#SBATCH --job-name="eav_${name}"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=./NJZ_logs/eav/eav_${name}.out
#SBATCH --error=./NJZ_logs/eav/eav_${name}.err

module purge
module load gcc/11.3.1
module load miniforge/24.9.0
conda activate tsf-low

set -x
srun python -u runner.py \\
    ${BASE_FLAGS} \\
    ${mflags} \\
    --experiment_name eav_${name}_adamae-cbramod_fold0
SL
  echo "eav_${name}.slurm"
}

# 1) video-favor gate sweep (video share p)
for p in 0.6 0.7 0.8 0.9 0.95; do
  ptag=$(echo "$p" | tr -d '.')
  gen "gate${ptag}" "--fusion_gate_fixed $p"
done
# 2) deep_fuse (principled fix for the eeg-skew: strong video branch contributes directly)
gen "deepfuse" "--deep_fuse"

cat > submit_all.sh <<'SUB'
#!/bin/bash
cd "$(dirname "$0")"
for f in eav_*.slurm; do echo "sbatch $f"; sbatch "$f"; done
SUB
chmod +x submit_all.sh
echo "generated $(ls eav_*.slurm | wc -l) slurm files + submit_all.sh"
