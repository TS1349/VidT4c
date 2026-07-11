#!/bin/bash
# Generate the final fusion-gate (epsilon) sweep + AdaMAE self-loop slurm files
# (Jean Zay H100, 80GB/GPU; 3 GPUs x batch 4 = effective batch 12).
#
# Contents:
#   1) epsilon sweep -- base all-clips GCN fusion with a FIXED fusion gate
#      w=epsilon (video weight), over {VideoMAE, AdaMAE} x {emognition, mdmer}.
#      Finds the optimal constant modality weighting per (backbone, dataset).
#   2) AdaMAE self-loop -- joint-graph self-loop (node self-identity in gcn2),
#      rest identical to base (learnable gate ~0.269), AdaMAE x both datasets.
#      (VideoMAE self-loop is run on our local machine, not here.)
#
# Run:  bash slurm_sweep/_gen_sweep_slurm.sh   then   bash slurm_sweep/submit_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p ../NJZ_logs/sweep
EMO_CSV=./datasets/updated_fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv
MD_CSV=./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold1.csv

# All-clips base fusion setup. Only --vemt_video and the method flags vary.
BASE_FLAGS="--fusion naive --eeg_signal --eeg_backbone cbramod \
--eeg_full_unfreeze --eeg_full_signal \
--dense_video_clips --frame_interval 4 --clip_overlap_ratio 0.5 --fps_normalize \
--dense_chunk_size 6 --dense_cache_features --video_unfreeze_last_n_blocks 1 --clip_pool attn \
--gcn --gcn_video_per_clip --gcn_temporal_adj --gcn_clip_pe \
--learning_rate 1e-4 --gcn_learning_rate 1e-4 --weight_decay 0.05 \
--epochs 100 --patience 40 --pretrained --checkpoint_dir ./checkpoints_clip \
--num_gpus 3 --batch_size 4"

EPS_LIST="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

gen() {  # $1=name $2=video_backbone $3=dataset $4=method_flags
  local name=$1 vid=$2 ds=$3 mflags=$4
  local csv gc fold
  if [ "$ds" = emognition ]; then csv=$EMO_CSV; gc=""; fold=fold0; else csv=$MD_CSV; gc="--grad_clip 0.5"; fold=fold1; fi
  cat > "sweep_${name}_${ds}.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name="sweep_${name}_${ds}"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=./NJZ_logs/sweep/sweep_${name}_${ds}.out
#SBATCH --error=./NJZ_logs/sweep/sweep_${name}_${ds}.err

module purge
module load gcc/11.3.1
module load miniforge/24.9.0
conda activate tsf-low

set -x
srun python -u runner.py \\
    --model vemt --vemt_video ${vid} ${BASE_FLAGS} ${gc} \\
    --dataset ${ds} --csv_file ${csv} \\
    ${mflags} \\
    --experiment_name sweep_${name}_$(echo ${vid} | tr 'A-Z' 'a-z')-cbramod_${ds}_${fold}
EOF
  echo "sweep_${name}_${ds}.slurm"
}

# 1) epsilon (fixed fusion gate) sweep — both backbones, both datasets.
for vid in VideoMAE AdaMAE; do
  vtag=$(echo "$vid" | tr 'A-Z' 'a-z')
  for eps in $EPS_LIST; do
    etag=$(echo "$eps" | tr -d '.')
    gen "eps${etag}_${vtag}" "$vid" emognition "--fusion_gate_fixed $eps"
    gen "eps${etag}_${vtag}" "$vid" mdmer      "--fusion_gate_fixed $eps"
  done
done

# 2) AdaMAE self-loop — rest identical to base (learnable gate), both datasets.
gen "selfloop_adamae" AdaMAE emognition "--gcn_self_loop 1.0"
gen "selfloop_adamae" AdaMAE mdmer      "--gcn_self_loop 1.0"

# submit-all helper
cat > submit_all.sh <<'EOF'
#!/bin/bash
cd "$(dirname "$0")"
for f in sweep_*.slurm; do echo "sbatch $f"; sbatch "$f"; done
EOF
chmod +x submit_all.sh
echo "generated $(ls sweep_*.slurm | wc -l) slurm files + submit_all.sh"
