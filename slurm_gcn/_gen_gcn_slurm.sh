#!/bin/bash
# Generate GCN-fusion test slurm files (Jean Zay H100, 80GB/GPU).
# 3 GPUs x batch 4 = effective batch 12. Methods x {emognition, mdmer}.
# Run:  bash slurm_gcn/_gen_gcn_slurm.sh   then   bash slurm_gcn/submit_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p ../NJZ_logs/gcn
EMO_CSV=./datasets/updated_fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv
MD_CSV=./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold1.csv

BASE_FLAGS="--model vemt --vemt_video VideoMAE --fusion naive --eeg_signal --eeg_backbone cbramod \
--eeg_full_unfreeze --eeg_full_signal \
--dense_video_clips --frame_interval 4 --clip_overlap_ratio 0.5 --fps_normalize \
--dense_chunk_size 6 --dense_cache_features --video_unfreeze_last_n_blocks 1 --clip_pool attn \
--gcn --gcn_video_per_clip --gcn_temporal_adj --gcn_clip_pe \
--learning_rate 1e-4 --gcn_learning_rate 1e-4 --weight_decay 0.05 \
--epochs 100 --patience 40 --pretrained --checkpoint_dir ./checkpoints_clip \
--num_gpus 3 --batch_size 4"

gen() {  # $1=name $2=dataset $3=method_flags
  local name=$1 ds=$2 mflags=$3
  local csv gc fold
  if [ "$ds" = emognition ]; then csv=$EMO_CSV; gc=""; fold=fold0; else csv=$MD_CSV; gc="--grad_clip 0.5"; fold=fold1; fi
  cat > "gcn_${name}_${ds}.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name="gcn_${name}_${ds}"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=./NJZ_logs/gcn/gcn_${name}_${ds}.out
#SBATCH --error=./NJZ_logs/gcn/gcn_${name}_${ds}.err

module purge
module load gcc/11.3.1
module load miniforge/24.9.0
conda activate tsf-low

set -x
srun python -u runner.py \\
    ${BASE_FLAGS} ${gc} \\
    --dataset ${ds} --csv_file ${csv} \\
    ${mflags} \\
    --experiment_name gcn_${name}_videomae-cbramod_${ds}_${fold}
EOF
  echo "gcn_${name}_${ds}.slurm"
}

# base = constant-gate (fusion_gate_fixed) runs on our local machine.
# GAT/cross-attn build ON the constant-gate base (imbalance fixed) + fix cross-modal
# edges; MISA is a gate-free alternative (shared/private). Per-dataset gate favors
# the stronger unimodal: emognition -> EEG (video 0.35), mdmer -> video (0.65).
for ds in emognition mdmer; do
  if [ "$ds" = emognition ]; then GV=0.35; else GV=0.65; fi
  gen gat        "$ds" "--fusion_gat --fusion_gate_fixed $GV"
  gen crossattn  "$ds" "--fusion_crossattn --fusion_gate_fixed $GV"
  gen misa       "$ds" "--fusion_misa"
done

# submit-all helper
cat > submit_all.sh <<'EOF'
#!/bin/bash
cd "$(dirname "$0")"
for f in gcn_*.slurm; do echo "sbatch $f"; sbatch "$f"; done
EOF
chmod +x submit_all.sh
echo "generated $(ls gcn_*.slurm | wc -l) slurm files + submit_all.sh"
