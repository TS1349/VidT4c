#!/bin/bash
# deep_fuse (3-stage deep-supervised hierarchical fusion) for the video backbones
# NOT yet run locally (AdaMAE is run on our machine). VideoMAE x
# {emognition fold0, mdmer fold1}, CBraMod EEG. Jean Zay H100, 3 GPU x batch 4 = 12.
# Run: bash slurm_deepfuse/_gen_deepfuse_slurm.sh ; bash slurm_deepfuse/submit_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p ../NJZ_logs/deepfuse
EMO_CSV=./datasets/updated_fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv
MD_CSV=./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold1.csv

BASE_FLAGS="--model vemt --eeg_backbone cbramod --fusion naive --eeg_signal \
--eeg_full_unfreeze --eeg_full_signal \
--dense_video_clips --frame_interval 4 --clip_overlap_ratio 0.5 --fps_normalize \
--dense_chunk_size 6 --dense_cache_features --video_unfreeze_last_n_blocks 1 --clip_pool attn \
--gcn --gcn_video_per_clip --gcn_temporal_adj --gcn_clip_pe --deep_fuse \
--learning_rate 1e-4 --gcn_learning_rate 1e-4 --weight_decay 0.05 \
--epochs 100 --patience 25 --pretrained --checkpoint_dir ./checkpoints_clip \
--num_gpus 3 --batch_size 4"

gen() {  # $1=video $2=dataset
  local vid=$1 ds=$2 vtag csv gc fold
  vtag=$(echo "$vid" | tr 'A-Z' 'a-z')
  if [ "$ds" = emognition ]; then csv=$EMO_CSV; gc=""; fold=fold0; else csv=$MD_CSV; gc="--grad_clip 0.5"; fold=fold1; fi
  cat > "deepfuse_${vtag}_${ds}.slurm" <<SL
#!/bin/bash
#SBATCH --job-name="df_${vtag}_${ds}"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=./NJZ_logs/deepfuse/deepfuse_${vtag}_${ds}.out
#SBATCH --error=./NJZ_logs/deepfuse/deepfuse_${vtag}_${ds}.err

module purge
module load gcc/11.3.1
module load miniforge/24.9.0
conda activate tsf-low

set -x
srun python -u runner.py \\
    ${BASE_FLAGS} ${gc} \\
    --vemt_video ${vid} --dataset ${ds} --csv_file ${csv} \\
    --experiment_name deepfuse_${vtag}-cbramod_${ds}_${fold}
SL
  echo "deepfuse_${vtag}_${ds}.slurm"
}

for vid in VideoMAE; do
  for ds in emognition mdmer; do
    gen "$vid" "$ds"
  done
done

cat > submit_all.sh <<'SUB'
#!/bin/bash
cd "$(dirname "$0")"
for f in deepfuse_*.slurm; do echo "sbatch $f"; sbatch "$f"; done
SUB
chmod +x submit_all.sh
echo "generated $(ls deepfuse_*.slurm | wc -l) slurm files + submit_all.sh"
