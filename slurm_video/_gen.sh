#!/bin/bash
# Video-only baselines in the framework input setup: fi4, overlap 0.5, mean
# pooling (attention pooling is our contribution, so the baseline keeps mean).
# Fills the benchmark table across the three datasets.
#   EAV:          all 5 backbones
#   emognition:   all 5 backbones
#   mdmer(fold1): videomae, vivit, swin, tsf (adamae already done)
# lr: 1e-4/wd0.05 for the MAE backbones, 2e-5/wd0.02 for the transformers.
# Run: bash slurm_video/_gen.sh ; bash slurm_video/submit_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p ../NJZ_logs/video
EMO_CSV=./datasets/updated_fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv
MD_CSV=./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold1.csv
EAV_CSV=./datasets/updated_fold_csv_files/EAV_fold_csv/EAV_dataset_updated_fold0.csv

gen() {  # $1=video $2=dataset $3=lr $4=wd
  local vid=$1 ds=$2 lr=$3 wd=$4 vtag csv fold
  vtag=$(echo "$vid" | tr 'A-Z' 'a-z')
  case "$ds" in
    emognition) csv=$EMO_CSV; fold=fold0 ;;
    mdmer)      csv=$MD_CSV;  fold=fold1 ;;
    eav)        csv=$EAV_CSV; fold=fold0 ;;
  esac
  cat > "video_${vtag}_${ds}.slurm" <<SL
#!/bin/bash
#SBATCH --job-name="vid_${vtag}_${ds}"
#SBATCH -A vfj@h100
#SBATCH -C h100
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=10:00:00
#SBATCH --output=./NJZ_logs/video/video_${vtag}_${ds}.out
#SBATCH --error=./NJZ_logs/video/video_${vtag}_${ds}.err

module purge
module load gcc/11.3.1
module load miniforge/24.9.0
conda activate tsf-low

set -x
srun python -u runner.py \\
    --model vemt --vemt_video ${vid} --set_video_only --fusion naive \\
    --dense_video_clips --frame_interval 4 --clip_overlap_ratio 0.5 --clip_pool mean \\
    --dense_chunk_size 6 --dense_cache_features --video_unfreeze_last_n_blocks 1 --fps_normalize \\
    --learning_rate ${lr} --weight_decay ${wd} \\
    --epochs 100 --patience 15 --pretrained --checkpoint_dir ./checkpoints_clip \\
    --num_gpus 3 --batch_size 4 --dataset ${ds} --csv_file ${csv} \\
    --experiment_name dense_fi4-overlap_1block-mean_${vtag}_${ds}_${fold}_${lr}_${wd}
SL
  echo "video_${vtag}_${ds}.slurm"
}

# EAV: all 5
gen ViViT    eav 2e-5 0.02
gen TSF      eav 2e-5 0.02
gen Swin     eav 2e-5 0.02
gen VideoMAE eav 1e-4 0.05
gen AdaMAE   eav 1e-4 0.05
# emognition: all 5
gen ViViT    emognition 2e-5 0.02
gen TSF      emognition 2e-5 0.02
gen Swin     emognition 2e-5 0.02
gen VideoMAE emognition 1e-4 0.05
gen AdaMAE   emognition 1e-4 0.05
# mdmer fold1: videomae, vivit, swin, tsf (adamae already done)
gen VideoMAE mdmer 1e-4 0.05
gen ViViT    mdmer 2e-5 0.02
gen Swin     mdmer 2e-5 0.02
gen TSF      mdmer 2e-5 0.02

cat > submit_all.sh <<'SUB'
#!/bin/bash
cd "$(dirname "$0")"
for f in video_*.slurm; do echo "sbatch $f"; sbatch "$f"; done
SUB
chmod +x submit_all.sh
echo "generated $(ls video_*.slurm | wc -l) slurm files + submit_all.sh"
