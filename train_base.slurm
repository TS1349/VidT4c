#!/bin/bash

#SBATCH --job-name="tmMDf0"

#SBATCH -A ntk@a100
#SBATCH -C a100
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --time=20:00:00

#SBATCH --output=mdmer_tsf_out.out
#SBATCH --error=mdmer_tsf_err.out

module purge
module load arch/a100
module load cuda/12.1.0
module load cudnn/9.2.0.82-cuda
module load gcc/11.3.1
module load anaconda-py3/2024.06

conda activate tsf4

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

fold_csv="./datasets/fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold0.csv"

set -x
srun python -u ./runner.py \
        --epochs 200\
        --batch_size 64\
        --learning_rate 0.01\
        --weight_decay 0.0004\
        --csv_file "$fold_csv"\
        --checkpoint_dir "./checkpoints"\
        --experiment_name "mdmr_fold_0"\
        --dataset "mdmer"\
        --model "tsf"
