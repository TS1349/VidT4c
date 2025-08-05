import argparse

def qos(gpu, debug): 
    if "a100" == gpu:
        if (debug):
            return "qos_gpu_a100-dev"
        else:
            return "qos_gpu_a100-t3"

    if "h100" == gpu:
        if (debug):
            return "qos_gpu_h100-dev"
        else:
            return "qos_gpu_h100-t3"
    else:
        raise ValueError(f"{gpu} isn't an appropriate gpu type")

def max_hour(gpu, debug):
    #returns maximal HH:MM:SS

    if "a100" == gpu:
        if (debug):
            return "2:00:00"
        else:
            return "20:00:00"

    if "h100" == gpu:
        if (debug):
            return "2:00:00"
        else:
            return "20:00:00"
    else:
        raise ValueError(f"{gpu} isn't an appropriate gpu type")

def cpu_per_task(gpu):
    #returns maximal HH:MM:SS

    if "a100" == gpu:
        return 8

    if "h100" == gpu:
        return 24
    else:
        raise ValueError(f"{gpu} isn't an appropriate gpu type")

def fold_name(dataset):
    if ("emognition" == dataset):
        return "Emognition"
    elif ("eav" == dataset):
        return "EAV"
    elif ("mdmer" == dataset):
        return "MDMER"
    else:
        raise ValueError(f"{dataset} isn't an appropriate dataset name")
def csv_location(dataset, fold):
    dataset = fold_name(dataset)
    return f"./datasets/updated_fold_csv_files/{dataset}_fold_csv/{dataset}_dataset_updated_fold{fold}.csv"

def fill_in(args):
    dataset = args.dataset
    fold = args.fold
    model = args.model
    gpu = args.gpu
    debug = args.debug
    num_gpus = args.num_gpus
    batch_size = args.batch_size
    return f'\
#!/bin/bash\n\n\
#SBATCH --job-name=\
"TM{(model[:2]).lower()}{(dataset[:2]).upper()}{fold}{gpu[0].upper()}"\n\
#SBATCH -A ntk@{gpu}\n\
#SBATCH -C {gpu}\n\
#SBATCH --gres=gpu:{num_gpus}\n\
#SBATCH --nodes=1\n\
#SBATCH --ntasks-per-node={num_gpus}\n\
#SBATCH --cpus-per-task={cpu_per_task(gpu)}\n\
#SBATCH --hint=nomultithread\n\
#SBATCH --qos={qos(gpu,debug)}\n\
#SBATCH --time={max_hour(gpu,debug)}\n\
#SBATCH --output={model}_{dataset}.out\n\
#SBATCH --error={model}_{dataset}.err\n\n\n\n\
module purge\n\
module load arch/{gpu}\n\
module load cuda/12.1.0\n\
module load cudnn/9.2.0.82-cuda\n\
module load gcc/11.3.1\n\
module load anaconda-py3/2024.06\n\n\n\n\
conda activate tsf4\n\n\
export MASTER_PORT=$((12000 + $RANDOM % 20000))\n\
export OMP_NUM_THREADS=1\n\n\
fold_csv="{csv_location(dataset, fold)}"\n\n\n\n\
set -x\n\
srun python -u ./runner.py \\\n\
        --epochs 200\\\n\
        --batch_size {batch_size}\\\n\
        --learning_rate 0.1\\\n\
        --weight_decay 0.001\\\n\
        --csv_file "$fold_csv"\\\n\
        --checkpoint_dir "./checkpoints"\\\n\
        --experiment_name "{model}_{dataset}_{fold}"\\\n\
        --dataset "{dataset}"\\\n\
        --model "{model}"\\\n\
        --pretrained \n\
'

if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="SlurmFileGenerator")


    parser.add_argument("--model",
                        type=str,
                        default="vivit",
                        choices=("vivit", "swin", "tsf", "hicmae", "tvlt"))

    parser.add_argument("--dataset",
                        type=str,
                        default="emognition",
                        choices=("eav", "mdmer", "emognition"))

    parser.add_argument("--fold",
                        type=int,
                        default=0,
                        choices=(0,1,2,3,4))

    parser.add_argument("--gpu",
                        type=str,
                        default="h100",
                        choices=("a100", "h100"))

    parser.add_argument("--debug",
                        type=bool,
                        default=False)

    parser.add_argument("--num_gpus",
                        type=int,
                        default=1)

    parser.add_argument("--batch_size",
                        type=int,
                        default=32)

    args = parser.parse_args()

    slurm_script = fill_in(args)

    file_name = f"./{'debug_' if args.debug else ''}{args.model}_{args.dataset}.slurm"
    with open(file_name, "w") as file:
        file.write(slurm_script)
